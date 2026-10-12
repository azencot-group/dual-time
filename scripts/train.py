#!/usr/bin/env python3
"""Launch Stage A (DeepSpeed ZeRO-2) or Stage B (DeepSpeed ZeRO-3) training with accelerate.

    python scripts/train.py --stage A                   # configs/stageA_dual8b.yaml
    python scripts/train.py --stage B                   # configs/stageB_dual8b.yaml (needs stageA_dual8b/best)
    python scripts/train.py --stage B --resume          # continue from the newest checkpoint
    python scripts/train.py --stage B --nproc 4         # 4 GPUs: accumulation is re-derived (see below)
    python scripts/train.py --stage B --per-device-bs 4 # smaller GPUs: fewer examples per GPU, same global batch
    python scripts/train.py --stage B --attn sdpa       # no flash-attn
    python scripts/train.py --config configs/my_experiment.yaml
    python scripts/train.py --stage B --output-dir artifacts/checkpoints/myrun   # forwarded to src.train

Flags this launcher does not know are forwarded verbatim to `python -m
src.train` (--output-dir, --model-path, --resume-adapter, --dino-model-id,
--save-steps, --seed; see `python -m src.train --help`). Any other config key
is overridden with --set KEY=VALUE.

Batch geometry: global batch = per-device batch x gradient accumulation x
GPUs. The configs declare the global batch (`global_batch_size`), and the
launcher derives the accumulation steps for the GPU count and per-device batch
it is given, so the recipe is the same on any number or size of GPUs. The
resolved geometry is printed before the launch, and the run is teed to
`logs/train_<config>_<timestamp>.log`.

Stage A runs on ZeRO-2: it trains the DINO mergers full-weight, and ZeRO-3's
parameter hooks crash on them. Stage B (LM LoRA only) runs on ZeRO-3 with
reentrant gradient checkpointing, the environment of the reference run; the
same recipe under ZeRO-2 lands 1-3 points lower on the TSRBench reasoning tasks.
"""
from __future__ import annotations

import argparse
import datetime
import os
import random
import signal
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE_CONFIG = {"A": "configs/stageA_dual8b.yaml", "B": "configs/stageB_dual8b.yaml"}
STAGE_ACCEL = {"A": "configs/accelerate_z2.yaml", "B": "configs/accelerate_z3.yaml"}


def visible_gpus() -> int:
    """Number of visible GPUs: torch.cuda.device_count(), else `nvidia-smi -L`."""
    try:
        import torch  # imported lazily so --help works without torch
        n = torch.cuda.device_count()
        if n:
            return n
    except Exception:  # noqa: BLE001
        pass
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20).stdout
        return sum(1 for line in out.splitlines() if line.startswith("GPU "))
    except Exception:  # noqa: BLE001
        return 0


def user_path(p: str, cwd: str) -> str:
    """A path given on the command line: absolute as is; relative first
    against the caller's cwd (if it exists there), else against the repo."""
    if not p or os.path.isabs(p):
        return p
    cand = os.path.join(cwd, p)
    return os.path.abspath(cand) if os.path.exists(cand) else p


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scripts/train.py",
        description="Launch Stage A or Stage B training (accelerate + DeepSpeed) and tee the run to logs/.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,   # so a forwarded flag is never taken as an abbreviation of one of ours
        epilog="examples:\n"
               "  python scripts/train.py --stage A\n"
               "  python scripts/train.py --stage B --nproc 4                 # 8 x 2 x 4 = 64\n"
               "  python scripts/train.py --stage B --nproc 8 --per-device-bs 4   # 4 x 2 x 8 = 64\n"
               "  python scripts/train.py --stage A --nproc 1 --per-device-bs 2 --attn sdpa\n"
               "  python scripts/train.py --stage B --set max_length=6144\n"
               "  python scripts/train.py --stage B --output-dir artifacts/checkpoints/myrun "
               "--resume-adapter artifacts/checkpoints/stageA_dual8b/best\n"
               "\n"
               "Flags not listed above are forwarded verbatim to `python -m src.train`, which\n"
               "takes the locations as explicit flags (--output-dir, --model-path,\n"
               "--resume-adapter, --dino-model-id, --save-steps, --seed); see\n"
               "`python -m src.train --help`. The data recipe, the tower, LoRA and\n"
               "optimisation parameters live in the config; --set KEY=VALUE overrides any key.\n")
    what = p.add_argument_group("what to train")
    what.add_argument("--stage", choices=["A", "B"],
                      help="A: configs/stageA_dual8b.yaml (ZeRO-2); B: configs/stageB_dual8b.yaml (ZeRO-3).")
    what.add_argument("--config", metavar="PATH",
                      help="Training config YAML (instead of, or overriding, the --stage default).")
    what.add_argument("--accel", metavar="PATH",
                      help="accelerate config (default: configs/accelerate_z2.yaml for Stage A, "
                           "configs/accelerate_z3.yaml for Stage B; for --config alone: z3 if the "
                           "config has resume_adapter, else z2). --nproc is passed on the command "
                           "line, so the file's num_processes is never used.")
    geo = p.add_argument_group("GPUs and batch geometry")
    geo.add_argument("--nproc", type=int, metavar="N",
                     help="GPUs / processes to use (default: all visible GPUs).")
    geo.add_argument("--per-device-bs", type=int, metavar="N",
                     help="Examples per GPU per step (default: the config's per_device_train_batch_size).")
    geo.add_argument("--global-batch", type=int, metavar="N",
                     help="Examples per optimizer step (default: the config's global_batch_size). "
                          "Must be a multiple of PER_DEVICE_BS x NPROC.")
    geo.add_argument("--grad-accum", type=int, metavar="N",
                     help="Gradient accumulation steps; default: GLOBAL_BATCH / (PER_DEVICE_BS x NPROC), "
                          "or the config's gradient_accumulation_steps if it has no global_batch_size.")
    geo.add_argument("--attn", choices=["flash_attention_2", "sdpa"],
                     help="Attention implementation (default: the config's attn_implementation).")
    run = p.add_argument_group("run control")
    run.add_argument("--max-steps", type=int, metavar="N", help="Stop after N optimizer steps.")
    run.add_argument("--resume", action="store_true",
                     help="Continue from the newest intact checkpoint in output_dir (also after a preemption).")
    run.add_argument("--port", type=int, metavar="PORT",
                     help="accelerate main process port (default: random in 20000-59999).")
    run.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                     help="Override any config value (YAML-parsed); repeatable. Passed through to src.train.")
    return p


def main() -> int:
    args, forwarded = build_parser().parse_known_args()   # unknown flags go to src.train
    caller_cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    import yaml  # noqa: E402  (after --help so the help needs no third-party module)

    if not args.stage and not args.config:
        sys.exit("one of --stage A|B or --config PATH is required")
    config = user_path(args.config, caller_cwd) if args.config else STAGE_CONFIG[args.stage]
    if not os.path.isfile(config):
        sys.exit(f"config not found: {config}")
    with open(config) as f:
        cfg = yaml.safe_load(f)

    accel = user_path(args.accel, caller_cwd) if args.accel else None
    if not accel:
        if args.stage:
            accel = STAGE_ACCEL[args.stage]
        else:   # a custom config: ZeRO-3 unless it trains a Stage A (merger) prior
            accel = STAGE_ACCEL["B"] if cfg.get("resume_adapter") else STAGE_ACCEL["A"]
    if not os.path.isfile(accel):
        sys.exit(f"accelerate config not found: {accel}")

    nproc = args.nproc or visible_gpus()
    if nproc < 1:
        sys.exit("no GPU visible (pass --nproc N)")
    port = args.port or random.randint(20000, 59999)

    per_device_bs = args.per_device_bs or int(cfg["per_device_train_batch_size"])
    global_batch = args.global_batch or cfg.get("global_batch_size")
    if args.grad_accum:
        grad_accum = args.grad_accum
    elif global_batch:
        per_step = per_device_bs * nproc
        if int(global_batch) % per_step:
            sys.exit(f"global batch {global_batch} is not a multiple of "
                     f"per-device batch x GPUs = {per_device_bs} x {nproc} = {per_step}")
        grad_accum = int(global_batch) // per_step
    else:
        grad_accum = int(cfg["gradient_accumulation_steps"])

    sets = [f"per_device_train_batch_size={per_device_bs}", f"gradient_accumulation_steps={grad_accum}"]
    if args.attn:
        sets.append(f"attn_implementation={args.attn}")
    sets += args.set

    cmd = ["accelerate", "launch", "--config_file", accel,
           "--main_process_port", str(port), "--num_processes", str(nproc),
           "-m", "src.train", "--config", config]
    for kv in sets:
        cmd += ["--set", kv]
    if args.max_steps:
        cmd += ["--max-steps", str(args.max_steps)]
    if args.resume:
        cmd.append("--resume")
    cmd += forwarded

    Path("logs").mkdir(exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log = Path("logs") / f"train_{Path(config).stem}_{stamp}.log"
    banner = (f"[train] stage={args.stage or 'custom'} config={config} accel={accel} nproc={nproc} "
              f"batch: {per_device_bs}/GPU x {grad_accum} accumulation x {nproc} GPUs = "
              f"{per_device_bs * grad_accum * nproc} attn={args.attn or 'config'} "
              f"max_steps={args.max_steps or 'full'} resume={args.resume} port={port}\n"
              f"[train] log: {log}\n[train] $ {' '.join(cmd)}\n")
    return run_teed(cmd, log, banner)


def _workers_of(launcher_pid: int) -> list[int]:
    """PIDs of the training workers: the children of the accelerate launcher
    (its grandchildren are dataloader workers and must not be signalled)."""
    try:
        out = subprocess.run(["pgrep", "-P", str(launcher_pid)], capture_output=True, text=True).stdout
        return [int(x) for x in out.split()]
    except Exception:  # noqa: BLE001
        return []


def run_teed(cmd: list[str], log: Path, banner: str) -> int:
    """Run `cmd`, writing its combined stdout/stderr to the terminal and `log`.
    SIGTERM (a scheduler preemption) is forwarded to the training workers, not
    to the launcher, so src/preemption.py can checkpoint before exiting."""
    with open(log, "ab") as fh:
        for sink in (sys.stdout.buffer, fh):
            sink.write(banner.encode()); sink.flush()
        child = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)

        def on_term(signum, frame):   # noqa: ARG001
            targets = _workers_of(child.pid) or [child.pid]
            for pid in targets:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        signal.signal(signal.SIGTERM, on_term)

        for chunk in iter(lambda: child.stdout.readline(), b""):
            sys.stdout.buffer.write(chunk); sys.stdout.buffer.flush()
            fh.write(chunk); fh.flush()
        return child.wait()


if __name__ == "__main__":
    sys.exit(main())
