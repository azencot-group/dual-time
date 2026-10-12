#!/usr/bin/env python3
"""Evaluate a Stage A + Stage B checkpoint pair on the benchmarks.

    python scripts/eval.py --kind tsexam    # TimeSeriesExam1 MCQ (746 items, per category) -> tsexam.json
    python scripts/eval.py --kind tsrb      # TSRBench, 12 tasks in one run, per-task protocol -> tsrbench.json
    python scripts/eval.py --kind chatts    # ChatTS alignment A + B (official scorer) -> chatts_eval/summary_<label>.json
    python scripts/eval.py --kind all       # tsexam, tsrb, chatts in turn

Every evaluator applies the Stage A prior and then the Stage B adapter on top
of the base model, runs one process per GPU (`torchrun`, items sharded across
ranks) and writes under `<results-dir>/<label>/`; the run is teed to
`logs/eval_<label>_<kind>_<timestamp>.log`. Protocols and reference numbers:
README.md.
"""
from __future__ import annotations

import argparse
import datetime
import os
import random
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "artifacts/models/Qwen3-VL-8B-Instruct"
DEFAULT_STAGE_A = "artifacts/checkpoints/stageA_dual8b/best"
DEFAULT_STAGE_B = "artifacts/checkpoints/stageB_dual8b/checkpoint-5714"
KINDS = ["tsexam", "tsrb", "chatts"]


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
        prog="scripts/eval.py",
        description="Run one benchmark (or all three) on a Stage A + Stage B checkpoint pair.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n"
               "  python scripts/eval.py --kind all\n"
               "  python scripts/eval.py --kind tsrb --limit 4                 # quick check\n"
               "  python scripts/eval.py --kind all \\\n"
               "      --stage-a artifacts/checkpoints/DualTime-8B/stageA/best \\\n"
               "      --stage-b artifacts/checkpoints/DualTime-8B/stageB/checkpoint-5714 --label DualTime-8B\n")
    p.add_argument("--kind", required=True, choices=KINDS + ["all"],
                   help="tsexam | tsrb (TSRBench) | chatts | all (the three in turn).")
    ck = p.add_argument_group("checkpoints")
    ck.add_argument("--stage-a", default=DEFAULT_STAGE_A, metavar="DIR",
                    help=f"Stage A prior adapter dir (default: {DEFAULT_STAGE_A}).")
    ck.add_argument("--stage-b", default=DEFAULT_STAGE_B, metavar="DIR",
                    help=f"Stage B adapter dir (default: {DEFAULT_STAGE_B}); \"\" = Stage A only.")
    ck.add_argument("--no-stage-b", action="store_true", help="Evaluate the Stage A prior alone.")
    ck.add_argument("--model", default=DEFAULT_MODEL, metavar="DIR",
                    help=f"Base model dir or Hub id (default: {DEFAULT_MODEL}).")
    ck.add_argument("--label", metavar="NAME",
                    help="Results go to <results-dir>/<label>/ (default: the name of the Stage B dir's "
                         "parent, e.g. stageB_dual8b; of the Stage A dir's parent without Stage B).")
    ck.add_argument("--results-dir", default="results", metavar="DIR",
                    help="Root of the results tree (default: results).")
    run = p.add_argument_group("run control")
    run.add_argument("--nproc", type=int, metavar="N", help="GPUs to use (default: all visible).")
    run.add_argument("--limit", type=int, metavar="N",
                     help="Evaluate only N items (per task for TSRBench, per dataset for ChatTS): a quick check.")
    run.add_argument("--port", type=int, metavar="PORT",
                     help="torchrun master port (default: random in 20000-59999).")
    proto = p.add_argument_group("protocol")
    proto.add_argument("--chart-style", default="v2",
                       help="Chart renderer; must match the training config (default: v2).")
    proto.add_argument("--deterministic", dest="deterministic", action="store_true", default=None,
                       help="Bit-exact kernels (~2x slower attention). Default: the evaluator's own "
                            "default, on for TSRBench, off for TSExam and ChatTS.")
    proto.add_argument("--no-deterministic", dest="deterministic", action="store_false",
                       help="Fast, non-bit-exact kernels.")
    proto.add_argument("--lenient-enumeration", action="store_true",
                       help="ChatTS only: score with '(N)' enumerations rewritten to 'N.' (a leniency "
                            "beyond the authors' protocol; can only raise a score). Off = upstream parity.")
    return p


def evaluator_cmd(kind: str, args, out_dir: Path, label: str, adapters: list[str]) -> list[str]:
    """The `-m src.evals.<kind>.eval ...` part of one run."""
    common = ["--model", args.model] + adapters + ["--chart-style", args.chart_style]
    if args.deterministic is not None:
        common.append("--deterministic" if args.deterministic else "--no-deterministic")
    if kind == "tsexam":
        cmd = ["-m", "src.evals.tsexam.eval", *common, "--max-new-tokens", "1024",
               "--out", str(out_dir / "tsexam.json")]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
    elif kind == "tsrb":
        # All 12 tasks in one run, each under its reported protocol (README.md):
        # ten tasks inline at 8k (= training), forecasting with plain-text
        # candidates at 16k, abductive reasoning with the game context at 16k.
        cmd = ["-m", "src.evals.tsrbench.eval", *common, "--prompt-style", "qa",
               "--max-new-tokens", "1024", "--out", str(out_dir / "tsrbench.json")]
        if args.limit:
            cmd += ["--limit-per-task", str(args.limit)]
    elif kind == "chatts":
        # One process per GPU; items are sharded and gathered on rank 0.
        cmd = ["-m", "src.evals.chatts.eval", "--dataset", "both", *common,
               "--exp-label", label, "--out-root", str(out_dir / "chatts_eval")]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        if args.lenient_enumeration:
            cmd.append("--lenient-enumeration")
    else:
        raise ValueError(kind)
    return cmd


def run_teed(cmd: list[str], log: Path, banner: str) -> int:
    """Run `cmd`, writing its combined stdout/stderr to the terminal and `log`."""
    with open(log, "ab") as fh:
        for sink in (sys.stdout.buffer, fh):
            sink.write(banner.encode()); sink.flush()
        child = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
        for chunk in iter(lambda: child.stdout.readline(), b""):
            sys.stdout.buffer.write(chunk); sys.stdout.buffer.flush()
            fh.write(chunk); fh.flush()
        return child.wait()


def main() -> int:
    args = build_parser().parse_args()
    caller_cwd = os.getcwd()
    os.chdir(REPO_ROOT)

    stage_a = user_path(args.stage_a, caller_cwd)
    stage_b = "" if args.no_stage_b else user_path(args.stage_b, caller_cwd)
    model = user_path(args.model, caller_cwd)
    results_dir = user_path(args.results_dir, caller_cwd) or "results"
    label = args.label or Path(stage_b or stage_a).resolve().parent.name
    out_dir = Path(results_dir) / label
    out_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    nproc = args.nproc or visible_gpus()
    if nproc < 1:
        sys.exit("no GPU visible (pass --nproc N)")
    adapters = ["--prior-adapter", stage_a]
    if stage_b:
        adapters += ["--adapter", stage_b]

    kinds = KINDS if args.kind == "all" else [args.kind]
    for kind in kinds:
        port = args.port or random.randint(20000, 59999)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log = Path("logs") / f"eval_{label}_{kind}_{stamp}.log"
        cmd = ["torchrun", "--standalone", "--nproc_per_node", str(nproc), "--master_port", str(port)]
        cmd += evaluator_cmd(kind, args, out_dir, label, adapters)
        banner = (f"[eval] kind={kind} nproc={nproc} limit={args.limit or 'full'} "
                  f"A={stage_a} B={stage_b or 'none'} -> {out_dir}\n"
                  f"[eval] log: {log}\n[eval] $ {' '.join(cmd)}\n")
        rc = run_teed(cmd, log, banner)
        if rc != 0:
            print(f"[eval] {kind} FAILED with exit code {rc} (see {log})", flush=True)
            return rc
        print(f"[eval] {kind} done: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
