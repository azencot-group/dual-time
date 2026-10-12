"""Preemption-aware and disk-bounded checkpointing for the HF Trainer.

`PreemptionCallback`: preemptible schedulers (batch-system requeue, Kubernetes,
cloud spot instances) announce an eviction with SIGTERM and kill the process
some grace period later. The callback catches SIGTERM (and SIGUSR1) inside the
training loop and, at the next optimizer-step boundary, tells the Trainer to
write a FULL checkpoint (model + optimizer + scheduler + RNG, overriding
`save_only_model`) and stop cleanly. Relaunching with `--resume` continues
from that checkpoint. The stop decision is all-reduced across ranks every step
so every worker saves and stops on the same step; otherwise ranks that did not
receive the signal would deadlock on the next collective. A worker must finish
its in-flight step + save inside the scheduler's grace period.

`RollingCheckpointCallback`: keeps the newest N checkpoints resumable, strips
the checkpoint nearest each epoch boundary to an adapter-only snapshot, and
deletes the rest, so disk use stays bounded on long runs.
"""

from __future__ import annotations

import os
import shutil
import signal
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import TrainerCallback


class PreemptionCallback(TrainerCallback):
    """Save a full checkpoint and stop when a preemption signal arrives.

    Args:
        signals: signal numbers to treat as a preemption request. SIGTERM is
            what schedulers send on eviction; SIGUSR1 is also honoured so a run
            can be asked to checkpoint early by hand (`kill -USR1 <pid>`).
    """

    def __init__(self, signals=(signal.SIGTERM, signal.SIGUSR1)):
        self._flag = False
        self.preempted = False  # set once we have actually saved+stopped
        self._prev_handlers = {}
        for sig in signals:
            try:
                self._prev_handlers[sig] = signal.signal(sig, self._handle)
            except (ValueError, OSError):
                # signal() only works on the main thread of the main interpreter;
                # dataloader workers etc. just skip registration.
                pass

    def _handle(self, signum, frame):
        # Keep the handler tiny and async-signal-safe: just flip a flag. The
        # actual save happens at the next step boundary in on_step_end.
        self._flag = True
        print(f"[preemption] caught signal {signum} (rank "
              f"{os.environ.get('RANK', '0')}); will checkpoint at next step",
              flush=True)

    def _should_stop(self) -> bool:
        """True if ANY rank has flagged preemption (all-reduce max)."""
        local = 1 if self._flag else 0
        if dist.is_available() and dist.is_initialized():
            t = torch.tensor(
                [local],
                device=torch.device("cuda", torch.cuda.current_device())
                if torch.cuda.is_available() else "cpu",
            )
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            return bool(t.item())
        return bool(local)

    def on_step_end(self, args, state, control, **kwargs):
        if self._should_stop():
            # Force a resumable checkpoint even if the config saves model-only:
            # optimizer/scheduler/RNG are needed to resume mid-epoch.
            args.save_only_model = False
            control.should_save = True
            control.should_training_stop = True
            self.preempted = True
            if state.is_world_process_zero:
                print(f"[preemption] requesting checkpoint + stop at "
                      f"global_step={state.global_step}", flush=True)
        return control


# --- Rolling checkpoints: bound disk without losing resume or per-epoch eval ---

# Files/dirs that make a checkpoint *resumable* (optimizer/scheduler/RNG); an
# adapter-only snapshot drops these. Same set the train.py best/ finalize uses.
_RESUME_STATE_PREFIXES = ("optimizer", "scheduler", "rng_state",
                          "global_step", "latest")


def _rotation_plan(steps, steps_per_epoch, keep_full, epoch_stride):
    """Decide the fate of each existing checkpoint after a save.

    Args:
        steps: iterable of checkpoint step numbers currently on disk.
        steps_per_epoch: optimizer steps in one epoch (0 => unknown; keep all).
        keep_full: how many of the NEWEST checkpoints stay FULL (resumable).
        epoch_stride: keep an adapter-only snapshot at every Nth epoch boundary
            (1 = every epoch, 2 = every other, 0 = keep none / pure rolling).

    Returns (keep_full_set, strip_set, delete_set) of step numbers:
        keep_full_set  -> leave untouched (optimizer kept, resumable)
        strip_set      -> strip to adapter-only (durable, tiny, eval reads it)
        delete_set     -> remove entirely (intermediate, rolled away)

    The newest `keep_full` are always kept full, so get_last_checkpoint() (max
    step) is always a resumable dir -> no resume gap. For each reached epoch
    boundary we keep the checkpoint *closest* to it (not an exact multiple) as a
    durable adapter-only checkpoint-<step> -> matches how the eval side selects
    per-epoch checkpoints, and works even when save_steps doesn't divide
    steps/epoch. Everything else (intermediate) is deleted.
    """
    steps = sorted(set(int(s) for s in steps))
    if not steps:
        return set(), set(), set()
    keep_full_set = set(steps[-keep_full:]) if keep_full > 0 else set()
    # Per reached epoch boundary (e*spe <= max_step), snapshot the nearest
    # checkpoint (ties -> the later step). Recomputed every save, so an
    # in-progress epoch's pick settles once a save lands at/after the boundary.
    snapshots = set()
    if steps_per_epoch > 0 and epoch_stride > 0:
        max_step = steps[-1]
        e = epoch_stride
        while e * steps_per_epoch <= max_step:
            b = e * steps_per_epoch
            snapshots.add(min(steps, key=lambda s: (abs(s - b), -s)))
            e += epoch_stride
    strip, delete = set(), set()
    for s in steps:
        if s in keep_full_set:
            continue
        (strip if s in snapshots else delete).add(s)
    return keep_full_set, strip, delete


class RollingCheckpointCallback(TrainerCallback):
    """Bound checkpoint disk use while preserving crash-resume + per-epoch eval.

    Runs on rank 0 after every Trainer save and enforces `_rotation_plan`:
      * newest `keep_full` checkpoints stay FULL (optimizer/scheduler/RNG) so a
        crash resumes from the latest with no lost epochs and no resume gap;
      * the checkpoint nearest each epoch boundary is STRIPPED to adapter-only
        and kept forever (tiny) -> per-epoch eval still reads it as
        checkpoint-<step>, even if save_steps doesn't divide steps/epoch;
      * every other (intermediate) checkpoint is deleted.

    HF's own save_total_limit MUST be disabled (None) so this callback owns
    rotation and can tell epoch boundaries from intermediates. steps/epoch is
    inferred from max_steps/num_train_epochs at train start; if that can't be
    computed, rotation is disabled (keep-all) rather than risk losing adapters.
    """

    def __init__(self, output_dir, keep_full=2, keep_epoch_adapters=True):
        self.out = Path(output_dir)
        self.keep_full = max(1, int(keep_full))
        # YAML may hand us True/False (bool is a subclass of int -> check first).
        if keep_epoch_adapters is True:
            self.epoch_stride = 1
        elif not keep_epoch_adapters:            # False / None / 0
            self.epoch_stride = 0
        else:
            self.epoch_stride = max(1, int(keep_epoch_adapters))
        self.spe = 0        # steps/epoch, computed at train start
        self.enabled = True

    def on_train_begin(self, args, state, control, **kwargs):
        ne = float(getattr(args, "num_train_epochs", 0) or 0)
        if state.max_steps and ne > 0:
            self.spe = max(1, round(state.max_steps / ne))
        else:
            # Can't locate epoch boundaries -> don't risk deleting eval
            # adapters; fall back to keep-all (no rotation).
            self.enabled = False
            if state.is_world_process_zero:
                print("[rolling-ckpt] cannot infer steps/epoch; disabling "
                      "rotation (keeping all checkpoints)", flush=True)
            return control
        if state.is_world_process_zero:
            mode = (f"every {self.epoch_stride} epoch(s)" if self.epoch_stride
                    else "off (pure rolling)")
            print(f"[rolling-ckpt] steps/epoch~={self.spe}, keep_full="
                  f"{self.keep_full}, epoch adapters: {mode}", flush=True)
        return control

    def _strip_to_adapter(self, ckpt: Path):
        for child in list(ckpt.iterdir()):
            n = child.name
            if n.startswith(_RESUME_STATE_PREFIXES) or n.endswith(".distcp"):
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    try:
                        child.unlink()
                    except OSError:
                        pass

    def on_save(self, args, state, control, **kwargs):
        if not (self.enabled and state.is_world_process_zero):
            return control
        ckpts = {int(p.name.split("-")[-1]): p
                 for p in self.out.glob("checkpoint-*")
                 if p.name.split("-")[-1].isdigit()}
        keep_full, strip, delete = _rotation_plan(
            ckpts.keys(), self.spe, self.keep_full, self.epoch_stride)
        for s in strip:
            self._strip_to_adapter(ckpts[s])
        for s in delete:
            shutil.rmtree(ckpts[s], ignore_errors=True)
        if strip or delete:
            print(f"[rolling-ckpt] step {state.global_step}: full="
                  f"{sorted(keep_full)} adapter-only={sorted(strip)} "
                  f"deleted={sorted(delete)}", flush=True)
        return control
