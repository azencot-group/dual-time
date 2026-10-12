#!/bin/bash
# Training job (Stage A or B) for a preemptible Slurm partition, submitted by
# scripts/slurm/submit_train.sh (which exports REPO, VENV, TRAIN_ARGS and
# TRAIN_FRESH). The whole body runs inside the container when CONTAINER_IMAGE
# is set in scripts/slurm/slurm.env.
#
# Preemption: the scheduler sends SIGTERM, then SIGKILL after a grace period,
# and requeues the job. The trap below forwards SIGTERM to the TRAINING WORKERS
# (children of the accelerate launcher, not the launcher itself: torch elastic
# would kill the workers instantly and no checkpoint would be written);
# src/preemption.py then writes a full checkpoint at the next step and exits.
# The requeued job resumes from it because --resume is on by default
# (submit_train.sh --fresh forces a from-scratch run).
#SBATCH --job-name=train
#SBATCH --open-mode=append
set -euo pipefail
cd "${REPO:?set REPO to the repository root}"
if [[ -n "${VENV:-}" ]]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
# shellcheck disable=SC2206
ARGS=(${TRAIN_ARGS:-})
echo "[job $SLURM_JOB_ID] $(hostname) args: ${ARGS[*]} restart=${SLURM_RESTART_COUNT:-0} venv=${VENV:-PATH}"
nvidia-smi -L || true
[[ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]] && echo "[job] requeue #${SLURM_RESTART_COUNT} detected -> resuming"
RESUME="--resume"; [[ "${TRAIN_FRESH:-0}" == "1" ]] && { RESUME=""; echo "[job] --fresh -> from scratch"; }
# Chunked runs (submit_train.sh --chunks N): every chunk is a full job that
# resumes the previous one; once training has finished (best/ exists, written
# only at the very end) the remaining chunks exit immediately.
OUT_DIR=$(python - "${ARGS[@]}" <<'PY'
import sys, yaml
from src import paths
argv = sys.argv[1:]
cfg_path = out_dir = None
for i, a in enumerate(argv):
    if a == "--config": cfg_path = argv[i + 1]
    elif a == "--stage" and cfg_path is None: cfg_path = f"configs/stage{argv[i + 1]}_dual8b.yaml"
    elif a == "--output-dir": out_dir = argv[i + 1]          # forwarded to src.train
    elif a == "--set" and argv[i + 1].startswith("output_dir="): out_dir = argv[i + 1].split("=", 1)[1]
print(paths.resolve(out_dir or yaml.safe_load(open(cfg_path))["output_dir"]))
PY
)
if [[ -f "$OUT_DIR/best/adapter_config.json" || -f "$OUT_DIR/best/merger_weights.pt" ]]; then
  echo "[job $SLURM_JOB_ID] $OUT_DIR/best exists -> training already complete"; exit 0
fi

descendants() { local p; for p in $(pgrep -P "$1" 2>/dev/null); do echo "$p"; descendants "$p"; done; }
forward_term() {
  # The rank workers are the CHILDREN of the `accelerate launch` process (its
  # own children's children are dataloader workers, which must not be
  # signalled). $PID is the scripts/train.py launcher, so walk the tree to the
  # accelerate process first. Signalling the launcher itself would make
  # torch-elastic kill the workers at once and no checkpoint would be written.
  LAUNCHER=""
  for p in $(descendants "$PID"); do
    if tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q "accelerate launch"; then LAUNCHER="$p"; break; fi
  done
  WORKERS=""; [[ -n "$LAUNCHER" ]] && WORKERS=$(pgrep -P "$LAUNCHER" 2>/dev/null | tr '\n' ' ')
  echo "[job $SLURM_JOB_ID] SIGTERM received; forwarding to training workers [$WORKERS] (accelerate PID=${LAUNCHER:-?}, launcher PID=$PID)"
  if [[ -n "$WORKERS" ]]; then kill -TERM $WORKERS 2>/dev/null || true; else kill -TERM "$PID" 2>/dev/null || true; fi
  wait "$PID" || true
  echo "[job $SLURM_JOB_ID] training exited after SIGTERM; leaving for requeue."
  exit 0
}
trap forward_term SIGTERM

NPROC_OPT=(); [[ " ${ARGS[*]} " == *" --nproc "* ]] || NPROC_OPT=(--nproc "${SLURM_GPUS_ON_NODE:-1}")
python scripts/train.py "${ARGS[@]}" "${NPROC_OPT[@]}" --port $(( 20000 + (SLURM_JOB_ID % 40000) )) $RESUME &
PID=$!
wait "$PID"
echo "[job $SLURM_JOB_ID] done."
