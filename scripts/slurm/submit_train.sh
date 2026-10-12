#!/usr/bin/env bash
# Submit a training stage as a preemptible Slurm job (requeue + resume).
#
#   scripts/slurm/submit_train.sh A --gpus 2 --time 20:00:00     # Stage A
#   scripts/slurm/submit_train.sh B --dep afterok:<JOBID>        # Stage B after Stage A
#   scripts/slurm/submit_train.sh custom --config configs/x.yaml
#
# Job options: --gpus N (2) --time HH:MM:SS (A: 05:00:00, B: 30:00:00, custom: 12:00:00)
#              --mem MEM (150G) --cpus N (16) --dep <dependency> --job-name NAME --test-only
#              --account/--partition/--qos (override scripts/slurm/slurm.env)
#              --chunks N   FALLBACK for queues that only backfill short jobs: split
#                           the run into N jobs of --time each, chained with afterany;
#                           each chunk gets SIGTERM 10 min before its limit,
#                           checkpoints, exits, and the next chunk resumes; surplus
#                           chunks exit at once. Default (no --chunks) = ONE job with
#                           preemption resume (--requeue + SIGTERM trap + --resume).
#              --fresh      start from scratch instead of resuming output_dir
# Training options, forwarded to `python scripts/train.py` inside the job
# (see `python scripts/train.py --help`): --config PATH --accel PATH --nproc N
# (default: the job's GPUs) --per-device-bs N --global-batch N --grad-accum N
# --attn flash_attention_2|sdpa --max-steps N --set KEY=VALUE (repeatable),
# and the src.train location flags --output-dir --model-path --resume-adapter
# --dino-model-id --save-steps --seed (see `python -m src.train --help`).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
# shellcheck disable=SC1091
source "$HERE/slurm.env"
STAGE="${1:?A|B|custom}"; shift || true
GPUS=2; TIME=""; MEM="150G"; CPUS=16; DEP=""; TEST_ONLY=""; CHUNKS=1; JOB=""; FRESH=0
TRAIN_ARGS=(); CONFIG=""
while [[ $# -gt 0 ]]; do case "$1" in
  --gpus) GPUS="$2"; shift 2 ;; --time) TIME="$2"; shift 2 ;; --mem) MEM="$2"; shift 2 ;;
  --cpus) CPUS="$2"; shift 2 ;; --dep) DEP="$2"; shift 2 ;; --job-name) JOB="$2"; shift 2 ;;
  --test-only) TEST_ONLY="--test-only"; shift ;; --chunks) CHUNKS="$2"; shift 2 ;;
  --fresh) FRESH=1; shift ;;
  --account) SLURM_ACCOUNT="$2"; shift 2 ;; --partition) SLURM_PARTITION="$2"; shift 2 ;;
  --qos) SLURM_QOS="$2"; shift 2 ;;
  --config) CONFIG="$2"; TRAIN_ARGS+=(--config "$2"); shift 2 ;;
  --accel|--nproc|--per-device-bs|--global-batch|--grad-accum|--attn|--max-steps|--set|\
  --output-dir|--model-path|--resume-adapter|--dino-model-id|--save-steps|--seed)
    TRAIN_ARGS+=("$1" "$2"); shift 2 ;;
  *) echo "unknown arg $1"; exit 2 ;;
esac; done
case "$STAGE" in
  A|a) TRAIN_ARGS=(--stage A "${TRAIN_ARGS[@]}"); TIME="${TIME:-05:00:00}"; JOB="${JOB:-train-stageA}" ;;
  B|b) TRAIN_ARGS=(--stage B "${TRAIN_ARGS[@]}"); TIME="${TIME:-30:00:00}"; JOB="${JOB:-train-stageB}" ;;
  custom) [[ -n "$CONFIG" ]] || { echo "custom needs --config PATH"; exit 2; }
          TIME="${TIME:-12:00:00}"; JOB="${JOB:-train-custom}" ;;
  *) echo "stage must be A, B or custom"; exit 2 ;;
esac
cd "$REPO"; mkdir -p "$LOG_DIR"
# The job body reads these (train.sbatch): the repo, the venv, the launcher's
# arguments (one string, word-split by the job) and the fresh/resume choice.
export REPO VENV TRAIN_FRESH="$FRESH"
export TRAIN_ARGS="${TRAIN_ARGS[*]}"
OPTS=()
[[ -n "$SLURM_ACCOUNT" ]]   && OPTS+=(--account="$SLURM_ACCOUNT")
[[ -n "$SLURM_PARTITION" ]] && OPTS+=(--partition="$SLURM_PARTITION")
[[ -n "$SLURM_QOS" ]]       && OPTS+=(--qos="$SLURM_QOS")
[[ -n "$CONTAINER_IMAGE" ]] && OPTS+=(--container-image="$CONTAINER_IMAGE")
[[ -n "$CONTAINER_MOUNTS" ]] && OPTS+=(--container-mounts="$CONTAINER_MOUNTS")
DEPA=(); [[ -n "$DEP" ]] && DEPA=(--dependency="$DEP")
# SIGTERM to the batch shell 10 min before the time limit (and on preemption
# the scheduler sends it anyway): train.sbatch forwards it to the workers,
# which checkpoint and exit; --requeue / the next chunk / a resubmission
# resumes from that checkpoint.
SIG=(--signal=B:TERM@600)
PREV=""
for (( c = 1; c <= CHUNKS; c++ )); do
  [[ -n "$PREV" ]] && DEPA=(--dependency="afterany:$PREV")
  [[ "$CHUNKS" -gt 1 ]] && JN="$JOB-c$c" || JN="$JOB"
  set -x
  JID=$(sbatch $TEST_ONLY --parsable --job-name="$JN" "${OPTS[@]}" \
    --gres="gpu:$GPUS" --cpus-per-task="$CPUS" --mem="$MEM" --time="$TIME" --requeue "${SIG[@]}" \
    --output="$LOG_DIR/%x-%j.out" --error="$LOG_DIR/%x-%j.err" "${DEPA[@]}" --export=ALL \
    "$HERE/train.sbatch")
  set +x
  echo "$JID"; PREV="$JID"
done
