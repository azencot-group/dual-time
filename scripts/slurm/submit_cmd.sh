#!/usr/bin/env bash
# Run one shell command as a Slurm job from the repo root (evaluations, data
# downloads, checks, ...).
#
#   scripts/slurm/submit_cmd.sh -n NAME [-g GPUS] [-t HH:MM:SS] [-m MEM] [-c CPUS]
#                               [-d <dependency>] [--test-only] -- 'python scripts/check_env.py'
#
# Site settings (account, partition, QOS, container, venv) come from
# scripts/slurm/slurm.env. The command is handed to the job through JOB_CMD
# and executed by run_cmd.sh. Pitfalls baked in: --container-image is passed
# on the CLI (an #SBATCH directive truncates at '#'); plain --export=ALL with
# the variables exported in the shell (the --export=ALL,VAR=x form breaks
# launch); a multi-GPU job carries an explicit --mem.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
# shellcheck disable=SC1091
source "$HERE/slurm.env"
NAME=""; GPUS=1; TIME="01:00:00"; MEM="64G"; CPUS=8; DEP=""; TEST_ONLY=""
while [[ $# -gt 0 ]]; do case "$1" in
  -n) NAME="$2"; shift 2 ;;  -g) GPUS="$2"; shift 2 ;;  -t) TIME="$2"; shift 2 ;;
  -m) MEM="$2"; shift 2 ;;   -c) CPUS="$2"; shift 2 ;;  -d) DEP="$2"; shift 2 ;;
  --account) SLURM_ACCOUNT="$2"; shift 2 ;; --partition) SLURM_PARTITION="$2"; shift 2 ;;
  --qos) SLURM_QOS="$2"; shift 2 ;; --test-only) TEST_ONLY="--test-only"; shift ;;
  --) shift; break ;;  *) echo "unknown arg $1"; exit 2 ;;
esac; done
[[ -n "$NAME" && $# -gt 0 ]] || { echo "usage: submit_cmd.sh -n NAME [opts] -- '<command>'"; exit 2; }
cd "$REPO"; mkdir -p "$LOG_DIR"
export REPO VENV JOB_CMD="$*"
OPTS=()
[[ -n "$SLURM_ACCOUNT" ]]   && OPTS+=(--account="$SLURM_ACCOUNT")
[[ -n "$SLURM_PARTITION" ]] && OPTS+=(--partition="$SLURM_PARTITION")
[[ -n "$SLURM_QOS" ]]       && OPTS+=(--qos="$SLURM_QOS")
[[ -n "$CONTAINER_IMAGE" ]] && OPTS+=(--container-image="$CONTAINER_IMAGE")
[[ -n "$CONTAINER_MOUNTS" ]] && OPTS+=(--container-mounts="$CONTAINER_MOUNTS")
DEPA=(); [[ -n "$DEP" ]] && DEPA=(--dependency="$DEP")
set -x
sbatch $TEST_ONLY --parsable --job-name="$NAME" "${OPTS[@]}" \
  --gres="gpu:$GPUS" --cpus-per-task="$CPUS" --mem="$MEM" --time="$TIME" \
  --output="$LOG_DIR/%x-%j.out" --error="$LOG_DIR/%x-%j.err" "${DEPA[@]}" --export=ALL \
  "$HERE/run_cmd.sh"
