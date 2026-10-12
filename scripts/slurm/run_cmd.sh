#!/bin/bash
# Job body for scripts/slurm/submit_cmd.sh: run $JOB_CMD from the repo root
# with $VENV activated. (sbatch --wrap would run under /bin/sh, which has no
# `source`; a real bash script avoids that and all quoting trouble.)
set -euo pipefail
cd "${REPO:?set REPO to the repository root}"
if [[ -n "${VENV:-}" ]]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi
export TOKENIZERS_PARALLELISM=false
echo "[job ${SLURM_JOB_ID:-?}] $(hostname) venv=${VENV:-PATH} cmd: ${JOB_CMD:?}"
eval "$JOB_CMD"
