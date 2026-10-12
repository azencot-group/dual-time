#!/usr/bin/env bash
# Fetch the TimeSeriesExam benchmark (AutonLab/TimeSeriesExam1, the revision
# pinned in src/evals/tsexam/data.py) into the Hugging Face datasets cache
# (set HF_HOME to relocate the cache).
#
#   bash scripts/download_tsexam.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE/.."
export HF_HUB_OFFLINE=0   # the download scripts always talk to the Hub
[[ $# -eq 0 ]] || { echo "usage: $0  (no options; the dataset goes to the HF cache)"; exit 2; }
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python - <<'PY'
from datasets import load_dataset
from src.evals.tsexam.data import DATASET_NAME, BENCHMARK_REVISION
ds = load_dataset(DATASET_NAME, revision=BENCHMARK_REVISION)
print("[tsexam]", DATASET_NAME, "cached:", {k: len(v) for k, v in ds.items()})
PY
