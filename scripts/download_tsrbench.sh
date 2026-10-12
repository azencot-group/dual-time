#!/usr/bin/env bash
# Fetch the TSRBench jsonl files (umd-zhou-lab/TSRBench).
#
#   bash scripts/download_tsrbench.sh                 # -> eval_data/tsrbench
#   bash scripts/download_tsrbench.sh --dest DIR      # (then pass --dataset-dir DIR to the evaluator)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE/.."
export HF_HUB_OFFLINE=0   # the download scripts always talk to the Hub
DEST="eval_data/tsrbench"
while [[ $# -gt 0 ]]; do case "$1" in
  --dest) DEST="$2"; shift 2 ;; *) echo "usage: $0 [--dest DIR]"; exit 2 ;;
esac; done
if [[ -f "$DEST/perception/perception.jsonl" ]]; then echo "[tsrbench] present: $DEST"; exit 0; fi
mkdir -p "$DEST"
python - "$DEST" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download("umd-zhou-lab/TSRBench", repo_type="dataset", local_dir=sys.argv[1],
                  allow_patterns=["*.jsonl", "README.md"])
print("[tsrbench] ->", sys.argv[1])
PY
