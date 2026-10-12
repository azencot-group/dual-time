#!/usr/bin/env bash
# Fetch the ChatTS alignment evaluation sets (Zenodo record 14349206).
#
#   bash scripts/download_chatts_eval.sh              # -> eval_data/chatts_eval/dataset_{a,b}.json
#   bash scripts/download_chatts_eval.sh --dest DIR   # (then pass --data-dir DIR to the evaluator)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE/.."
DEST="eval_data/chatts_eval"
while [[ $# -gt 0 ]]; do case "$1" in
  --dest) DEST="$2"; shift 2 ;; *) echo "usage: $0 [--dest DIR]"; exit 2 ;;
esac; done
mkdir -p "$DEST"
for n in a b; do
    if [[ -f "$DEST/dataset_$n.json" ]]; then echo "[chatts] present: $DEST/dataset_$n.json"; continue; fi
    echo "[chatts] dataset_$n.json -> $DEST"
    curl -L -o "$DEST/dataset_$n.json" "https://zenodo.org/api/records/14349206/files/dataset_$n.json/content"
done
