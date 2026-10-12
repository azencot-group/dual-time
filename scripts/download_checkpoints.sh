#!/usr/bin/env bash
# Fetch the released adapters (Stage A prior + Stage B LoRA) from the Hub.
#
#   bash scripts/download_checkpoints.sh                 # -> artifacts/checkpoints/DualTime-8B
#   bash scripts/download_checkpoints.sh --dest DIR
#
# The repository is private until the public release: `hf auth login` with an
# account that has access. Never print or commit a token.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE/.."
export HF_HUB_OFFLINE=0   # the download scripts always talk to the Hub
HF_REPO="SequentialLearning/DualTime-8B"; DEST="artifacts/checkpoints/DualTime-8B"
while [[ $# -gt 0 ]]; do case "$1" in
  --dest) DEST="$2"; shift 2 ;; --repo) HF_REPO="$2"; shift 2 ;;
  *) echo "usage: $0 [--dest DIR] [--repo OWNER/NAME]"; exit 2 ;;
esac; done
command -v hf >/dev/null || { echo "hf CLI missing: install the environment first (README.md)"; exit 1; }
mkdir -p "$DEST"
hf download "$HF_REPO" --local-dir "$DEST"
echo "[checkpoints] $HF_REPO -> $DEST"; ls "$DEST"
