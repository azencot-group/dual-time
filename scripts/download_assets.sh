#!/usr/bin/env bash
# Download everything the pipeline needs: the base model (artifacts/models),
# DINOv3, CaTS-Bench and ChatTS ift/sft (Hugging Face cache; set HF_HOME to
# relocate it), the in-house training corpora (data/) and the benchmark data
# (eval_data/).
#
#   bash scripts/download_assets.sh                       # everything
#   bash scripts/download_assets.sh --skip-base           # skip the 17 GB base model
#   bash scripts/download_assets.sh --models-dir DIR --data-dir DIR --eval-dir DIR
#
# Needs network + the Python environment (hf CLI). The DINOv3 repo is gated:
# accept Meta's DINOv3 licence on its Hub page once and `hf auth login`
# (README.md); the other repos are public. Never print or commit a token.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE/.."
export HF_HUB_OFFLINE=0   # the download scripts always talk to the Hub
MODELS_DIR="artifacts/models"; DATA_DIR="data"; EVAL_DIR="eval_data"; SKIP_BASE=0
DATA_REPO="SequentialLearning/DualTime-data"; CHATTS_SUBSETS="ift sft"
while [[ $# -gt 0 ]]; do case "$1" in
  --models-dir) MODELS_DIR="$2"; shift 2 ;; --data-dir) DATA_DIR="$2"; shift 2 ;;
  --eval-dir) EVAL_DIR="$2"; shift 2 ;;   --skip-base) SKIP_BASE=1; shift ;;
  *) echo "usage: $0 [--models-dir DIR] [--data-dir DIR] [--eval-dir DIR] [--skip-base]"; exit 2 ;;
esac; done
command -v hf >/dev/null || { echo "hf CLI missing: install the environment first (README.md)"; exit 1; }
echo "[assets] models=$MODELS_DIR  data=$DATA_DIR  eval_data=$EVAL_DIR  (HF cache: default, or HF_HOME)"
mkdir -p "$MODELS_DIR" "$DATA_DIR" "$EVAL_DIR"

# 1) Base model (the configs point at artifacts/models/Qwen3-VL-8B-Instruct).
if [[ "$SKIP_BASE" != "1" ]]; then
    M="$MODELS_DIR/Qwen3-VL-8B-Instruct"
    if [[ -f "$M/config.json" ]]; then echo "[assets] base model present: $M"
    else echo "[assets] Qwen3-VL-8B-Instruct -> $M"; hf download Qwen/Qwen3-VL-8B-Instruct --local-dir "$M"; fi
fi
# 2) DINOv3 ViT-L/16 delay tower (Meta's official repo, gated; HF cache).
echo "[assets] DINOv3 ViT-L/16"; hf download facebook/dinov3-vitl16-pretrain-lvd1689m >/dev/null
# 3) ChatTS training subsets (Stage B: ift + sft; HF cache).
for sub in $CHATTS_SUBSETS; do
    echo "[assets] ChatTS subset: $sub"
    hf download ChatTSRepo/ChatTS-Training-Dataset --repo-type dataset --include "${sub}/*" >/dev/null
done
# 4) CaTS-Bench captions (Stage A; HF cache).
echo "[assets] CaTS-Bench"; hf download mhfisher/CaTSBench --repo-type dataset >/dev/null
# 5) Benchmark data (the per-benchmark scripts are idempotent).
bash scripts/download_tsexam.sh
bash scripts/download_tsrbench.sh --dest "$EVAL_DIR/tsrbench"
bash scripts/download_chatts_eval.sh --dest "$EVAL_DIR/chatts_eval"

# 6) The in-house training corpora (Hub dataset) -> $DATA_DIR/stage{A,B}.
missing=0
for rel in stageA/chatts_style_captions.jsonl stageA/structural_captions.json \
           stageB/tsexam_perception.json stageB/numerical_corpus.json stageB/reasoning_corpus.json; do
    [[ -f "$DATA_DIR/$rel" ]] || missing=$((missing+1))
done
if (( missing == 0 )); then echo "[assets] training corpora present: $DATA_DIR"
else echo "[assets] $DATA_REPO -> $DATA_DIR"; hf download "$DATA_REPO" --repo-type dataset --local-dir "$DATA_DIR"; fi
echo "[assets] DONE"
