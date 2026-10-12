#!/usr/bin/env bash
# Start an interactive container with all GPUs and the repo at /workspace:
#   bash docker/run.sh                                   # shell
#   bash docker/run.sh --artifacts /data/dual-time            # mount DIR at /workspace/artifacts
#   bash docker/run.sh -- python scripts/train.py --stage A   # run a command
# Options: --artifacts DIR (default: the repo's own artifacts/), --image TAG
# (default dual-time:latest). HF_TOKEN / HF_HOME / HF_HUB_OFFLINE are passed
# through when set; with HF_HOME set, that directory is mounted at the same
# path so the cache is shared with the host.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/.." && pwd)"
ARTIFACTS="$REPO/artifacts"; TAG="dual-time:latest"
while [[ $# -gt 0 ]]; do case "$1" in
  --artifacts) ARTIFACTS="$2"; shift 2 ;; --image) TAG="$2"; shift 2 ;;
  --) shift; break ;; *) break ;;
esac; done
mkdir -p "$ARTIFACTS"
EXTRA=()
for v in HF_TOKEN HF_HOME HF_HUB_OFFLINE; do
    [[ -n "${!v:-}" ]] && EXTRA+=( -e "$v=${!v}" )
done
if [[ -n "${HF_HOME:-}" ]]; then mkdir -p "$HF_HOME"; EXTRA+=( -v "$HF_HOME:$HF_HOME" ); fi
exec docker run --gpus all -it --rm --ipc=host --shm-size=64g \
    -v "$REPO:/workspace" -v "$ARTIFACTS:/workspace/artifacts" \
    ${EXTRA[@]+"${EXTRA[@]}"} -w /workspace "$TAG" "$@"
