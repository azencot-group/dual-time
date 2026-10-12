#!/usr/bin/env bash
# Build the image from the repo root:  bash docker/build.sh [tag]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE/.."
TAG="${1:-dual-time:latest}"
docker build -f docker/Dockerfile -t "$TAG" .
echo "built $TAG"
