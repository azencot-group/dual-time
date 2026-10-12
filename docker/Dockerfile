# Self-contained image for the dual-tower Qwen3-VL-8B pipeline.
#
# CUDA 12.8 devel + Python 3.11 + the full environment in /opt/venv (the same
# install as the README's Environment section: torch 2.7.0+cu128, the pinned
# requirements.txt, flash-attn 2.8.3). The repo and the artifacts are
# bind-mounted at runtime (docker/run.sh), so code or data changes never need
# an image rebuild.
#
#   bash docker/build.sh        # from the repo root
#   bash docker/run.sh          # interactive shell with GPUs + mounts
FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0" \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# Python 3.11 (deadsnakes; Ubuntu 24.04 ships 3.12) + build tools + uv.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl git build-essential \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh \
    && uv --version

# Environment: only requirements.txt is copied, so the layer is cached until
# the dependencies change.
#   1. torch 2.7.0 + torchvision 0.22.0 from the CUDA 12.8 wheel index
#      (uv checks --extra-index-url BEFORE PyPI, so the +cu128 build wins);
#   2. build helpers, then requirements.txt;
#   3. flash-attn 2.8.3: the prebuilt wheel matching torch 2.7 / the C++11 ABI
#      of this torch build, with a source build as the fallback.
COPY requirements.txt /tmp/build/requirements.txt
RUN uv venv --python /usr/bin/python3.11 /opt/venv \
    && . /opt/venv/bin/activate \
    && uv pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.7.0 torchvision==0.22.0 \
    && uv pip install setuptools wheel ninja packaging \
    && uv pip install -r /tmp/build/requirements.txt \
    && TV="$(python -c 'import torch;print(".".join(torch.__version__.split("+")[0].split(".")[:2]))')" \
    && ABI="$(python -c 'import torch;print("TRUE" if torch.compiled_with_cxx11_abi() else "FALSE")')" \
    && ( uv pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch${TV}cxx11abi${ABI}-cp311-cp311-linux_x86_64.whl" \
         || MAX_JOBS=8 uv pip install "flash_attn==2.8.3" --no-build-isolation ) \
    && python -c "import torch, flash_attn; assert '+cu128' in torch.__version__, torch.__version__; print('torch', torch.__version__, 'flash_attn', flash_attn.__version__)" \
    && rm -rf /tmp/build /root/.cache

ENV PATH="/opt/venv/bin:${PATH}"
WORKDIR /workspace
CMD ["/bin/bash"]
