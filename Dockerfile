FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive


# -------------------------------------------------------------------------
# System deps for building native extensions (xformers) + basics
# -------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip python3-venv python-is-python3 \
    build-essential cmake ninja-build pkg-config \
    git curl wget ffmpeg bash libspatialindex-dev \
    && python3 -m pip install --upgrade --no-cache-dir pip wheel setuptools packaging \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
SHELL ["/bin/bash", "-lc"]

# -------------------------------------------------------------------------
# Install Python Dependencies
# -------------------------------------------------------------------------

# PyTorch
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.5.1+cu118 torchvision==0.20.1+cu118 torchaudio==2.5.1+cu118

# Exposing CUDA to build scripts
ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA=1
# T4=7.5, A100=8.0, RTX30=8.6, RTX4090/6000Ada=8.9, H100=9.0
ENV TORCH_CUDA_ARCH_LIST="7.5"

# xFormers
RUN pip install -v --no-build-isolation -U \
    "git+https://github.com/facebookresearch/xformers.git@v0.0.28.post3#egg=xformers"

# Scatter
RUN pip install --no-cache-dir torch-scatter \
    -f https://data.pyg.org/whl/torch-2.5.1+cu118.html

# Rest of the Dependencies
COPY docker/requirements.docker.txt         /tmp/requirements.docker.txt
RUN pip install --no-cache-dir -r /tmp/requirements.docker.txt


# -------------------------------------------------------------------------
# Third-Party compiled Python deps (mega-sam base)
# -------------------------------------------------------------------------

COPY third_party/mega-sam /app/third_party/mega-sam
RUN cd /app/third_party/mega-sam/base && python setup.py install


COPY third_party/Hunyuan3D-2 /app/third_party/Hunyuan3D-2
RUN cd /app/third_party/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer && python setup.py install
RUN cd /app/third_party/Hunyuan3D-2/hy3dgen/texgen/differentiable_renderer && python setup.py install
# -------------------------------------------------------------------------
# Runtime env
# -------------------------------------------------------------------------
ENV PYTHONPATH=/app
