FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip python3-venv python-is-python3 \
    python3-distutils \
    libopenblas-dev \
    build-essential cmake ninja-build pkg-config \
    git curl wget ffmpeg bash libspatialindex-dev \
    ncurses-bin ncurses-term \
    libglu1-mesa libxkbcommon-x11-0 libegl1 libgl1 libgles2 \
    libxinerama1 libxcursor1 libsm6 libice6 libxrender1 libxrandr2 \
    libxt6 libxmu6 \
    && python3 -m pip install --upgrade --no-cache-dir pip wheel setuptools packaging \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
SHELL ["/bin/bash", "-lc"]

ENV MAX_JOBS=2
ENV MAKEFLAGS="-j2"
ENV CMAKE_BUILD_PARALLEL_LEVEL=2

ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA=1

# RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu118 \
#     torch==2.5.1+cu118 torchvision==0.20.1+cu118 torchaudio==2.5.1+cu118

RUN pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128 
RUN pip install transforms3d


USER root
# Patch shared_ptr_base.h
RUN sed -i 's|auto __raw = __to_address(__r.get())|auto __raw = std::__to_address(__r.get())|' \
    /usr/include/c++/11/bits/shared_ptr_base.h

ENV TORCH_CUDA_ARCH_LIST="8.9+PTX"
# curobo - install to /opt to avoid being overwritten by volume mount
COPY third_party/curobo /app/third_party/curobo
RUN cd /app/third_party/curobo && \
     pip install -e . --no-build-isolation

# Isaac Sim 4.5.0
RUN pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com

# Revert the fixed pytorch version
# Install IsaacLab to /opt to avoid being overwritten by volume mount
ENV TERM=xterm-256color
COPY third_party/IsaacLab /app/third_party/IsaacLab
RUN cd /app/third_party/IsaacLab && \
    sed -i 's|torch==2.5.1|torch==2.7.0|' source/isaaclab/setup.py && \
    sed -i 's|torch==2.5.1|torch==2.7.0|' source/isaaclab_tasks/setup.py && \
    sed -i 's|torch==2.5.1|torch==2.7.0|' source/isaaclab_rl/setup.py && \
    ./isaaclab.sh --install

# usd conversion 
# curobo changes the numpy version from 1.23.4 to 2.x, need to get it back
RUN pip install loguru rich numpy==1.23.4 transforms3d

# Ensure isaaclab can be imported
# IsaacLab is installed to /opt/IsaacLab to avoid being overwritten by volume mount
ENV PYTHONPATH=/app:/app/third_party:/app/third_party/IsaacLab/source
