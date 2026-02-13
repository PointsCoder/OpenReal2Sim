FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive


# -------------------------------------------------------------------------
# System deps for building native extensions (xformers) + basics
# -------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip python3-venv python-is-python3 \
    build-essential cmake ninja-build pkg-config libopenblas-dev \
    git curl wget ffmpeg bash libspatialindex-dev \
    docker.io \
    && python3 -m pip install --upgrade --no-cache-dir pip wheel setuptools packaging \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
SHELL ["/bin/bash", "-lc"]

ENV MAX_JOBS=2
ENV MAKEFLAGS="-j2"
ENV CMAKE_BUILD_PARALLEL_LEVEL=2

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
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"

# xFormers
RUN pip install -v --no-build-isolation -U \
    "git+https://github.com/facebookresearch/xformers.git@v0.0.28.post3#egg=xformers"

# Scatter
RUN pip install --no-cache-dir torch-scatter \
    -f https://data.pyg.org/whl/torch-2.5.1+cu118.html

# Rest of the Dependencies
COPY docker/real2sim/requirements.docker.txt         /tmp/requirements.docker.txt
RUN pip install --no-cache-dir -r /tmp/requirements.docker.txt

# gdown for Google Drive downloads
RUN pip install --no-cache-dir gdown


# -------------------------------------------------------------------------
# Third-Party compiled Python deps
# -------------------------------------------------------------------------

# mega-sam
COPY third_party/mega-sam /app/third_party/mega-sam
RUN cd /app/third_party/mega-sam/base && python setup.py install

# Grounded-SAM-2 CUDA extension
COPY third_party/Grounded-SAM-2 /app/third_party/Grounded-SAM-2
RUN cd /app/third_party/Grounded-SAM-2 && python build_cuda.py build_ext --inplace

# Hunyuan3D
COPY third_party/Hunyuan3D-2 /app/third_party/Hunyuan3D-2
RUN cd /app/third_party/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer && python setup.py install
RUN cd /app/third_party/Hunyuan3D-2/hy3dgen/texgen/differentiable_renderer && python setup.py install

# pybind11 (needed for FoundationPose C++ extension)
RUN cd / && git clone https://github.com/pybind/pybind11 &&\
    cd pybind11 && git checkout v2.10.0 &&\
    mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DPYBIND11_INSTALL=ON -DPYBIND11_TEST=OFF &&\
    make -j6 && make install

# Eigen3 (needed for FoundationPose C++ extension)
RUN cd / && wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz &&\
    tar xvzf ./eigen-3.4.0.tar.gz &&\
    cd eigen-3.4.0 &&\
    mkdir build &&\
    cd build &&\
    cmake .. &&\
    make install

# FoundationPose
RUN apt-get update && apt-get install -y libboost-dev libboost-system-dev libboost-program-options-dev
COPY third_party/FoundationPose /app/third_party/FoundationPose
RUN python -m pip install --no-build-isolation --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git@abb07ca0358f3a21c3942b50c54aa1eacd329af9
RUN python -m pip install --no-cache-dir kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu118.html
# RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git"
RUN PIP_NO_BUILD_ISOLATION=1 pip install --no-build-isolation --no-cache-dir \
    "git+https://github.com/facebookresearch/pytorch3d.git"

# FoundationPose C++ extension
RUN cd /app/third_party/FoundationPose/mycpp && \
    rm -rf build && mkdir build && cd build && \
    cmake .. -DPYTHON_EXECUTABLE=$(which python) && \
    make -j11

# FoundationPose bundlesdf CUDA extension
RUN cd /app/third_party/FoundationPose/bundlesdf/mycuda && \
    rm -rf build *.egg-info && \
    pip install --no-build-isolation .

COPY third_party/GraspGen /app/third_party/GraspGen
RUN cd /app/third_party/GraspGen && pip install -e .
RUN cd /app/third_party/GraspGen/pointnet2_ops && pip install --no-build-isolation .

RUN pip install --no-build-isolation --no-cache-dir "git+https://github.com/mattloper/chumpy"

EXPOSE 5000
# -------------------------------------------------------------------------
# Runtime env
# -------------------------------------------------------------------------
ENV PYTHONPATH=/app

# -------------------------------------------------------------------------
# Create cache directories with proper permissions for torch extensions
# -------------------------------------------------------------------------
RUN mkdir -p /tmp/.cache/torch_extensions && \
    chmod -R 777 /tmp/.cache

# Set cache environment variables to use /tmp paths
ENV TORCH_EXTENSIONS_DIR=/tmp/.cache/torch_extensions
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
ENV HF_HOME=/app/.cache/huggingface

