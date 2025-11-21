FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive


# -------------------------------------------------------------------------
# System deps for building native extensions (xformers) + basics
# -------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip python3-venv python-is-python3 \
    build-essential cmake ninja-build pkg-config libopenblas-dev \
    git curl wget ffmpeg bash libspatialindex-dev \
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

# Install chumpy separately (setup.py needs pip module, ensure build deps are available)
RUN pip install --no-build-isolation --no-cache-dir "git+https://github.com/mattloper/chumpy"


# -------------------------------------------------------------------------
# Third-Party compiled Python deps
# -------------------------------------------------------------------------

# mega-sam
COPY third_party/mega-sam /app/third_party/mega-sam
RUN cd /app/third_party/mega-sam/base && python setup.py install

# Hunyuan3D
COPY third_party/Hunyuan3D-2 /app/third_party/Hunyuan3D-2
RUN cd /app/third_party/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer && python setup.py install
RUN cd /app/third_party/Hunyuan3D-2/hy3dgen/texgen/differentiable_renderer && python setup.py install

# FoundationPose
RUN apt-get update && apt-get install -y libboost-dev libboost-system-dev libboost-program-options-dev
COPY third_party/FoundationPose /app/third_party/FoundationPose
RUN python -m pip install --no-build-isolation --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git
RUN python -m pip install --no-cache-dir kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu118.html
# RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git"
RUN PIP_NO_BUILD_ISOLATION=1 pip install --no-build-isolation --no-cache-dir \
    "git+https://github.com/facebookresearch/pytorch3d.git"

RUN cd / && git clone https://github.com/pybind/pybind11 &&\
    cd pybind11 && git checkout v2.10.0 &&\
    mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DPYBIND11_INSTALL=ON -DPYBIND11_TEST=OFF &&\
    make -j6 && make install

RUN cd / && wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz &&\
    tar xvzf ./eigen-3.4.0.tar.gz &&\
    cd eigen-3.4.0 &&\
    mkdir build &&\
    cd build &&\
    cmake .. &&\
    make install

RUN cd /
COPY third_party/graspnetAPI /app/third_party/graspnetAPI
RUN cd /app/third_party/graspnetAPI && pip install -e .

COPY third_party/graspness_unofficial /app/third_party/graspness_unofficial
RUN cd /app/third_party/graspness_unofficial/pointnet2 && python -m pip install --no-cache-dir -v --no-build-isolation .
RUN cd /app/third_party/graspness_unofficial/knn && python -m pip install --no-cache-dir -v --no-build-isolation .

COPY third_party/MinkowskiEngine /app/third_party/MinkowskiEngine
RUN cd /app/third_party/MinkowskiEngine && python setup.py install --blas=openblas --force_cuda

# -------------------------------------------------------------------------
# Runtime env
# -------------------------------------------------------------------------
ENV PYTHONPATH=/app:/app/third_party:/app/third_party/graspness_unofficial/pointnet2:
