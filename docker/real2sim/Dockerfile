# nvidia/cuda:11.8.0-devel-ubuntu22.04
FROM nvidia/cuda@sha256:94fd755736cb58979173d491504f0b573247b1745250249415b07fefc738e41f

ENV DEBIAN_FRONTEND=noninteractive


# -------------------------------------------------------------------------
# System deps for building native extensions (xformers) + basics
# -------------------------------------------------------------------------
# pip stays only to bootstrap setuptools/wheel: uv is not in the image yet, and
# the `setup.py install` and `--no-build-isolation` layers below need them.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip python3-venv python-is-python3 \
    build-essential cmake ninja-build pkg-config libopenblas-dev \
    git curl wget ffmpeg bash libspatialindex-dev \
    libegl1 libglu1-mesa \
    docker.io \
    && python3 -m pip install --upgrade --no-cache-dir pip wheel setuptools packaging \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
SHELL ["/bin/bash", "-lc"]

ENV MAX_JOBS=2
ENV MAKEFLAGS="-j2"
ENV CMAKE_BUILD_PARALLEL_LEVEL=2

# -------------------------------------------------------------------------
# uv: every Python install below runs through `uv pip install --system`
# -------------------------------------------------------------------------
# ghcr.io/astral-sh/uv:0.13.8
COPY --from=ghcr.io/astral-sh/uv@sha256:5cc1805e2426484abca4ad46ea783bab281fdb225fd82d8ce6d66a81254bac85 /uv /usr/local/bin/uv

# The CUDA 11.8 torch pins apply to every resolution, not just the big
# requirements one: any later layer that drags torch back into the resolver
# (xformers -U, kaolin, GraspGen) would otherwise be free to replace the
# cu118 wheels and break the CUDA extensions, which is what OPE-16 hit.
COPY docker/real2sim/constraints.txt /tmp/constraints.txt
ENV UV_CONSTRAINT=/tmp/constraints.txt
ENV UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu118
# pip's own behaviour: consider every index and take the best version, rather
# than stopping at the first index that happens to know the package name.
ENV UV_INDEX_STRATEGY=unsafe-best-match

# -------------------------------------------------------------------------
# Install Python Dependencies
# -------------------------------------------------------------------------

# PyTorch
RUN uv pip install --system --no-cache --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.5.1+cu118 torchvision==0.20.1+cu118 torchaudio==2.5.1+cu118

# Exposing CUDA to build scripts
ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA=1
# T4=7.5, A100=8.0, RTX30=8.6, RTX4090/6000Ada=8.9, H100=9.0
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"

# xFormers
RUN uv pip install --system --no-cache -v --no-build-isolation -U \
    "xformers @ git+https://github.com/facebookresearch/xformers.git@f3bc7a78e687f7987a920e7bad0dca61c927c5b4"

# Scatter
RUN uv pip install --system --no-cache torch-scatter \
    -f https://data.pyg.org/whl/torch-2.5.1+cu118.html

# Rest of the Dependencies, fully pinned: requirements.docker.in is the list we
# maintain, requirements.docker.lock is what `uv pip compile` resolved from it.
# Regenerate with the command recorded in the lock's header.
COPY docker/real2sim/requirements.docker.lock        /tmp/requirements.docker.lock
RUN uv pip install --system --no-cache -r /tmp/requirements.docker.lock

# gdown for Google Drive downloads
RUN uv pip install --system --no-cache gdown


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
    cd pybind11 && git checkout aa304c9c7d725ffb9d10af08a3b34cb372307020 &&\
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
RUN uv pip install --system --no-cache --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git@abb07ca0358f3a21c3942b50c54aa1eacd329af9
RUN uv pip install --system --no-cache kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu118.html
# RUN uv pip install --system "git+https://github.com/facebookresearch/pytorch3d.git"
RUN uv pip install --system --no-cache --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git@e4ad6e5687a360ccdc6ebd91348e8dbbac97bf31"

# FoundationPose C++ extension
RUN cd /app/third_party/FoundationPose/mycpp && \
    rm -rf build && mkdir build && cd build && \
    cmake .. -DPYTHON_EXECUTABLE=$(which python) && \
    make -j11

# FoundationPose bundlesdf CUDA extension
RUN cd /app/third_party/FoundationPose/bundlesdf/mycuda && \
    rm -rf build *.egg-info && \
    uv pip install --system --no-cache --no-build-isolation .

COPY third_party/GraspGen /app/third_party/GraspGen
RUN cd /app/third_party/GraspGen && uv pip install --system --no-cache -e .
RUN cd /app/third_party/GraspGen/pointnet2_ops && uv pip install --system --no-cache --no-build-isolation .

RUN uv pip install --system --no-cache --no-build-isolation "git+https://github.com/mattloper/chumpy@580566eafc9ac68b2614b64d6f7aaa84eebb70da"

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

