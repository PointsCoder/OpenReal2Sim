FROM mambaorg/micromamba:jammy-cuda-12.1.0
USER root

# Create conda env from default.yml (setup.md lines 8-9)
COPY environments/default.yml /tmp/default.yml
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV TMPDIR=/root/mamba-tmp
RUN mkdir -p "${TMPDIR}"
RUN micromamba env create -f /tmp/default.yml -y && micromamba clean --all --yes

# System packages needed for editable installs (git, etc.) and open3d (X11 libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libx11-6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Run subsequent steps inside the created env (setup.md line 10: mamba activate sam3d-objects)
SHELL ["micromamba", "run", "-n", "sam3d-objects", "/bin/bash", "-lc"]

# setup.md line 13: for pytorch/cuda dependencies
ENV PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"

# CUDA architecture for RTX 6000 Ada Generation (Ada Lovelace = 8.9)
ENV TORCH_CUDA_ARCH_LIST="8.9"

# Copy repo
WORKDIR /opt/sam3d
COPY . /opt/sam3d/
# Strip nvidia-pyindex (already covered by PIP_EXTRA_INDEX_URL)
RUN sed -i '/^nvidia-pyindex/d' requirements.txt

# setup.md lines 16-17: install sam3d-objects and core dependencies
RUN pip install -e '.[dev]'
RUN pip install -e '.[p3d]'

# setup.md line 20: for inference
ENV PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"

# setup.md line 21: install inference dependencies
RUN pip install -e '.[inference]'

# setup.md line 24: patch hydra
RUN chmod +x ./patching/hydra && ./patching/hydra

# Install nvdiffrast build dependencies
RUN pip install -U wrapt ninja cmake

# Install nvdiffrast from source (symlink conda CUDA to expected path)
RUN ln -s /opt/conda/envs/sam3d-objects /usr/local/cuda-12.1
RUN pip install --no-build-isolation --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git

# Default interactive shell in the env
CMD ["bash"]
