FROM python:3.10

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    ca-certificates wget git build-essential cmake swig \
    libgl1 libglx-mesa0 libosmesa6 libglu1-mesa \
    libx11-6 libxrandr2 libxinerama1 libxcursor1 libxi6 libxxf86vm1 \
    mesa-utils libglfw3 libglew2.2 \
    mesa-vulkan-drivers libgl1-mesa-dri \
    vim tree tmux zsh \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY docker/mujoco/requirements.docker.txt /tmp/requirements.docker.txt
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install -r /tmp/requirements.docker.txt \
 && rm /tmp/requirements.docker.txt

ENV LANG=C.UTF-8

CMD ["/bin/bash"]
