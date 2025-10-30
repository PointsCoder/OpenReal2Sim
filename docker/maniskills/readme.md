# Docker

Docker provides a convenient way to package software into standardized units for development, shipment and deployment. See the [official website](https://www.docker.com/resources/what-container/) for more details about Docker. [NVIDIA Container Tookit](https://github.com/NVIDIA/nvidia-docker) enables users to build and run GPU accelerated Docker containers.

First, install [nvidia-docker v2](https://github.com/NVIDIA/nvidia-docker) following the [official instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker). It is recommended to complete post-install steps for Linux.

To verify the installation:

```bash
# You should be able to run this without sudo.
docker run hello-world
```

## Run ManiSkill in Docker

We provide a docker image (`maniskill/base`) and its corresponding [Dockerfile](https://github.com/haosulab/ManiSkill/blob/main/docker/Dockerfile).

You should be able to run both CPU and GPU simulation, which you can test below

```bash
docker pull maniskill/base
docker run --rm -it --gpus all --pid host maniskill/base python -m mani_skill.examples.demo_random_action
docker run --rm -it --gpus all --pid host maniskill/base python -m mani_skill.examples.benchmarking.gpu_sim
```

Note that inside a docker image you generally cannot render a GUI to see the results. You can still record videos and the demo scripts have options to record videos instead of rendering a GUI.
