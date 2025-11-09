# Docker Instructions

Docker provides a sharable environment to run the code in this repository.

We provide separate Dockerfiles for real2sim reconstruction and different robot simulation environments:
- `docker/real2sim/Dockerfile` — for Real2Sim preprocessing and reconstruction.
- `docker/isaaclab/Dockerfile` — for IsaacLab simulation and policy training.

Each can be built and run independently depending on your use case.

## How to Run the Docker Container

Here are the steps to build and run the Docker container for this repository.

### Access to Docker Commands

Make sure you have Docker installed and running on your machine. 

You can check this by running:
   ```bash
   docker --version
   ```

Make sure you have been added to the `docker` group to run docker commands without `sudo`.

if you encountered permission issues when running `docker_build` or `docker_run`, you can ask your administrator to try:
   ```bash
   sudo usermod -aG docker $USER
   ```
to add you to the `docker` group. You may need to log out and log back in for this to take effect.

if sometimes the group change does not take effect, you can also try:
   ```bash
   newgrp docker
   ```
to switch to the `docker` group in the current terminal session.

### Use Pre-Built Docker Image

If you want to use a pre-built image from Docker Hub, you can pull it directly:
   ```bash
   docker pull ghcr.io/pointscoder/openreal2sim:dev
   docker tag ghcr.io/pointscoder/openreal2sim:dev openreal2sim:dev
   ```

### (Optional) Build the Docker Image

You can also build the docker image yourself. Once the container is running, you can execute any of the scripts above inside it.

**Build the container image**
   From the repository root:
   ```bash
   docker compose -f docker/compose.yml build openreal2sim # or isaaclab
   ```

Optionally, you may want to push the image to a remote registry (e.g., GitHub Container Registry) for easier sharing.

If so, you can tag the image and push it:
   ```bash
   docker tag openreal2sim:dev ghcr.io/<username>/openreal2sim:dev
   docker push ghcr.io/<username>/openreal2sim:dev
   ```
Don't forget to change the visibility of the pushed image to public if you want others to access it.

### After Getting the Docker Image

Every time you want to run a script inside the repo, follow these steps:

#### Run the real2sim container
   ```bash
   HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose -p "$USER" -f docker/compose.yml run openreal2sim
   ```

**Inside the docker container**, run the following script to download pretrained checkpoints and compile c++/cuda extensions:
```
python scripts/installation/install.py
```

**Execute a script inside the container**
   
   Inside the container’s terminal and from the repository root:
   ```bash
   python <path_to_script>.py
   ```

#### Run the isaaclab container
   ```bash
   HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose -p "$USER" -f docker/compose.yml run isaaclab
   ```

#### Tips if you are using VSCode

1. Install the [Container Tools](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-containers) extension for setting up the docker container as a remote development environment.

2. Install the [vscode-3d-preview](https://marketplace.visualstudio.com/items?itemName=tatsy.vscode-3d-preview) extension for visualizing 3D point clouds directly in VSCode.

3. Install the [glTF Model Viewer](https://marketplace.visualstudio.com/items?itemName=cloudedcat.vscode-model-viewer) extension for visualizing textured meshes (.glb files) directly in VSCode.