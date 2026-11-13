# Docker Instructions

Docker provides a sharable environment to run the code in this repository.

We provide separate Dockerfiles for real2sim reconstruction and different robot simulation environments:
- `docker/real2sim/Dockerfile` — for Real2Sim preprocessing and reconstruction.
- `docker/isaaclab/Dockerfile` — for IsaacLab simulation and policy training.
- `docker/mujoco/Dockerfile` — for Mujoco simulation.

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

You also need to have NVIDIA CUDA Container Toolkit installed. See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

### Set Up with init.py

`init.py` at the repository root does the whole first-time setup: it checks docker
and the nvidia runtime, creates the host directories the containers write to,
pulls the images, and downloads the pretrained weights. From the repository root:
   ```bash
   python3 init.py setup
   ```

Pulling the pre-built images is the normal way to get started; building them from
the Dockerfiles takes hours. The packages are private until Esteban makes them
public, so until then you need to authenticate first with a GitHub token that has
the `read:packages` scope:
   ```bash
   docker login ghcr.io
   ```

The weights are about 22 GB and take around 20 minutes. `setup` skips them when
`outputs/.weights_installed` exists; `--force-weights` downloads them again and
`--skip-weights` leaves them out.

### (Optional) Build the Docker Image

Build the images yourself only when you change a Dockerfile. Once the container is running, you can execute any of the scripts above inside it.

**Build the container image**
   From the repository root:
   ```bash
   python3 init.py setup --build
   ```

Optionally, you may want to push the image to a remote registry (e.g., GitHub Container Registry) for easier sharing.

If so, you can tag the image and push it:
   ```bash
   docker tag ghcr.io/licesma/openreal2sim/reconstruction:dev ghcr.io/<username>/openreal2sim/reconstruction:dev
   docker push ghcr.io/<username>/openreal2sim/reconstruction:dev
   ```
Don't forget to change the visibility of the pushed image to public if you want others to access it.

### After Getting the Docker Image

**Run one script in a container**
   ```bash
   python3 init.py run reconstruction -- python openreal2sim/preprocess/preprocess_manager.py
   python3 init.py run simulation -- python <path_to_script>.py
   ```
   Everything after `--` is the command line executed inside the container, from
   the repository root. The container is removed when the command ends.

**Pick the GPUs**
   ```bash
   python3 init.py run reconstruction --gpus 0,1 -- python <path_to_script>.py
   ```
   Without `--gpus` the container sees every GPU on the host.

**Publish a port for the GUIs**
   ```bash
   python3 init.py run reconstruction --port 7860:7860 -- python openreal2sim/reconstruction/tools/segmentation_annotator.py
   python3 init.py run reconstruction --port 8000:5000 -- python openreal2sim/motion/tool/gripper_pose_editor.py
   ```

**Open an interactive shell**
   ```bash
   python3 init.py shell reconstruction   # or simulation
   ```
   `shell` takes the same `--gpus` and `--port` flags.

#### Tips if you are using VSCode

1. Install the [Container Tools](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-containers) extension for setting up the docker container as a remote development environment.

2. Install the [vscode-3d-preview](https://marketplace.visualstudio.com/items?itemName=tatsy.vscode-3d-preview) extension for visualizing 3D point clouds directly in VSCode.

3. Install the [glTF Model Viewer](https://marketplace.visualstudio.com/items?itemName=cloudedcat.vscode-model-viewer) extension for visualizing textured meshes (.glb files) directly in VSCode.

### Without init.py

`init.py` only assembles `docker compose` calls, so the raw commands still work:
   ```bash
   docker compose -f docker/compose.yml pull reconstruction simulation
   HOST_UID=$(id -u) HOST_GID=$(id -g) DOCKER_GID=$(getent group docker | cut -d: -f3) \
     docker compose -p "$USER" -f docker/compose.yml run --rm --publish 7860:7860 reconstruction bash
   ```
`HOST_UID`/`HOST_GID` keep the files the container writes owned by you, and
`DOCKER_GID` lets the reconstruction container reach the docker socket. Selecting
a subset of GPUs needs a second `-f` file overriding
`deploy.resources.reservations.devices`, which is what `--gpus` writes.
