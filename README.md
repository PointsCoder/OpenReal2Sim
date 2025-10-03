# OpenReal2Sim
A toolbox for real-to-sim reconstruction and robotic simulation

## Installation

Clone this repository recursively:
```
git clone git@github.com:PointsCoder/OpenReal2Sim.git --recurse-submodules
```

Next, we need to set up the Python environment. We recommend using `docker` for managing dependencies.

Please refer to [docker installation](docker/README.md) for launching the docker environment.

Now only `openreal2sim` is required, which contains all dependencies.

Launching the docker container:
```
docker compose -p "$USER" -f docker/compose.yml run --rm openreal2sim
```

**Inside the docker container**, run the following script to download pretrained checkpoints:
```
bash scripts/installation/ckpt_download.sh
```

**Tips if you are using VSCode**:

1. Install the [Container Tools](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-containers) extension for setting up the docker container as a remote development environment.

2. Install the [vscode-3d-preview](https://marketplace.visualstudio.com/items?itemName=tatsy.vscode-3d-preview) extension for visualizing 3D point clouds directly in VSCode.

3. Install the [glTF Model Viewer](https://marketplace.visualstudio.com/items?itemName=cloudedcat.vscode-model-viewer) extension for visualizing textured meshes (.glb files) directly in VSCode.

## How to Use

Our framework contains several stages:
- [Preprocess](openreal2sim/preprocess/README.md): collect or estimate depths and camera information from images and videos
- [Reconstruction](openreal2sim/reconstruction/README.md): build physically interactable scenes from images and videos
- [Simulation](openreal2sim/simulation/README.md): import physical scenes into the simulator, collect robotic demonstrations

### Preprocess

Running this scripts for all preprocessing steps:
```
bash scripts/running/preprocess.sh
```
