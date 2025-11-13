# OpenReal2Sim
A toolbox for real-to-sim reconstruction and robotic simulation

## Installation

Clone this repository recursively:
```
git clone git@github.com:PointsCoder/OpenReal2Sim.git --recurse-submodules
```

Next, we need to set up the Python environment. We recommend using `docker` for managing dependencies.

Please refer to [docker installation](docker/README.md) for launching the docker environment.

## How to Use

Our framework contains several stages:
- [Preprocess](openreal2sim/preprocess/README.md): collect or estimate depths and camera information from images and videos
- [Reconstruction](openreal2sim/reconstruction/README.md): build physically interactable scenes from images and videos
- [Simulation](openreal2sim/simulation/README.md): import physical scenes into the simulator, collect robotic demonstrations

### Preprocess

Running this scripts for all preprocessing steps:
```
python openreal2sim/preprocess/preprocess_manager.py
```

### Reconstruction

We first need to segment objects that needs to be reconstructed. We provide a GUI for this purpose:
```
python openreal2sim/reconstruction/tools/segmentation_annotator.py
```

**How to use the GUI annotator:**

1. Input `key_name` (e.g. `demo_image`) in the `Output-key` textbox and press `load` to load image frames

2. Select the objects you want to segment by simply clicking on the image

3. Modify the object name from the default `pc_obj` to the class name in the `Point-click name` and press `Confirm mask` and `Save mask_dict`.

4. If you are processing a video, press the `PROPAGATE & SAVE` button to propagate the segmentation masks across frames.

**Please note that we must have a `ground` mask annotated, since we need this to find the ground plane for reconstruction.**

Then, run the whole physical scene reconstruction pipeline:
```
python openreal2sim/reconstruction/recon_agent.py
```