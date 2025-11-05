# User Guide

Our framework contains several stages:
- [Preprocess](openreal2sim/preprocess/): collect or estimate depths and camera information from images and videos
- [Reconstruction](openreal2sim/reconstruction/): build physically interactable scenes from images and videos
- [Simulation](openreal2sim/simulation/): import physical scenes into the simulator, collect robotic demonstrations

### Before We Start

1. We will start with the provided examples in the `data/` folder. 

You can also prepare your own data following the same structure:
```
data/
    key_name.jpg or .png  # for single image input
    key_name.mp4          # for video input
    key_name.mp4 + key_name_depth.png # for video input with GT depths
```
and make sure the `key_name` is also in `config/config.yaml`.

2. Make sure we have pulled or built the docker images following the [docker setup](../docker/README.md).

We will use `openreal2sim:dev` image for the reconstruction and `isaaclab:dev` image for the IsaacLab simulation.

### Preprocess

Launch the docker container with mounted repository and data folder:
```
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose -p "$USER" -f docker/compose.yml run openreal2sim
```

The following steps are all performed inside the docker container.

For the first-time use, **inside the docker container**, run the following script to download pretrained checkpoints and compile c++/cuda extensions:

```
python scripts/installation/install.py
```



Running this scripts for all preprocessing steps:
```
python openreal2sim/preprocess/preprocess_manager.py
```

This will provide estimated depths and camera intrinsics/extrinsics for the reconstruction stage. 

We can check the point cloud reconstruction quality at `outputs/{key_name}/geometry/dynamic_pcd.ply`:

<div style="text-align:center;">
  <video
    src="../data/demo_genvideo.mp4"
    controls
    muted
    playsinline
    loop
    style="width:45%; display:inline-block; vertical-align:top; margin-right:8px;"
  >
    Your browser does not support the video tag.
  </video>

  <img
    src="../assets/pcd_viz.jpg"
    alt="Point cloud visualization"
    style="width:45%; display:inline-block; vertical-align:top;"
  />
</div>




### Reconstruction

We first need to segment objects that needs to be reconstructed. We provide a GUI for this purpose:
```
python openreal2sim/reconstruction/tools/segmentation_annotator.py
```

![Segmentation UI](../assets/UI.jpg)



**How to use the GUI annotator:**

1. Input `key_name` (e.g. `demo_image`) in the `Output-key` textbox and press `load` to load image frames

2. Select the objects you want to segment by simply clicking on the image

3. Modify the object name from the default `pc_obj` to the class name in the `Point-click name` and press `Confirm mask` and `Save mask_dict`.

4. If you are processing a video, press the `PROPAGATE & SAVE` button to propagate the segmentation masks across frames.

**Please note that we must have a `ground` mask annotated, since we need this to find the ground plane for reconstruction.**

Example annotated masks are in `outputs/{key_name}/annotated_images`:
<p align="center">
  <img src="../assets/seg_image.jpg" width="30%">
  <img src="../assets/seg_video.jpg" width="30%">
  <img src="../assets/seg_genvideo.jpg" width="30%">
</p>

Then, run the whole physical scene reconstruction pipeline:
```
python openreal2sim/reconstruction/recon_agent.py
```

We will get a portable scene assets folder at `outputs/{key_name}/simulation` and camera & scene information at `outputs/{key_name}/simulation/scene.json`.

We can check the mesh reconstruction quality at `outputs/{key_name}/simulation/scene_optimized.glb`:
<p align="center">
  <img src="../assets/obj1.jpg" width="30%">
  <img src="../assets/obj2.jpg" width="30%">
  <img src="../assets/obj3.jpg" width="30%">
</p>



### Robotic Simulation

We support importing the reconstructed scenes into different physics simulators and collecting robotic trajectories by cross-embodiment transfer from videos. Please refer to the [simulation](../openreal2sim/simulation/) part for more details.

Current supported physics simulators:

- [x] IssacLab ([IssacLab v2.0.2](https://isaac-sim.github.io/IsaacLab/v2.0.2/source/setup/ecosystem.html) & [IssacSim v4.5.0](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/index.html))
- [x] Maniskills
- [ ] Mujoco [WIP]

#### IsaacLab Simulation:

**On the host machine and before entering the container**, run
```
xhost +local:
```
This is to allow the container to access the host's X server for IsaacSim GUI.

Then, launch the container:
```
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose -p "$USER" -f docker/compose.yml up -d isaaclab 
```
and enter it:
```
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose -p "$USER" -f docker/compose.yml exec isaaclab bash
```

In the container, we need to convert the `glb` meshes to `usd` format. This is done by running the following script:
```
python openreal2sim/simulation/isaaclab/sim_preprocess/usd_conversion.py
```

**Note**: when running the above script for the first time, it may take quite a while to load IsaacSim. Just be patient.


Then, we generate grasp proposals for each object in the scene:
```
python openreal2sim/simulation/isaaclab/sim_preprocess/grasp_generation.py
```

We can check the grasp proposals at `outputs/{key_name}/grasps/`.



Next, we can import scenes in IsaacSim. 
We provide a heuristic policy using grasping and motion planning:
```
python openreal2sim/simulation/isaaclab/sim_heuristic_manip.py
```

and you may observe something like this in the IsaacSim GUI:

<div style="text-align:center;">
  <video
    src="../assets/isaaclab.mp4"
    controls
    muted
    playsinline
    loop
    style="width:75%; display:inline-block; vertical-align:top; margin-right:8px;"
  >
    Your browser does not support the video tag.
  </video>
</div>

<br><br>

and you can find the visuo-motor trajectories at `outputs/{key_name}/demos/`.


**Note**: The heuristic policy can be very un-stable, largely depending on the video object pose estimation quality. Please check the object pose estimation results at `outputs/{key_name}/reconstruction/objects/*.mp4`.

If the results are not good, a solution might be re-running the reconstruction stage from object pose estimation.
```
python openreal2sim/reconstruction/recon_agent.py --stage "scenario_fdpose_optimization"
```
by tuning the `fdpose_est_refine_iter` and `fdpose_track_refine_iter` in `config/config.yaml`.