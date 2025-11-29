## Motion & Grasp Preparation

This directory processes human demonstrations into robot-friendly motion goals and grasp priors.  
It consumes `outputs/<key>/scene/scene.pkl` and `simulation/scene.json` from the reconstruction pipeline and augments them with:

- frame-aligned hand/keypoint observations
- cleaned trajectories and task metadata
- contact-based grasp points & approach directions
- dense grasp proposals for every object

This folder operates on `motion` in `scene.pkl`, and `scene.json`.

The resulting assets are consumed by the simulation agents (`openreal2sim/simulation/*`) for replay, teleoperation, or policy training.

---

## Code Structure

```
openreal2sim/motion
├── motion_manager.py 
├── modules
│   ├── hand_extraction.py
│   ├── demo_motion_process.py 
│   ├── grasp_point_extraction.py
│   └── grasp_generation.py
├── utils
│   ├── grasp_utils.py
│   └── visualize_scene.py
└── README.md
```

Each module can be run independently, but `motion_manager.py` provides a stage-aware wrapper similar to `recon_agent.py`.

---

## Workflow

### 1. Hand Extraction
`modules/hand_extraction.py`

- Detects hands with YOLO + SAM2, then lifts 3D hand pose using **WiLoR**.
- Records per-frame keypoints, MANO global orientation, and binary masks under `scene_dict["motion"]`.
- These signals define contact windows and guide later grasp localization.

### 2. Demonstration Motion Processing
`modules/demo_motion_process.py`

- Reprojects FD-POSE / simple / hybrid trajectories into the world frame.
- Picks the manipulated object by total displacement, classifies remaining objects as static, and decides which trajectory family (`traj_key`) best matches the demonstration.
- Downsample the trajectory, exports a final GLB at the end pose, and writes helper metadata (`manipulated_oid`, `start_frame_idx`, `task_type`, `start/end_related`, `gripper_closed`, etc.) back into `simulation/scene.json`.
- Whether a static object is start-related or end-related is determined by whether it is interleaved with the manipulated object in the x-y bounding box at corresponding timestep. We consider the start/end-related object as semantcially related with the manipulated object concerning the task.
- Currently we support three type of tasks: a task is `simple_pick` if the gripper is finally closed, and there is no end-related object; a task is `simple_pick_place` if the manipulated object is final put onto the ground and gripper is opened, and if there is no end-related object; a task is `targetted_pick_place` if there exists end-related objects.


### 3. Grasp Point & Direction Extraction
`modules/grasp_point_extraction.py`

- Aligns the manipulated mesh to the selected keyframe, renders it with PyTorch3D, and fuses depth/hand keypoints to locate the contact patch.
- Projects the contact into object coordinates to obtain `grasp_point` and estimates a world-frame approach direction from MANO orientation.
- Stores diagnostics (render overlays) under `outputs/<key>/simulation/debug/`.

### 4. Grasp Proposal Generation
`modules/grasp_generation.py`

- Samples dense surface points per object, runs **GraspNet** to produce grasp candidates, and optionally applies NMS + score truncation.
- Saves raw `.npy` proposals & `.ply` visualizations under `outputs/<key>/grasps/`.
- For the manipulated object, rescores grasps using the contact point/direction hints and writes both `grasps` and `rescored_grasps` paths into `scene.json`.
- There are three strategies to rescore grasps:
    - `default`: use the z-align strategy to rescore grasps.
    - `point_only`: only use the contact point to rescore grasps.
    - `point_and_direction`: use both the contact point and the approach direction to rescore grasps.

---

## How to Use

1. **Prerequisites**
   - Run preprocess + reconstruction so that `outputs/<key>/scene/scene.pkl` and `simulation/scene.json` exist.
   - Ensure third-party weights are in place:
     - `third_party/WiLoR/pretrained_models` and `third_party/WiLoR/mano_data/MANO_RIGHT.pkl` 
     - `third_party/Grounded-SAM-2/checkpoints`
     - GraspNet checkpoints 

2. **All-in-one pipeline**

```bash
python openreal2sim/motion/motion_manager.py
```


```bash
python openreal2sim/motion/motion_manager.py \
    --stage grasp_generation \
    --key demo_video
```

3. **Stage-specific scripts**

```bash
python openreal2sim/motion/modules/hand_extraction.py

python openreal2sim/motion/modules/demo_motion_process.py

python openreal2sim/motion/modules/grasp_point_extraction.py


python openreal2sim/motion/modules/grasp_generation.py \
    --n_points 120000 --keep 200 --nms True
```

All scripts read keys from `config/config.yaml`; set `OPENREAL2SIM_KEYS` or pass `--key` through `motion_manager.py` to limit processing.

---

## Outputs

Per key, after running the pipeline you should find:

- `outputs/<key>/scene/scene.pkl`
  - `motion.hand_kpts`: `[N,21,2]` MANO keypoints (image plane)
  - `motion.hand_global_orient`: rotation matrices per frame
  - `motion.hand_masks`: binary hand silhouettes
- `outputs/<key>/simulation/scene.json`
  - `manipulated_oid`, `traj_key`, `start_frame_idx`, `task_type`, `start_related`, `end_related`, `gripper_closed`
  - `objects[oid].final_trajs`, 
  - `objects[manipulated_oid].grasp_point`, `grasp_direction`, `grasps`, `rescored_grasps`
- `outputs/<key>/motion/scene_end.glb`: fused background + final manipulated mesh.
- `outputs/<key>/grasps/*.npy`: raw GraspNet candidates per object.
- `outputs/<key>/simulation/debug/grasp_point_visualization_*.png`: sanity-check renders.

These artifacts are consumed by the Isaac Lab / ManiSkill / MuJoCo simulation runners during trajectory replay and policy datasets export.

---

