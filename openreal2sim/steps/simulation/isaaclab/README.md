# IsaacLab

## Installation

**Pull the pre-built docker image**

Pull the pre-built image rather than building it; build only when you change a Dockerfile:
   ```bash
   python3 init.py setup
   ```
   The packages are private until Esteban makes them public, so until then run
   `docker login ghcr.io` first with a GitHub token that has the `read:packages` scope.

**On the host machine and before entering the container**, run
```
xhost +local:
```
This is to allow the container to access the host's X server for IsaacSim GUI.

Then run the scripts below from the repository root on the host, each in a fresh
container:
```
python3 init.py run simulation -- python <path_to_script>.py
```
Add `--gpus 0,1` to pick GPUs, or use `python3 init.py shell simulation` for an
interactive prompt. Without `init.py`:
```
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose -p "$USER" -f docker/compose.yml up -d simulation
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose -p "$USER" -f docker/compose.yml exec simulation bash
```

## Pipeline

The pipeline consists of three main stages:
1. **USD Conversion** - Converts scene data to USD format for simulation
2. **Heuristic Manipulation** - Performs object manipulation using heuristic policies with grasping and motion planning. This will create a new folder in `task` and `h5py` folder for each key. In the task folder, a json is formulated, and required assets for the scene are structured. 
3. **Randomized Rollout** - Generates randomized scene configurations and collects successful manipulation trajectories. This will add demos to the `task` folder and trajectory configs to the task config, and add new h5py file to the `h5py` folder.

## Code Structure

```
isaaclab/
├── sim_agent.sh                    # Pipeline script that runs all stages
├── sim_pure_render_agent.sh        # Calibration and direct rendering pipeline
├── sim_base.py                     # Base simulator class with common functionality
├── sim_env_factory.py              # Environment factory for creating simulation environments
├── sim_heuristic_manip.py          # Heuristic manipulation policy implementation
├── sim_randomize_rollout.py        # Randomized rollout for data augmentation
├── sim_calib.py                    # Camera and robot calibration
├── sim_direct_render.py            # Direct rendering of generated trajectories
├── sim_utils/                      # Utility functions for simulation
│   ├── usd_conversion.py           # Mesh to USD conversion (first pipeline stage)
│   ├── grasp_group_utils.py        # Utilities for grasp group operations
│   ├── calibration_utils.py        # Extrinsics loading and robot pose calibration
│   ├── sim_utils.py                # General simulation utilities
│   └── transform_utils.py          # Coordinate transformation utilities
├── envs/                           # Environment configuration and utilities
│   ├── task_cfg.py                 # Task configuration classes (TaskCfg, TrajectoryCfg, etc.)
│   ├── task_construct.py           # Task construction and serialization utilities
│   ├── randomizer.py               # Scene randomization for data augmentation
│   └── running_cfg.py              # Runtime configuration management
└── tools/
    └── demo_judge.py               # Gradio interface for filtering collected demos
```

## Running the pipeline
Please run `bash openreal2sim/steps/simulation/isaaclab/sim_agent.sh` to collect the demo. This will iterate through the keys in `config/config.yaml` and run the three stages for each one.

## Demo Judging
We provide an easy-to-use interace for filtering out these bad demos. Please run `python openreal2sim/steps/simulation/isaaclab/tools/demo_judge.py`. On the gui, you can first choose whether to preserve or delete one demo, and final confirms deletion after browsing all demos for one task.


## Running Config Explanation
There are two editable configs for this script:

First, running device and start/end options defined in the bash script.

Second, the randomizer config, environment num and total trajectry num are all defined in the `running_cfg.py` file. Please edit that file for your own needs.




#### TaskCfg (Main Task Configuration)

The top-level configuration that defines a complete manipulation task:

- **task_key** (str): Unique identifier for the task
- **task_id** (int): Sequential task ID
- **task_desc** (List[str]): Description of the task
- **task_type** (TaskType): Type of task (SIMPLE_PICK, SIMPLE_PICK_PLACE, TARGETTED_PICK_PLACE)
- **background_cfg** (BackgroundCfg): Background configuration
- **camera_info** (CameraInfo): Camera intrinsic and extrinsic parameters
- **manipulated_oid** (int): Object ID of the manipulated object
- **start_related** (List[int]): List of object IDs related to the start state
- **end_related** (List[int]): List of object IDs related to the end state
- **objects** (List[ObjectCfg]): List of all objects in the scene
- **reference_trajectory** (List[TrajectoryCfg], optional): Reference trajectory generated by heuristic manipulation
- **generated_trajectories** (List[TrajectoryCfg]): List of randomized trajectories for data augmentation

#### TrajectoryCfg (Trajectory Configuration)

Defines a specific manipulation trajectory:

- **robot_pose** (List[float]): Robot base pose [x, y, z, qw, qx, qy, qz] in world frame
- **object_poses** (Dict[int, List[float]]): Initial poses of all objects, keyed by object ID
- **object_trajectory** (List[List[float]]): Object trajectory as sequence of poses [x, y, z, qw, qx, qy, qz]
- **final_gripper_close** (bool): Whether gripper should be closed at the end
- **success_metric** (SuccessMetric): Criteria for task success
- **pregrasp_pose** (List[float], optional): Pre-grasp end-effector pose in world frame
- **grasp_pose** (List[float], optional): Grasp end-effector pose in world frame
- **robot_type** (RobotType, optional): Type of robot (FRANKA or UR5)

#### SuccessMetric (Success Criteria)

Defines how task success is evaluated:

- **success_metric_type** (SuccessMetricType): Type of success metric
  - `SIMPLE_LIFT`: Object must be lifted by a certain height
  - `TARGET_POINT`: Object must reach a specific target pose
  - `TARGET_PLANE`: Object must reach a target plane (height)
- **final_gripper_close** (bool): Whether gripper should be closed at success
- **lift_height** (float, optional): Required lift height for SIMPLE_LIFT
- **ground_value** (float, optional): Target height for TARGET_PLANE
- **end_pose** (List[float], optional): Target pose [x, y, z, qw, qx, qy, qz] for TARGET_POINT

#### ObjectCfg (Object Configuration)

Defines an object in the scene:

- **object_id** (int): Unique object identifier
- **object_name** (str): Name of the object
- **mesh_path** (str): Path to object mesh file (GLB format)
- **usd_path** (str): Path to object USD file

#### BackgroundCfg (Background Configuration)

Defines the background of the scene:

- **background_rgb_path** (str): Path to background RGB image
- **background_mesh_path** (str): Path to background mesh file (GLB format)
- **background_usd_path** (str): Path to background USD file
- **background_point** (List[float]): Background point [x, y, z]

#### CameraInfo (Camera Configuration)

Camera intrinsic and extrinsic parameters:

- **width, height** (float): Image dimensions
- **fx, fy** (float): Focal lengths
- **cx, cy** (float): Principal point
- **camera_opencv_to_world** (List[List[float]]): 4x4 transformation matrix from OpenCV to world frame
- **camera_position** (List[float]): Camera position [x, y, z]
- **camera_heading_wxyz** (List[float]): Camera orientation quaternion [w, x, y, z]

#### Configuration Workflow

1. **Initial Task Creation**: `construct_task_config()` creates a `TaskCfg` from scene reconstruction data
2. **Reference Trajectory**: `add_reference_trajectory()` adds the successful trajectory/trajectories from heuristic manipulation
3. **Data Augmentation**: `add_generated_trajectories()` adds randomized trajectories from rollout stage
4. **Serialization**: Task configs are saved as JSON files in `tasks/<key>/task.json`
5. **Loading**: `load_task_cfg()` reconstructs `TaskCfg` objects from JSON files

#### Example Task Configuration Structure

```json
{
  "task_key": "demo_video",
  "task_id": 0,
  "task_type": "TARGETTED_PICK_PLACE",
  "manipulated_oid": 2,
  "objects": [
    {
      "object_id": 2,
      "object_name": "pen",
      "mesh_path": "tasks/demo_video/object_2.glb",
      "usd_path": "tasks/demo_video/object_2.usd"
    }
  ],
  "background_cfg": {
    "background_rgb_path": "tasks/demo_video/bg_rgb.jpg",
    "background_mesh_path": "tasks/demo_video/background.glb",
    "background_usd_path": "tasks/demo_video/background.usd",
    "background_point": [0.0, 0.0, 0.0]
  },
  "reference_trajectory": {
    "robot_pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "object_poses": {"2": [0.1, 0.2, 0.05, 1.0, 0.0, 0.0, 0.0]},
    "object_trajectory": [[...], [...], ...],
    "pregrasp_pose": [0.1, 0.2, 0.15, 1.0, 0.0, 0.0, 0.0],
    "grasp_pose": [0.1, 0.2, 0.1, 1.0, 0.0, 0.0, 0.0],
    "success_metric": {
      "success_metric_type": "TARGET_POINT",
      "end_pose": [0.3, 0.4, 0.1, 1.0, 0.0, 0.0, 0.0],
      "final_gripper_close": false
    }
  },
  "generated_trajectories": [...]
}
```
