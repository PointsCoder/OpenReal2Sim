"""
Direct Render Simulation: Heuristic Manip + Randomize Rollout Fusion
======================================================================

This script combines heuristic manipulation and randomize rollout approaches:
- Uses heuristic manipulation to find optimal grasp poses
- Uses randomization to generate diverse trajectory variations
- Uses Curobo for motion planning instead of physics simulation
- Directly sets robot/object states for rendering (no physics rollout)

Key Features:
-------------
1. **Selective Physics Simulation**:
   - ENABLED: After reset (to settle objects) and at last step if gripper opens (3 steps)
   - DISABLED: During trajectory execution (direct state setting for efficiency)

2. **Curobo Motion Planning**: Uses Curobo to plan dense, collision-free joint
   trajectories from sparse waypoints (pre-grasp, grasp, lift, trajectory goals).

3. **Fixed EEF-Object Transform**: After grasping, assumes the object is rigidly
   attached to the end-effector, computing object poses from EEF poses.

4. **Randomization**: Generates diverse trajectory variations by randomizing
   object poses, robot poses, and grasp poses.

Workflow:
---------
1. **Heuristic Manipulation Stage**:
   - Initialize all environments
   - Use GraspNet to generate grasp proposals
   - Select best grasp pose via physical trial-and-error
   - Generate initial task config with reference trajectories

2. **Randomization Stage**:
   - Load task config
   - Generate randomized trajectory variations (object/robot poses)
   - For each randomized config:
     a. Reset with physics enabled (settle objects, 10 steps)
     b. **Move object away** (100m up) to avoid collisions during FK
     c. Compute sparse EEF waypoints (pre-grasp, grasp, lift, trajectory)
     d. Use Curobo to plan dense joint trajectories
     e. Compute object poses assuming fixed EEF-object transform (via FK with object away)
     f. Directly set robot/object states and render (object back in scene)
     g. If last step has open gripper: enable physics for 3 steps
     h. Record observations

3. **Output**:
   - HDF5 files with observation data
   - Updated task config with generated trajectories

Usage:
------
```bash
python sim_direct_render.py --key <scene_key> --num_envs 4
```

Configuration:
--------------
All configuration is loaded from `envs/running_cfg.py`:
- `HeuristicConfig`: num_envs, num_trials, grasp_num, robot, goal_offset
- `RandomizerConfig`: randomization parameters (grid, angles, etc.)
- `RolloutConfig`: total_num (total trajectories to generate)
- `SimulationConfig`: physics_freq, decimation, save_interval

Physics and Collision Control Strategy:
---------------------------------------
**ENABLED (rigid body dynamics + collision detection)**:
1. After reset: Enable physics/collision → step 10 times to settle objects
2. At last trajectory step IF gripper is open: Enable physics/collision → step 3 times

**DISABLED (spatial separation)**:
- During FK computation: Move manipulated object 100m up to avoid collisions
  - Object is moved far away from the scene
  - No USD attribute modification (avoids GPU memory corruption)
  - Robot can move freely for FK computation without colliding with object

Implementation:
- `disable_physics_and_collision()`: Moves object 100m up
- `enable_physics_and_collision()`: Re-enables physics (object position set during rendering)
- Spatial separation → no collisions during FK, preserves GPU state

This hybrid approach:
- ✅ Stable initial conditions (physics settling after reset)
- ✅ No collision interference during FK (object moved away)
- ✅ Realistic object release (physics when gripper opens)
- ✅ Efficient execution (direct state control during main trajectory)
- ✅ No GPU corruption (position change only, not USD modification)
"""
from __future__ import annotations

# ─────────── AppLauncher ───────────
import argparse
from typing import Optional, List
from pathlib import Path
import os
import numpy as np
import torch
import transforms3d
from isaaclab.app import AppLauncher
import sys

file_path = Path(__file__).resolve()
sys.path.append(str(file_path.parent))


from envs.task_cfg import TrajectoryCfg, TaskType, SuccessMetric, SuccessMetricType, RobotType
from envs.task_construct import load_task_cfg, add_generated_trajectories
from envs.running_cfg import get_randomizer_config, get_rollout_config, get_direct_render_config, get_simulation_config
from envs.randomizer import Randomizer

# ─────────── CLI ───────────
parser = argparse.ArgumentParser("sim_direct_render")
parser.add_argument("--key", type=str, default="demo_video", help="scene key (outputs/<key>)")
parser.add_argument("--robot", type=str, default="franka")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments")
parser.add_argument("--total_num", type=int, default=None, help="Total number of trajectories")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.headless = True
app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

# ─────────── Runtime imports ───────────
from isaaclab.utils.math import subtract_frame_transforms

# Curobo imports
from curobo.types.robot import JointState
from curobo.types.math import Pose
from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

# ─────────── Simulation environments ───────────
from sim_base_demo import BaseSimulator
from sim_env_factory_demo import make_env, get_prim_name_from_oid


from sim_utils_demo.transform_utils import pose_to_mat, mat_to_pose
from sim_utils_demo.sim_utils import load_sim_parameters

BASE_DIR = Path.cwd()
out_dir = BASE_DIR / "outputs"


# ──────────────────────────── Direct Render Simulator ────────────────────────────

def compute_poses_from_traj_cfg(traj_cfg_list):
    """
    Extract poses and trajectories from a list of TrajectoryCfg objects.
    
    Args:
        traj_cfg_list: List of TrajectoryCfg objects
        
    Returns:
        robot_poses_list: List of robot poses [7] for each trajectory
        object_poses_dict: Dict mapping oid -> list of (pos, quat) tuples
        object_trajectory_list: List of object trajectories
        final_gripper_state_list: List of final gripper states
        grasping_phase_list: List of grasping phases
        placing_phrase_list: List of placing phases
    """
    robot_poses_list = []
    object_poses_dict = {}  # {oid: [(pos, quat), ...]}
    object_trajectory_list = []
    final_gripper_state_list = []
    pregrasp_pose_list = []
    grasp_pose_list = []
    end_pose_list = []


    for traj_cfg in traj_cfg_list:
        robot_poses_list.append(traj_cfg.robot_pose)
        
        # Extract object poses: traj_cfg.object_poses is a list of (oid, pose) tuples
        for oid in traj_cfg.object_poses.keys():
            pose = traj_cfg.object_poses[oid]
            oid_str = str(oid)
            if oid_str not in object_poses_dict:
                object_poses_dict[oid_str] = []
            object_poses_dict[oid_str].append(np.array(pose, dtype=np.float32))
        traj = []
        for i in range(len(traj_cfg.object_trajectory)):
            mat = pose_to_mat(traj_cfg.object_trajectory[i][:3], traj_cfg.object_trajectory[i][3:7])
            traj.append(mat)
        object_trajectory_list.append(np.array(traj, dtype=np.float32))
        final_gripper_state_list.append(traj_cfg.final_gripper_close)
        pregrasp_pose_list.append(np.array(traj_cfg.pregrasp_pose, dtype=np.float32))
        grasp_pose_list.append(np.array(traj_cfg.grasp_pose, dtype=np.float32))
        
        end_pose_list.append(np.array(traj_cfg.success_metric.end_pose, dtype=np.float32))

    return robot_poses_list, object_poses_dict, object_trajectory_list, final_gripper_state_list, pregrasp_pose_list, grasp_pose_list, end_pose_list


class DirectRenderSimulator(BaseSimulator):
    """
    Direct rendering simulator that combines heuristic manipulation and randomize rollout.

    Key differences from RandomizeExecution:
    - Disables physics simulation after grasp
    - Uses Curobo to plan dense joint trajectories
    - Directly sets robot and object states for rendering
    - Assumes fixed eef-object relative pose after grasping
    """

    def __init__(self, sim, scene, sim_cfgs: dict, data_dir: Path, record: bool = True,
                 args_cli: Optional[argparse.Namespace] = None, goal_offset: float = 0.03,
                 save_interval: int = 1, decimation: int = 1, gripper_close_width: float = 0.002):
        robot_pose = torch.tensor(
            sim_cfgs["robot_cfg"]["robot_pose"],
            dtype=torch.float32,
            device=sim.device,
        )
        super().__init__(
            sim=sim, sim_cfgs=sim_cfgs, scene=scene, args=args_cli,
            robot_pose=robot_pose, cam_dict=sim_cfgs["cam_cfg"],
            out_dir=out_dir, img_folder=args_cli.key, data_dir=data_dir,
            enable_motion_planning=True,  # Enable Curobo
            set_physics_props=True, debug_level=0,
            save_interval=save_interval,
            decimation=decimation,
        )

        self.selected_object_id = sim_cfgs["demo_cfg"]["manip_object_id"]
        self._selected_object_id = str(self.selected_object_id)
        self._update_object_prim()
        self.record = record
        self.task_type = sim_cfgs["demo_cfg"]["task_type"]
        self.goal_offset = [0, 0, goal_offset]

        # Update gripper_close_tensor with configured width
        # BaseSimulator initializes it as zeros, we override with configured value
        self.gripper_close_tensor = gripper_close_width * torch.ones(
            (self.num_envs, len(self.robot_gripper_cfg.joint_ids)),
            device=self.robot.device,
        )

        # Will be populated by run_batch_trajectory
        self.traj_cfg_list = None
        self.robot_poses_list = None
        self.object_poses_dict = None
        self.object_trajectory_list = None
        self.final_gripper_state_list = None
        self.pregrasp_pose_list = None
        self.grasp_pose_list = None
        self.end_pose_list = None

        # Track physics state
        self.physics_enabled = True

   

    def reset(self, env_ids=None, enable_physics=True):
        """
        Reset environments with randomized poses from traj_cfg_list.

        Args:
            env_ids: Environment IDs to reset (None for all)
            enable_physics: Whether to enable physics simulation after reset
        """
        super().reset(env_ids)

        # Track physics state (for now we just note it, actual physics control happens elsewhere)
        self.physics_enabled = enable_physics
        device = self.object_prim.device
        if env_ids is None:
            env_ids_t = self._all_env_ids.to(device)
        else:
            env_ids_t = torch.as_tensor(env_ids, device=device, dtype=torch.long).view(-1)

        M = int(env_ids_t.shape[0])
        env_origins = self.scene.env_origins.to(device)[env_ids_t]

        # Set poses for all objects from object_poses_dict
        for oid in self.object_poses_dict.keys():
            prim_name = get_prim_name_from_oid(str(oid))
            object_prim = self.scene[prim_name]

            if len(self.object_poses_dict[oid]) == 0:
                continue

            pos = np.array(self.object_poses_dict[oid], dtype=np.float32)[env_ids_t.cpu().numpy(), :3]
            quat = np.array(self.object_poses_dict[oid], dtype=np.float32)[env_ids_t.cpu().numpy(), 3:7]

            object_pose = torch.zeros((M, 7), device=device, dtype=torch.float32)
            object_pose[:, :3] = env_origins + torch.tensor(pos, dtype=torch.float32, device=device)
            object_pose[:, 3:7] = torch.tensor(quat, dtype=torch.float32, device=device)

            object_prim.write_root_pose_to_sim(object_pose, env_ids=env_ids_t)
            object_prim.write_root_velocity_to_sim(
                torch.zeros((M, 6), device=device, dtype=torch.float32), env_ids=env_ids_t
            )
            object_prim.write_data_to_sim()

        # Set robot poses
        rp_local = np.array(self.robot_poses_list, dtype=np.float32)
        robot_pose_world = rp_local.copy()
        robot_pose_world[:, :3] = env_origins.cpu().numpy() + robot_pose_world[env_ids_t.cpu().numpy(), :3]

        self.robot.write_root_pose_to_sim(
            torch.tensor(robot_pose_world, dtype=torch.float32, device=device), env_ids=env_ids_t
        )
        self.robot.write_root_velocity_to_sim(
            torch.zeros((M, 6), device=device, dtype=torch.float32), env_ids=env_ids_t
        )

        # Set robot joints to default
        joint_pos = self.robot.data.default_joint_pos.to(self.robot.device)[env_ids_t]
        joint_vel = self.robot.data.default_joint_vel.to(self.robot.device)[env_ids_t]
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_t)
        self.robot.write_data_to_sim()

      

# ──────────────────────────── Simple Grasp Selection ────────────────────────────
def select_best_grasp_simple(sim_cfgs: dict, scene):
    """
    Simple grasp selection: load grasps, rescore all, select top-1.
    No physical rollout, just heuristic scoring.

    Args:
        sim_cfgs: Simulation configs
        scene: Isaac scene (for object pose)

    Returns:
        best_grasp: Dict with 'position' and 'quaternion' (in canonical object frame)
        cur_4x4_mat: Current object transformation matrix (for converting back)
    """
    from demo.sim_utils_demo.grasp_group_utils import GraspGroup

    # Load grasp path from config
    grasp_path = sim_cfgs["demo_cfg"]["grasp_path"]

    if grasp_path is None or not os.path.exists(grasp_path):
        raise FileNotFoundError(f"Grasps file not found: {grasp_path}")

    print(f"[INFO] Loading grasps from: {grasp_path}")

    # Load and transform grasps
    gg = GraspGroup().from_npy(npy_file_path=grasp_path)
    gg = gg.to_world_transform()

    if len(gg) == 0:
        raise ValueError(f"No grasp proposals found in: {grasp_path}")

    print(f"[INFO] Loaded {len(gg)} grasps, applying heuristic scoring to all...")

    # Get current object pose to transform grasps
    # Assuming object is in scene (we'll get it from the first env)
    selected_object_id = sim_cfgs["demo_cfg"]["manip_object_id"]
    from demo.sim_env_factory_demo import get_prim_name_from_oid
    prim_name = get_prim_name_from_oid(str(selected_object_id))
    object_prim = scene[prim_name]

    current_object_pose = object_prim.data.root_state_w[:, :7].cpu().numpy()
    cur_4x4_mat = np.eye(4, dtype=np.float32)
    cur_4x4_mat[:3, :3] = transforms3d.quaternions.quat2mat(current_object_pose[0, 3:7])
    cur_4x4_mat[:3, 3] = current_object_pose[0, :3] - scene.env_origins.cpu().numpy()[0]

    # Transform grasps to current object pose
    gg = gg.transform(cur_4x4_mat.astype(np.float32))

    # Apply heuristic scoring to all grasps
    # First rescore based on original scores only
    gg = gg.rescore(only_score=True)

    # Then rescore with direction hint (prefer downward approach)
    gg = gg.rescore(direction_hint=[0, 0, -1], reorder_num=10)

    print(f"[INFO] Heuristic scoring complete, {len(gg)} grasps rescored and sorted")

    # Select top-1 grasp
    print(f"[INFO] Using top-ranked grasp (index 0 after heuristic scoring)")
    p_w, q_w = gg.retrieve_grasp_group(0)

    # Convert to local frame (relative to env origin)
    env_origin = scene.env_origins[0].cpu().numpy()
    p_local = p_w - env_origin

    # Convert grasp pose back to canonical object frame (object at identity pose)
    # This matches heuristic_manip.py lines 946-960
    grasp_mat = np.eye(4, dtype=np.float32)
    grasp_mat[:3, :3] = transforms3d.quaternions.quat2mat(q_w)
    grasp_mat[:3, 3] = p_local

    # Transform back: ori_grasp_mat = inv(cur_4x4_mat) @ grasp_mat
    ori_grasp_mat = np.linalg.inv(cur_4x4_mat) @ grasp_mat
    ori_grasp_pos = ori_grasp_mat[:3, 3]
    ori_grasp_quat = transforms3d.quaternions.mat2quat(ori_grasp_mat[:3, :3])

    return {
        'position': ori_grasp_pos.astype(np.float32),
        'quaternion': ori_grasp_quat.astype(np.float32)
    }, cur_4x4_mat


# ──────────────────────────── Entry Point ────────────────────────────
def sim_direct_render(key: str, args_cli: argparse.Namespace):

    args_cli.key = key
    sim_cfgs = load_sim_parameters(BASE_DIR, key)

    # Create a temporary scene to get object pose for grasp transformation
    from demo.envs.task_construct import construct_task_config
    import json

    scene_json_path = BASE_DIR / "outputs" / key / "simulation" / "scene.json"
    if not scene_json_path.exists():
        raise FileNotFoundError(f"Scene file not found: {scene_json_path}")

    scene_dict = json.load(open(scene_json_path))

    # Check if task config already exists
    task_json_path = BASE_DIR / "tasks" / key / "task.json"

   
    print(f"[INFO] Creating new task config from scene...")

    # Create minimal environment to get scene
    direct_render_cfg = get_direct_render_config(key)
    simulation_cfg = get_simulation_config(key)
    
    # Ensure scene context is initialized before calling select_best_grasp_simple
    from demo.sim_env_factory_demo import init_scene_from_scene_dict
    init_scene_from_scene_dict(
        scene=sim_cfgs["scene_cfg"],
        cfgs=sim_cfgs,
        use_ground_plane=False,
    )
    
    env, _ = make_env(
        cfgs=sim_cfgs, num_envs=direct_render_cfg.num_envs,
        device=args_cli.device,
        bg_simplify=False,
        physics_freq=simulation_cfg.physics_freq,
        decimation=simulation_cfg.decimation,
    )

    # Reset environment to initialize scene objects
    env.reset()
    
    # Select best grasp using simple rescoring
    best_grasp, cur_4x4_mat = select_best_grasp_simple(sim_cfgs, env.scene)

    print(f"[INFO] Selected grasp (in canonical frame) - Position: {best_grasp['position']}, Quaternion: {best_grasp['quaternion']}")

    # Read actual object poses from environment before closing
    object_poses = {}
    from demo.sim_env_factory_demo import get_prim_name_from_oid
    for obj in sim_cfgs["scene_cfg"]["objects"].values():
        prim_name = get_prim_name_from_oid(str(obj["oid"]))
        if prim_name in env.scene.keys():
            object_prim = env.scene[prim_name]
            current_pose = object_prim.data.root_state_w[:, :7].cpu().numpy()[0]
            # Convert to canonical frame (relative to environment origin)
            pos = current_pose[:3] - env.scene.env_origins.cpu().numpy()[0]
            quat = current_pose[3:7]
            object_poses[obj["oid"]] = np.concatenate([pos, quat]).tolist()
        else:
            # Fallback to identity pose if object not found
            object_poses[obj["oid"]] = [0, 0, 0, 1, 0, 0, 0]

    # Close temporary environment
    env.close()

    # Create task config with the selected grasp
    task_base_folder = BASE_DIR / "tasks"
    task_cfg, _ = construct_task_config(key, scene_dict, task_base_folder)

    # Add reference trajectory with the selected grasp

    from demo.envs.task_cfg import TrajectoryCfg, SuccessMetric, SuccessMetricType, RobotType

    # Get robot type
    robot_type_str = args_cli.robot if hasattr(args_cli, 'robot') else sim_cfgs["robot_cfg"].get("robot_type", "franka")
    robot_type = RobotType.FRANKA if robot_type_str.lower() == "franka" else RobotType.UR5

    # Load trajectory from scene if available
    traj_path = sim_cfgs["demo_cfg"]["traj_path"]
    if traj_path and os.path.exists(traj_path):
        object_trajectory = np.load(traj_path).astype(np.float32)
        pose_quat_traj = []
        for pose_mat in object_trajectory:
            pose, quat = mat_to_pose(pose_mat)
            pose_quat = np.concatenate([np.array(pose), np.array(quat)])
            pose_quat_traj.append(pose_quat)
        pose_quat_traj = np.array(pose_quat_traj).reshape(-1, 7).tolist()
    else:
        # Create a simple lift trajectory if no trajectory file
        pose_quat_traj = [
            [0, 0, 0, 1, 0, 0, 0],  # Initial
            [0, 0, 0.15, 1, 0, 0, 0],  # Lift up 15cm
        ]

    # Create pre-grasp pose (grasp - offset along approach axis)
    from demo.sim_utils_demo.transform_utils import grasp_approach_axis_batch
    approach_axis = grasp_approach_axis_batch(best_grasp['quaternion'].reshape(1, 4))[0]
    pregrasp_pos = best_grasp['position'] - 0.12 * approach_axis
    pregrasp_pose = np.concatenate([pregrasp_pos, best_grasp['quaternion']])

    # Create grasp pose (apply grasp_delta along approach axis)
    print(f"[INFO] Grasp delta: {direct_render_cfg.grasp_delta}")
    grasp_pos = best_grasp['position'] + direct_render_cfg.grasp_delta * approach_axis
    grasp_pose = np.concatenate([grasp_pos, best_grasp['quaternion']])

    # Get robot pose from config
    robot_pose = sim_cfgs["robot_cfg"]["robot_pose"]

    # Get final object pose from trajectory
    # Following heuristic_manip.py lines 1070-1091
    final_gripper_close = sim_cfgs["demo_cfg"].get("final_gripper_closed", True)
    # Transform the last trajectory pose back to canonical frame
    if len(pose_quat_traj) > 0:
        # Last pose in trajectory
        final_pose_mat = pose_to_mat(pose_quat_traj[-1][:3], pose_quat_traj[-1][3:7])
        
        final_pos = final_pose_mat[:3, 3]
        final_quat = transforms3d.quaternions.mat2quat(final_pose_mat[:3, :3])
        end_pose = np.concatenate([final_pos, final_quat]).tolist()
    else:
        # Fallback: simple lift
        end_pose = [0, 0, 0.15, 1, 0, 0, 0]

    # Create success metric based on task type (matching heuristic_manip.py)
    if task_cfg.task_type == TaskType.TARGETTED_PICK_PLACE:
        success_metric = SuccessMetric(
            success_metric_type=SuccessMetricType.TARGET_POINT,
            end_pose=end_pose,
            final_gripper_close=final_gripper_close,
        )
    elif task_cfg.task_type == TaskType.SIMPLE_PICK_PLACE:
        # For simple pick-place, use ground plane as target
        ground_value = float(end_pose[2])  # Z-coordinate of final pose
        success_metric = SuccessMetric(
            success_metric_type=SuccessMetricType.TARGET_PLANE,
            ground_value=ground_value,
            final_gripper_close=final_gripper_close,
            end_pose=end_pose
        )
    else:  # SIMPLE_PICK
        success_metric = SuccessMetric(
            success_metric_type=SuccessMetricType.SIMPLE_LIFT,
            lift_height=0.05,
            final_gripper_close=final_gripper_close,
            end_pose=end_pose
        )

    # Create trajectory config
    traj_cfg = TrajectoryCfg(
        robot_pose=robot_pose,
        object_poses=object_poses,
        object_trajectory=pose_quat_traj,
        final_gripper_close=final_gripper_close,
        success_metric=success_metric,
        pregrasp_pose=pregrasp_pose.tolist(),
        grasp_pose=grasp_pose.tolist(),
        robot_type=robot_type,
    )

    # Add reference trajectory
    from demo.envs.task_construct import add_reference_trajectory
    add_reference_trajectory(task_cfg, [traj_cfg], task_json_path.parent)

    print(f"[INFO] Created task config with selected grasp")
    env.close()

   

    return True


def main():
    key = args_cli.key
    sim_direct_render(key, args_cli)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERR] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        simulation_app.close()
