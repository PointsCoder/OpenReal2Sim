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
                 save_interval: int = 1, decimation: int = 1, gripper_close_width: float = 0.002,
                 require_gravity_final: bool = False):
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
        self.require_gravity_final = require_gravity_final
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

        
    def set_gravity_on(self):
        from pxr import UsdPhysics, PhysxSchema
        B = self.scene.num_envs
        env_base_path = "/World/envs"
        stage = self.sim.stage
        for env_id in range(B):
            obj_name = self.object_prim.cfg.prim_path.split("/")[-1]
            env_path = f"{env_base_path}/env_{env_id}"
            obj_prim_path = f"{env_path}/{obj_name}"
            obj_prim = stage.GetPrimAtPath(obj_prim_path)
            if obj_prim.IsValid():
                rigid_body_schema = PhysxSchema.PhysxRigidBodyAPI.Apply(obj_prim)
                rigid_body_schema.CreateDisableGravityAttr().Set(False)

    def set_gravity_off(self):
        from pxr import PhysxSchema
        B = self.scene.num_envs
        env_base_path = "/World/envs"
        stage = self.sim.stage
        for env_id in range(B):
            obj_name = self.object_prim.cfg.prim_path.split("/")[-1]
            env_path = f"{env_base_path}/env_{env_id}"
            obj_prim_path = f"{env_path}/{obj_name}"
            obj_prim = stage.GetPrimAtPath(obj_prim_path)
            if obj_prim.IsValid():
                rigid_body_schema = PhysxSchema.PhysxRigidBodyAPI.Apply(obj_prim)
                rigid_body_schema.CreateDisableGravityAttr().Set(True)

    def compute_components(self):
        """Extract components from traj_cfg_list"""
        (self.robot_poses_list, self.object_poses_dict, self.object_trajectory_list,
         self.final_gripper_state_list, self.pregrasp_pose_list, self.grasp_pose_list,
         self.end_pose_list) = compute_poses_from_traj_cfg(self.traj_cfg_list)

    def compute_sparse_eef_poses(self):
        """
        Compute sparse eef poses for the trajectory:
        - pre-grasp
        - grasp
        - lift
        - trajectory waypoints (from object_trajectory_list)

        Returns:
            sparse_eef_poses: List[np.ndarray] - each element is [N, 7] (pos + quat)
        """
        B = self.scene.num_envs
        T = self.object_trajectory_list[0].shape[0]

        # Compute goal trajectory in world frame
        self.compute_object_goal_traj()
        object_real_poses = self.object_prim.data.root_state_w[:, :7].cpu().numpy()
        # For each env, compute sparse eef poses
        sparse_eef_poses = []
        
        for b in range(B):
            poses = []

            # 1. Pre-grasp pose (already in local frame)
            pregrasp_pose = self.pregrasp_pose_list[b]  # [7] pos + quat
            poses.append(pregrasp_pose)

            # 2. Grasp pose (already in local frame)
            grasp_pose = self.grasp_pose_list[b]  # [7]
            poses.append(grasp_pose)

            # 3. Lift pose (grasp + offset in z)
            lift_pose = grasp_pose.copy()
            lift_pose[2] += self.goal_offset[2]  # Add lift height
            poses.append(lift_pose)

            # 4. Trajectory waypoints - compute eef poses from object trajectory
            # Compute T_ee_in_obj at grasp moment
            T_grasp_ee = pose_to_mat(grasp_pose[:3], grasp_pose[3:7])
            T_grasp_obj = pose_to_mat(object_real_poses[b, :3], object_real_poses[b, 3:7])
            T_ee_in_obj = np.linalg.inv(T_grasp_obj) @ T_grasp_ee

            # For each waypoint in trajectory, compute eef pose
            # Sample trajectory waypoints (every N steps)
            
            t_iter = list(range(1, T))
            if t_iter[-1] != T-1:
                t_iter.append(T-1)

            for t in t_iter:
                T_obj_goal = self.obj_goal_traj_w[b, t]  # [4, 4]
                T_obj_goal[:3, 3] -= self.scene.env_origins[b].cpu().numpy()
                T_ee_goal = T_obj_goal @ T_ee_in_obj
                ee_pos, ee_quat = mat_to_pose(T_ee_goal)

                ee_pose = np.concatenate([ee_pos, ee_quat])
                poses.append(ee_pose)

            sparse_eef_poses.append(np.array(poses, dtype=np.float32))
        
        actions_arr = np.array(sparse_eef_poses, dtype=np.float32)
        self.actions_arr = actions_arr.transpose(1, 0, 2)
        return sparse_eef_poses

    def compute_object_goal_traj(self):
        """Compute object goal trajectory in world frame"""
        B = self.scene.num_envs
        obj_state = self.object_prim.data.root_state_w[:, :7]
        obj_state_np = obj_state.detach().cpu().numpy().astype(np.float32)
        offset_np = np.asarray(self.goal_offset, dtype=np.float32)
     

        origins = self.scene.env_origins.detach().cpu().numpy().astype(np.float32)
        obj_state_np[:, :3] -= origins

        T = self.object_trajectory_list[0].shape[0]
        obj_goal = np.zeros((B, T, 4, 4), dtype=np.float32)
   
        for b in range(B):
            T_init = pose_to_mat(obj_state_np[b, :3], obj_state_np[b, 3:7])
            for t in range(1, T):
                goal_4x4 = self.object_trajectory_list[b][t] @ T_init
                goal_4x4[:3, 3] += origins[b]
                if t < T-1:
                    goal_4x4[:3, 3] += offset_np
                obj_goal[b, t] = goal_4x4

        self.obj_goal_traj_w = obj_goal

    def refine_grasp_pose(self, init_root_pose, grasp_pose, pregrasp_pose):
        """
        Refine grasp and pregrasp poses based on the difference between 
        reference root pose (from traj) and current root pose.
        
        Args:
            init_root_pose: (B, 7) reference root pose [pos(3) + quat(4)] from traj
            grasp_pose: (B, 3) grasp positions in world frame
            pregrasp_pose: (B, 3) pregrasp positions in world frame
        
        Returns:
            grasp_pose_w: (B, 3) refined grasp positions
            pregrasp_pose_w: (B, 3) refined pregrasp positions
        """
        # Get current root pose from API (in world frame)
        current_root_pose_w = self.object_prim.data.root_state_w[:, :7].cpu().numpy()  # (B, 7)
        env_origins = self.scene.env_origins.cpu().numpy()  # (B, 3)
        
        # Convert to numpy if needed
        if isinstance(init_root_pose, torch.Tensor):
            init_root_pose = init_root_pose.cpu().numpy()
        if isinstance(grasp_pose, torch.Tensor):
            grasp_pose = grasp_pose.cpu().numpy()
        if isinstance(pregrasp_pose, torch.Tensor):
            pregrasp_pose = pregrasp_pose.cpu().numpy()
        
        # Ensure init_root_pose has batch dimension
        if init_root_pose.ndim == 1:
            init_root_pose = init_root_pose[np.newaxis, :].repeat(grasp_pose.shape[0], axis=0)
        
        # Convert current root pose to local frame (relative to env_origin)
        current_root_pose = current_root_pose_w.copy()
        current_root_pose[:, :3] -= env_origins  # Convert to local frame
        
        # Calculate pose difference (both in local frame)
        ref_pos = init_root_pose[:, :3]  # (B, 3) - reference pose in local frame
        ref_quat = init_root_pose[:, 3:7]  # (B, 4)
        curr_pos = current_root_pose[:, :3]  # (B, 3) - current pose in local frame
        curr_quat = current_root_pose[:, 3:7]  # (B, 4)
        
        # Make copies to avoid modifying input arrays
        grasp_pose_w = grasp_pose.copy()  # (B, 3)
        pregrasp_pose_w = pregrasp_pose.copy()  # (B, 3)
        
        B = grasp_pose.shape[0]
        for b in range(B):
            # Calculate position offset
            pos_offset = curr_pos[b] - ref_pos[b]  # (3,)
            
            # Calculate rotation difference: delta_quat = curr_quat * ref_quat^-1
            ref_mat = transforms3d.quaternions.quat2mat(ref_quat[b])
            curr_mat = transforms3d.quaternions.quat2mat(curr_quat[b])
            delta_mat = curr_mat @ ref_mat.T  # rotation from ref to curr
            
            # Transform grasp and pregrasp positions
            # Note: grasp_pose and pregrasp_pose are in world frame
            # We need to convert them to local frame first, then transform, then back to world frame
            
            # Convert to local frame (relative to env_origin)
            grasp_pos_local = grasp_pose_w[b] - env_origins[b]  # (3,)
            pregrasp_pos_local = pregrasp_pose_w[b] - env_origins[b]  # (3,)
            
            # Express positions relative to reference frame
            grasp_pos_rel = grasp_pos_local - ref_pos[b]  # (3,)
            pregrasp_pos_rel = pregrasp_pos_local - ref_pos[b]  # (3,)
            
            # Apply rotation
            grasp_pos_rotated = delta_mat @ grasp_pos_rel  # (3,)
            pregrasp_pos_rotated = delta_mat @ pregrasp_pos_rel  # (3,)
            
            # Translate to current frame (in local frame)
            grasp_pos_new_local = curr_pos[b] + grasp_pos_rotated
            pregrasp_pos_new_local = curr_pos[b] + pregrasp_pos_rotated
            
            # Convert back to world frame
            grasp_pose_w[b] = grasp_pos_new_local + env_origins[b]
            pregrasp_pose_w[b] = pregrasp_pos_new_local + env_origins[b]
        
        return grasp_pose_w, pregrasp_pose_w

    def plan_dense_joint_trajectory(self, sparse_eef_poses: List[np.ndarray]):
        """
        Use Curobo to plan dense joint trajectories from sparse eef poses.

        Args:
            sparse_eef_poses: List of [N, 7] arrays (one per env)
                Structure: [pre-grasp (0), grasp (1), lift (2), trajectory waypoints (3+), ...]

        Returns:
            dense_joint_trajs: List of [M, 7] arrays (dense joint poses per env)
            success: Boolean array indicating planning success
            waypoint_indices: List of tuples (pregrasp_end, grasp_end) for each env
                pregrasp_end: index in dense trajectory where pre-grasp phase ends
                grasp_end: index in dense trajectory where grasp phase ends
            action_indices: List of [M, 1] arrays (action indices per env)
        """
  
        B = self.scene.num_envs
        dense_joint_trajs = []
        success_list = []
        gripper_close_indices = []
        action_indices_list = []

        for b in range(B):
            sparse_poses = sparse_eef_poses[b]  # [N, 7]
            N = sparse_poses.shape[0]

            # Convert sparse poses to world frame (for robot base frame)
            env_origin = self.scene.env_origins[b].cpu().numpy()

            # Plan trajectory through all waypoints
            joint_traj_segments = []
            segment_lengths = []  # Track length of each segment
            current_joint_pos = self.robot.data.joint_pos[b, self.robot_entity_cfg.joint_ids].cpu().numpy()

            # Debug: Check if we're getting the correct number of joints
            if len(current_joint_pos.shape) == 0 or current_joint_pos.shape[0] != 7:
                print(f"[ERROR] Expected 7 joint positions, but got shape {current_joint_pos.shape}")
                print(f"[DEBUG] robot_entity_cfg.joint_ids: {self.robot_entity_cfg.joint_ids}")
                print(f"[DEBUG] Available joint names: {self.robot.data.joint_names}")
                print(f"[ERROR] Skipping environment {b} due to incorrect joint configuration")
                dense_joint_trajs.append(None)
                success_list.append(False)
                continue

            all_success = True
            for i in range(N):
                # Convert from local frame to world frame
                target_pos_world = sparse_poses[i, :3] + env_origin
                target_quat_world = sparse_poses[i, 3:7]

                # Convert to base frame
                # Add batch dimension for subtract_frame_transforms (expects [B, 3] and [B, 4])
                target_pos_world_t = torch.tensor(target_pos_world, dtype=torch.float32, device=self.sim.device).unsqueeze(0)  # [1, 3]
                target_quat_world_t = torch.tensor(target_quat_world, dtype=torch.float32, device=self.sim.device).unsqueeze(0)  # [1, 4]
                robot_base_pos = self.robot.data.root_state_w[b, :3].unsqueeze(0)  # [1, 3]
                robot_base_quat = self.robot.data.root_state_w[b, 3:7].unsqueeze(0)  # [1, 4]

                target_pos_base, target_quat_base = subtract_frame_transforms(
                    robot_base_pos, robot_base_quat,
                    target_pos_world_t, target_quat_world_t
                )  # Returns [1, 3] and [1, 4]

                # # Debug: Check current_joint_pos before creating JointState
                # if i == 0 or i == 1:  # Debug first two waypoints
                #     print(f"[DEBUG] Env {b}, waypoint {i}: current_joint_pos type={type(current_joint_pos)}, shape={current_joint_pos.shape if hasattr(current_joint_pos, 'shape') else 'no shape'}, value preview={current_joint_pos if hasattr(current_joint_pos, '__len__') and len(current_joint_pos) <= 7 else 'array too long'}")

                # Plan using motion_planning_single
                joint_pos_tensor = torch.tensor(current_joint_pos, dtype=torch.float32, device=self.sim.device)
                # if i == 0 or i == 1:
                #     print(f"[DEBUG] Env {b}, waypoint {i}: joint_pos_tensor shape before unsqueeze = {joint_pos_tensor.shape}")

                start_state = JointState.from_position(
                    joint_pos_tensor.unsqueeze(0)  # Should be [1, 7]
                )
                goal_pose = Pose(
                    position=target_pos_base,  # Already [1, 3]
                    quaternion=target_quat_base  # Already [1, 4]
                )

                plan_cfg = MotionGenPlanConfig(max_attempts=1, enable_graph=False)

                try:
                    result = self.motion_gen.plan_single(start_state, goal_pose, plan_cfg)

                    if result is None:
                        print(f"[WARN] Motion planning returned None for env {b}, waypoint {i}/{N}")
                        all_success = False
                        break

                    traj = result.get_interpolated_plan()
                    if traj is None:
                        print(f"[WARN] Failed to get interpolated plan for env {b}, waypoint {i}/{N}")
                        all_success = False
                        break

                    # Extract joint positions
                    traj_np = traj.position.cpu().numpy()  # Expected: [1, T, 7]

                    # # Debug: Check trajectory shape
                    # if i == 0 or i == 1:  # Debug first two waypoints
                        # print(f"[DEBUG] Env {b}, waypoint {i}: traj.position shape = {traj.position.shape}, numpy shape = {traj_np.shape}")

                    # Handle different trajectory shapes
                    if traj_np.ndim == 3:
                        traj_b = traj_np[0]  # [T, 7]
                    elif traj_np.ndim == 2:
                        traj_b = traj_np  # Already [T, 7]
                    else:
                        print(f"[ERROR] Unexpected trajectory shape {traj_np.shape} for env {b}, waypoint {i}/{N}")
                        all_success = False
                        break

                    if traj_b.shape[0] == 0:
                        print(f"[ERROR] Empty trajectory returned for env {b}, waypoint {i}/{N}")
                        all_success = False
                        break

                    if traj_b.ndim < 2 or traj_b.shape[-1] != 7:
                        print(f"[ERROR] Invalid trajectory shape {traj_b.shape} for env {b}, waypoint {i}/{N}, expected [T, 7]")
                        all_success = False
                        break

                    joint_traj_segments.append(traj_b)
                    segment_lengths.append(traj_b.shape[0])  # Record segment length

                    # Update current joint position
                    current_joint_pos = traj_b[-1]  # Should be [7]

                    if current_joint_pos.shape[0] != 7:
                        print(f"[ERROR] Invalid current_joint_pos shape {current_joint_pos.shape} for env {b}, waypoint {i}/{N}")
                        all_success = False
                        break

                except Exception as e:
                    print(f"[ERROR] Motion planning exception for env {b}, waypoint {i}/{N}: {e}")
                    import traceback
                    traceback.print_exc()
                    all_success = False
                    break

            if all_success and len(joint_traj_segments) > 0:
                # Concatenate all segments
                dense_traj = np.concatenate(joint_traj_segments, axis=0)
                dense_joint_trajs.append(dense_traj)
                success_list.append(True)
                
                # Compute waypoint indices in dense trajectory
                # waypoint 0: pre-grasp, waypoint 1: grasp, waypoint 2: lift, waypoint 3+: trajectory
                # Pre-grasp ends after first segment (waypoint 0 -> waypoint 1)
                pregrasp_end = segment_lengths[0] if len(segment_lengths) > 0 else 0
                
                # Grasp ends after second segment (waypoint 1 -> waypoint 2)
                if len(segment_lengths) > 1:
                    grasp_end = segment_lengths[0] + segment_lengths[1]
                else:
                    grasp_end = pregrasp_end
                
                gripper_close_indices.append(grasp_end)

                # Create action indices - each joint pose gets its step index in the trajectory
                action_indices = np.arange(dense_traj.shape[0]).reshape(-1, 1)
                action_indices_list.append(action_indices)
            else:
                dense_joint_trajs.append(None)
                success_list.append(False)
                gripper_close_indices.append(0)
                action_indices_list.append(None)

        action_indices_list = np.array(action_indices_list, dtype=np.int32)
        self.action_indices_list = action_indices_list.transpose(1, 0, 2)
        return dense_joint_trajs, np.array(success_list), gripper_close_indices, action_indices_list

    def compute_object_poses_from_eef(self, dense_joint_trajs: List[np.ndarray], gripper_close_indices: List[tuple]):
        """
        Compute object poses based on eef-object relative transform.
        Assumes the relative pose is fixed after grasping.

        Args:
            dense_joint_trajs: List of [M, 7] joint trajectories

        Returns:
            object_pose_trajs: List of [M, 7] object pose trajectories (pos + quat)
        """

        B = self.scene.num_envs
        
        # Reset with physics enabled to get initial states
        self.reset(enable_physics=True)
        
        # Get object poses and robot joint positions
        object_real_poses = self.object_prim.data.root_state_w[:, :7].cpu().numpy()
        initial_robot_joint_pos = self.robot.data.joint_pos.clone()  # Save initial joint positions

        
        
        # Get object pose in world frame
        object_pose_trajs = []
        for b in range(B):
            if dense_joint_trajs[b] is None:
                object_pose_trajs.append(None)
                continue

            joint_traj = dense_joint_trajs[b]  # [M, 7]
            M = joint_traj.shape[0]

            # Compute T_obj_in_ee at grasp moment (object relative to eef)
            grasp_pose = self.grasp_pose_list[b]  # [7] - eef pose in local frame
            env_origin = self.scene.env_origins[b].cpu().numpy()

            # Convert grasp pose to world frame for FK computation
            grasp_pose_world = grasp_pose.copy()
            grasp_pose_world[:3] += env_origin

            T_grasp_ee = pose_to_mat(grasp_pose_world[:3], grasp_pose_world[3:7])
            T_grasp_obj = pose_to_mat(object_real_poses[b, :3], object_real_poses[b, 3:7])

            # T_obj_in_ee: object pose in end-effector frame
            T_obj_in_ee = np.linalg.inv(T_grasp_ee) @ T_grasp_obj

            # For each joint state, compute eef pose via FK, then object pose
            obj_poses = []

            grasp_index = gripper_close_indices[b]
            # Use the object pose we got before disabling physics
            initial_object_pose = object_real_poses[b].copy()
            # Batch FK: set all joint states at once for this env
            for m in range(M):
                joint_pos = joint_traj[m]

                # Set robot joint state for this env
                # Use the saved initial joint positions instead of accessing robot.data.joint_pos
                joint_pos_full = initial_robot_joint_pos.clone()
                joint_pos_full[b, self.robot_entity_cfg.joint_ids] = torch.tensor(
                    joint_pos, dtype=torch.float32, device=self.sim.device
                )

                self.robot.write_joint_state_to_sim(
                    joint_pos_full,
                    torch.zeros_like(joint_pos_full),
                    env_ids=self._all_env_ids
                )
                self.robot.write_data_to_sim()

                # Update internal state (this triggers FK computation)
                # We need to step physics to update kinematics
                self.step()
                self.robot.update(self.sim_dt)
                # obs = self.get_observation(gripper_open = False)
                # self.record_data(obs)
        
                # Get eef pose in world frame
                ee_state_w = self.robot.data.body_state_w[b, self.robot_entity_cfg.body_ids[0], :7]
                ee_pos_w = ee_state_w[:3].cpu().numpy()
                ee_quat_w = ee_state_w[3:7].cpu().numpy()

                # Compute object pose: T_obj_w = T_ee_w @ T_obj_in_ee
                T_ee_w = pose_to_mat(ee_pos_w, ee_quat_w)
                T_obj_w = T_ee_w @ T_obj_in_ee
                obj_pos_w, obj_quat_w = mat_to_pose(T_obj_w)

                # Convert to local frame
                obj_pos_local = obj_pos_w - env_origin
                obj_pose = np.concatenate([obj_pos_local, obj_quat_w])
                if m <= grasp_index:
                    obj_pose = initial_object_pose.copy()
                    obj_pose[:3] -= env_origin
                
                obj_poses.append(obj_pose)

            object_pose_trajs.append(np.array(obj_poses, dtype=np.float32))
        # self.save_data()
        return object_pose_trajs

    def render_trajectory(self, dense_joint_trajs: List[np.ndarray],
                         object_pose_trajs: List[np.ndarray],
                         gripper_close_indices: List[tuple]):
        """
        Directly set robot and object states and render.

        Physics and collision control:
        - DISABLED during trajectory execution (prevents collision interference)
        - ENABLED at the last step if gripper is commanded to open (realistic release)

        Args:
            dense_joint_trajs: List of [M, 7] joint trajectories
            object_pose_trajs: List of [M, 7] object pose trajectories
            waypoint_indices: List of tuples (pregrasp_end, grasp_end) for each env
                pregrasp_end: index in dense trajectory where pre-grasp phase ends
                grasp_end: index in dense trajectory where grasp phase ends
        """
        B = self.scene.num_envs

        # Determine max trajectory length
        max_len = max([traj.shape[0] if traj is not None else 0 for traj in dense_joint_trajs])

        if max_len == 0:
            print("[ERROR] No valid trajectories to render")
            return

        # Disable physics and collision before trajectory execution
        # This prevents collision detection from interfering with our direct state setting
        self.reset(enable_physics=False)
        self.save_dict["actions"] = self.actions_arr
        self.save_dict["action_indices"] = self.action_indices_list
        # Initialize gripper state tracking for smooth transitions
        # Track: current state (True=open, False=closed), target state, transition step
        gripper_state_tracking = {}
        for b in range(B):
            # Initialize: start with open gripper
            gripper_state_tracking[b] = {
                'current_open': True,
                'target_open': True,
                'transition_step': 0,  # 0 means no transition in progress
                'transition_total': 5  # 5 steps for transition
            }

        # Render each timestep
        for t in range(max_len):
            is_last_step = (t == max_len - 1)

            # Set robot joint states
            joint_states = []
            gripper_target_states = []

            for b in range(B):
                if dense_joint_trajs[b] is not None and t < dense_joint_trajs[b].shape[0]:
                    joint_states.append(dense_joint_trajs[b][t])

                    # Determine target gripper state based on trajectory progress
                    grasp_end = gripper_close_indices[b]
                    traj_len = dense_joint_trajs[b].shape[0] if dense_joint_trajs[b] is not None else 0

                    if t < grasp_end:
                        target_gripper_open = True  # Open during pre-grasp and approach
                    elif t < traj_len - 1:
                        target_gripper_open = False  # Closed during grasping phase
                    else:
                        target_gripper_open = not self.final_gripper_state_list[b]  # Final state at trajectory end

                    gripper_target_states.append(target_gripper_open)
                else:
                    # Use last valid joint state
                    if dense_joint_trajs[b] is not None:
                        joint_states.append(dense_joint_trajs[b][-1])
                        gripper_target_states.append(not self.final_gripper_state_list[b])  # Final state
                    else:
                        joint_states.append(self.robot.data.joint_pos[b, self.robot_entity_cfg.joint_ids].cpu().numpy())
                        gripper_target_states.append(True)

            # Set arm joint positions (only the 7 arm joints, not gripper)
            joint_states_t = torch.tensor(np.array(joint_states), dtype=torch.float32, device=self.sim.device)
            # Use write_joint_state_to_sim for direct state setting (bypassing physics)
            # We need to specify joint_ids to only set the arm joints
            self.robot.write_joint_position_to_sim(
                joint_states_t,
                joint_ids=self.robot_entity_cfg.joint_ids,
                env_ids=self._all_env_ids
            )
            self.robot.write_joint_velocity_to_sim(
                torch.zeros_like(joint_states_t),
                joint_ids=self.robot_entity_cfg.joint_ids,
                env_ids=self._all_env_ids
            )

            # Set gripper positions with smooth 5-step transitions
            # Create gripper position tensor for all environments
            gripper_positions = torch.zeros((B, len(self.robot_gripper_cfg.joint_ids)),
                                           dtype=torch.float32, device=self.sim.device)
            current_gripper_states = []
            
            for b in range(B):
                tracking = gripper_state_tracking[b]
                target_open = gripper_target_states[b]

                # Check if we need to start a new transition
                if target_open != tracking['current_open'] and tracking['transition_step'] == 0:
                    # Start new transition
                    tracking['target_open'] = target_open
                    tracking['transition_step'] = 1
                elif tracking['target_open'] != target_open:
                    # Target changed during transition, update target
                    tracking['target_open'] = target_open
                
                # Update transition progress
                if tracking['transition_step'] > 0:
                    # In transition: interpolate between current and target
                    alpha = tracking['transition_step'] / tracking['transition_total']  # 0 to 1
                    
                    # Interpolate between open (0.04) and closed (0.0) positions
                    open_pos = self.gripper_open_tensor[b]
                    close_pos = self.gripper_close_tensor[b]
                    
                    if tracking['target_open']:
                        # Transitioning to open
                        gripper_positions[b] = close_pos + alpha * (open_pos - close_pos)
                    else:
                        # Transitioning to closed
                        gripper_positions[b] = open_pos - alpha * (open_pos - close_pos)
                    
                    # Update transition step
                    tracking['transition_step'] += 1
                    if tracking['transition_step'] > tracking['transition_total']:
                        # Transition complete
                        tracking['current_open'] = tracking['target_open']
                        tracking['transition_step'] = 0
                else:
                    # No transition: use current state directly
                    if tracking['current_open']:
                        gripper_positions[b] = self.gripper_open_tensor[b]
                    else:
                        gripper_positions[b] = self.gripper_close_tensor[b]

                current_gripper_states.append(tracking['current_open'])

            self.robot.write_joint_position_to_sim(
                gripper_positions,
                joint_ids=self.robot_gripper_cfg.joint_ids,
                env_ids=self._all_env_ids
            )
            self.robot.write_joint_velocity_to_sim(
                torch.zeros_like(gripper_positions),
                joint_ids=self.robot_gripper_cfg.joint_ids,
                env_ids=self._all_env_ids
            )

            # Compute average gripper state for recording
            avg_gripper_open = sum(current_gripper_states) / len(current_gripper_states) > 0.5
            
            # Check if all grippers are fully open (for physics activation)
            all_grippers_fully_open = all(
                tracking['current_open'] and tracking['transition_step'] == 0 
                for tracking in gripper_state_tracking.values()
            )

            # Set object poses
            object_poses = []
            for b in range(B):
                if object_pose_trajs[b] is not None and t < object_pose_trajs[b].shape[0]:
                    obj_pose_local = object_pose_trajs[b][t]
                else:
                    # Use last valid object pose
                    if object_pose_trajs[b] is not None:
                        obj_pose_local = object_pose_trajs[b][-1]
                    else:
                        obj_pose_local = np.concatenate([
                            self.object_prim.data.root_state_w[b, :3].cpu().numpy() - self.scene.env_origins[b].cpu().numpy(),
                            self.object_prim.data.root_state_w[b, 3:7].cpu().numpy()
                        ])

                # Convert to world frame
                env_origin = self.scene.env_origins[b].cpu().numpy()
                obj_pose_world = obj_pose_local.copy()
                obj_pose_world[:3] += env_origin
                object_poses.append(obj_pose_world)

            object_poses_t = torch.tensor(np.array(object_poses), dtype=torch.float32, device=self.sim.device)
            self.object_prim.write_root_pose_to_sim(object_poses_t, env_ids=self._all_env_ids)
            self.object_prim.write_root_velocity_to_sim(
                torch.zeros((B, 6), device=self.sim.device, dtype=torch.float32),
                env_ids=self._all_env_ids
            )

            # Write data to sim
            self.robot.write_data_to_sim()
            self.object_prim.write_data_to_sim()

            # Normal step: direct state setting, no physics
            # Physics is already disabled, so sim.step will only update kinematics and render
            self.step()

            # Record observation if needed
            if self.record and self.count % self.save_interval == 0:
                obs = self.get_observation(gripper_open=avg_gripper_open)
                self.record_data(obs)

            self.count += 1
            self.task_step_count = self.count // self.decimation
   
        # After trajectory ends, check if we need to open grippers and enable physics
        need_physics_after_traj = any(not self.final_gripper_state_list[b] for b in range(B))

        if need_physics_after_traj:
            print(f"[INFO] Trajectory ended with open gripper command, enabling physics for object release...")

            # Enable physics ONCE at the start
            if self.require_gravity_final:
                self.set_gravity_on()

            # Start opening grippers (if needed)
            for b in range(B):
                if not self.final_gripper_state_list[b]:  # Need to open
                    tracking = gripper_state_tracking[b]
                    if not tracking['current_open'] and tracking['transition_step'] == 0:
                        # Start opening transition
                        tracking['target_open'] = True
                        tracking['transition_step'] = 1

            # Execute gripper opening transition + physics settling
            max_steps = 10  # Safety limit (5-step transition + settling)
            for step_idx in range(max_steps):
                # Check if all grippers are fully open
                all_fully_open = all(
                    (gripper_state_tracking[b]['current_open'] and
                     gripper_state_tracking[b]['transition_step'] == 0) or
                    self.final_gripper_state_list[b]  # Or doesn't need to open
                    for b in range(B)
                )

                # At least 5 steps for transition, then check if we can exit
                if all_fully_open and step_idx >= 5:
                    break

                # Keep robot joints at final position (no movement during gripper transition)
                joint_states = []
                for b in range(B):
                    if dense_joint_trajs[b] is not None:
                        joint_states.append(dense_joint_trajs[b][-1])
                    else:
                        joint_states.append(self.robot.data.joint_pos[b, self.robot_entity_cfg.joint_ids].cpu().numpy())

                joint_states_t = torch.tensor(np.array(joint_states), dtype=torch.float32, device=self.sim.device)
                self.robot.write_joint_position_to_sim(
                    joint_states_t,
                    joint_ids=self.robot_entity_cfg.joint_ids,
                    env_ids=self._all_env_ids
                )
                self.robot.write_joint_velocity_to_sim(
                    torch.zeros_like(joint_states_t),
                    joint_ids=self.robot_entity_cfg.joint_ids,
                    env_ids=self._all_env_ids
                )

                # DO NOT set object position here - let physics handle it!
                # The object should naturally fall/move when gripper opens

                # Update gripper positions
                gripper_positions = torch.zeros((B, len(self.robot_gripper_cfg.joint_ids)),
                                               dtype=torch.float32, device=self.sim.device)
                current_gripper_states = []

                for b in range(B):
                    tracking = gripper_state_tracking[b]

                    if tracking['transition_step'] > 0:
                        # In transition: interpolate between current and target
                        alpha = tracking['transition_step'] / tracking['transition_total']  # 0 to 1

                        # Interpolate between open (0.04) and closed (0.0) positions
                        open_pos = self.gripper_open_tensor[b]
                        close_pos = self.gripper_close_tensor[b]

                        if tracking['target_open']:
                            # Transitioning to open
                            gripper_positions[b] = close_pos + alpha * (open_pos - close_pos)
                        else:
                            # Transitioning to closed
                            gripper_positions[b] = open_pos - alpha * (open_pos - close_pos)

                        # Update transition step
                        tracking['transition_step'] += 1
                        if tracking['transition_step'] > tracking['transition_total']:
                            # Transition complete
                            tracking['current_open'] = tracking['target_open']
                            tracking['transition_step'] = 0
                    else:
                        # No transition: use current state directly
                        if tracking['current_open']:
                            gripper_positions[b] = self.gripper_open_tensor[b]
                        else:
                            gripper_positions[b] = self.gripper_close_tensor[b]

                    current_gripper_states.append(tracking['current_open'])

                self.robot.write_joint_position_to_sim(
                    gripper_positions,
                    joint_ids=self.robot_gripper_cfg.joint_ids,
                    env_ids=self._all_env_ids
                )
                self.robot.write_joint_velocity_to_sim(
                    torch.zeros_like(gripper_positions),
                    joint_ids=self.robot_gripper_cfg.joint_ids,
                    env_ids=self._all_env_ids
                )

                # Write data to sim
                self.robot.write_data_to_sim()
                self.object_prim.write_data_to_sim()

                # Step physics
                self.step()

                # Record observation
                if self.record and self.count % self.save_interval == 0:
                    obs = self.get_observation(gripper_open=True)
                    self.record_data(obs)

                # Update counters
                self.count += 1
                self.task_step_count = self.count // self.decimation
            
            if self.require_gravity_final:
                self.set_gravity_off()

            print(f"[INFO] Physics simulation complete after {step_idx + 1} steps")


    def run_batch_trajectory(self, traj_cfg_list: List[TrajectoryCfg]):
        """
        Main execution function: process a batch of trajectory configs.

        Workflow:
        1. Reset with physics enabled (to settle objects)
        2. Compute sparse EEF poses
        3. Plan dense joint trajectories with Curobo
        4. Compute object poses from EEF (using FK)
        5. Render trajectory (physics disabled except last step with open gripper)

        Args:
            traj_cfg_list: List of trajectory configurations

        Returns:
            success_env_ids: List of successful environment indices
        """
        self.traj_cfg_list = traj_cfg_list
        self.compute_components()

        # Reset environments with physics enabled (Step 1)
        print("[INFO] Resetting environments with physics enabled...")
        self.reset(enable_physics=True)

        # Refine grasp and pregrasp poses based on current vs reference root pose
        print("[INFO] Refining grasp and pregrasp poses...")
        B = self.scene.num_envs
        
        # Get reference root pose from trajectory (initial object pose from traj)
        init_root_pose_list = []
        for b in range(B):
            # Get initial object pose from trajectory (first frame)
            T_init_obj = self.object_trajectory_list[b][0]  # [4, 4]
            init_pos = T_init_obj[:3, 3]  # (3,)
            init_rot = T_init_obj[:3, :3]  # (3, 3)
            init_quat = transforms3d.quaternions.mat2quat(init_rot)  # (4,)
            init_root_pose = np.concatenate([init_pos, init_quat])  # (7,)
            init_root_pose_list.append(init_root_pose)
        init_root_pose = np.array(init_root_pose_list)  # (B, 7)
        
        # Extract grasp and pregrasp positions (convert to world frame)
        grasp_pos_local = np.array([self.grasp_pose_list[b][:3] for b in range(B)])  # (B, 3)
        pregrasp_pos_local = np.array([self.pregrasp_pose_list[b][:3] for b in range(B)])  # (B, 3)
        env_origins = self.scene.env_origins.cpu().numpy()  # (B, 3)
        grasp_pos_world = grasp_pos_local + env_origins  # (B, 3)
        pregrasp_pos_world = pregrasp_pos_local + env_origins  # (B, 3)
        
        # Refine poses
        grasp_pos_world_refined, pregrasp_pos_world_refined = self.refine_grasp_pose(
            init_root_pose, grasp_pos_world, pregrasp_pos_world
        )
        
        # # Convert back to local frame and update lists
        grasp_pos_local_refined = grasp_pos_world_refined - env_origins  # (B, 3)
        pregrasp_pos_local_refined = pregrasp_pos_world_refined - env_origins  # (B, 3)
        
        # Update pregrasp_pose_list and grasp_pose_list (only position, keep quaternion)
        for b in range(B):
            self.pregrasp_pose_list[b][:3] = pregrasp_pos_local_refined[b]
            self.grasp_pose_list[b][:3] = grasp_pos_local_refined[b]

        print("[INFO] Computing sparse eef poses...")
        sparse_eef_poses = self.compute_sparse_eef_poses()

        print("[INFO] Planning dense joint trajectories with Curobo...")
        dense_joint_trajs, planning_success, gripper_close_indices, action_indices_list = self.plan_dense_joint_trajectory(sparse_eef_poses)

        if not np.any(planning_success):
            print("[ERROR] All motion planning failed")
            return []

        print(f"[INFO] Motion planning succeeded for {planning_success.sum()}/{len(planning_success)} envs")

        print("[INFO] Computing object poses from eef trajectories...")
        object_pose_trajs = self.compute_object_poses_from_eef(dense_joint_trajs, gripper_close_indices)

        print("[INFO] Rendering trajectories...")
        self.render_trajectory(dense_joint_trajs, object_pose_trajs, gripper_close_indices)

   

        # Determine success (for now, use planning success)
        success_env_ids = np.where(planning_success)[0].tolist()

        print(f"[INFO] Success: {len(success_env_ids)}/{self.scene.num_envs} environments")

        if self.record and len(success_env_ids) > 0:
            self.save_data(ignore_keys=["segmask", "depth"], env_ids=success_env_ids, export_hdf5=True)

        return success_env_ids



# ──────────────────────────── Entry Point ────────────────────────────
def sim_direct_render(key: str, args_cli: argparse.Namespace):
    """
    Main function: Simple grasp selection, then randomize and render.

    Workflow:
    1. Simple grasp selection (rescore + select top-1, no rollout)
    2. Create initial task config with selected grasp
    3. Randomize trajectories
    4. Direct render with Curobo planning
    """
    # Step 1: Simple grasp selection (no physical rollout)
    print("="*80)
    print("STEP 1: Simple Grasp Selection (Rescore + Select Top-1)")
    print("="*80)

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

    assert task_json_path.exists(), f"Task config not found: {task_json_path}"
    print(f"[INFO] Task config already exists, loading from: {task_json_path}")
    task_cfg = load_task_cfg(task_json_path)

    print("\n" + "="*80)
    print("STEP 1: Loading task config and generating randomized trajectories")
    print("="*80)

    task_json_path = BASE_DIR / "tasks" / key / "task.json"
    task_cfg = load_task_cfg(task_json_path)

    # Get configs
    rollout_cfg = get_direct_render_config(key)
    randomizer_cfg = get_randomizer_config(key)
    simulation_cfg = get_simulation_config(key)

    # Override with CLI args if provided
    if args_cli.num_envs is not None:
        num_envs = args_cli.num_envs
    else:
        num_envs = rollout_cfg.num_envs

    if args_cli.total_num is not None:
        total_require_traj_num = args_cli.total_num
    else:
        total_require_traj_num = rollout_cfg.total_num

    goal_offset = rollout_cfg.goal_offset
    gripper_close_width = rollout_cfg.gripper_close_width
    save_interval = simulation_cfg.save_interval
    physics_freq = simulation_cfg.physics_freq
    decimation = simulation_cfg.decimation

    print(f"[INFO] Config: num_envs={num_envs}, total_num={total_require_traj_num}")
    print(f"[INFO] Randomizer: {randomizer_cfg.to_kwargs()}")

    # Generate randomized trajectories
    randomizer = Randomizer(task_cfg)
    randomizer_kwargs = randomizer_cfg.to_kwargs()
    random_task_cfg_list = randomizer.generate_randomized_scene_cfg(**randomizer_kwargs)

    print(f"[INFO] Generated {len(random_task_cfg_list)} randomized trajectory configs")

    # Step 3: Direct render with randomized trajectories
    print("\n" + "="*80)
    print("STEP 2: Direct rendering with Curobo motion planning")
    print("="*80)

    args_cli.key = key
    sim_cfgs = load_sim_parameters(BASE_DIR, key)
    data_dir = BASE_DIR / "h5py" / key

    success_trajectory_config_list = []
    current_timestep = 0

    # Create environment
    env, _ = make_env(
        cfgs=sim_cfgs, num_envs=num_envs,
        device=args_cli.device,
        bg_simplify=False,
        physics_freq=physics_freq,
        decimation=decimation,
        has_collision=False,
    )
    sim, scene = env.sim, env.scene

    # Create direct render simulator
    my_sim = DirectRenderSimulator(
        sim, scene, sim_cfgs=sim_cfgs, data_dir=data_dir, record=True,
        args_cli=args_cli, goal_offset=goal_offset, save_interval=save_interval,
        decimation=decimation, gripper_close_width=gripper_close_width,
        require_gravity_final=rollout_cfg.require_gravity_final
    )
    my_sim.task_cfg = task_cfg

    # Process batches
    while len(success_trajectory_config_list) < total_require_traj_num:
        traj_cfg_list = random_task_cfg_list[current_timestep: current_timestep + num_envs]

        if len(traj_cfg_list) == 0:
            print("[WARN] No more randomized trajectories available")
            break

        current_timestep += num_envs

        print(f"\n[INFO] Processing batch {current_timestep // num_envs}, "
              f"trajectories {current_timestep - num_envs} to {current_timestep}")

        success_env_ids = my_sim.run_batch_trajectory(traj_cfg_list)

        if len(success_env_ids) > 0:
            for env_id in success_env_ids:
                success_trajectory_config_list.append(traj_cfg_list[env_id])
                add_generated_trajectories(task_cfg, [traj_cfg_list[env_id]], task_json_path.parent)

        print(f"[INFO] Total success: {len(success_trajectory_config_list)}/{total_require_traj_num}")

    env.close()

    print("\n" + "="*80)
    print(f"COMPLETE: Generated {len(success_trajectory_config_list)} successful trajectories")
    print("="*80)

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
