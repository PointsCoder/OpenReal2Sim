"""
Heuristic manipulation policy in Isaac Lab simulation.
Using grasping and motion planning to perform object manipulation tasks.
"""
from __future__ import annotations

# ───────────────────────────────────────────────────────────────────────────── AppLauncher ─────────────────────────────────────────────────────────────────────────────
import argparse, os, json, random, transforms3d
from pathlib import Path
import numpy as np
import torch
import yaml
import sys
from isaaclab.app import AppLauncher

# Add current directory first (for local sim_base), then parent directory
sys.path.insert(0, str(Path(__file__).parent))  # data_collection directory (for sim_base)
sys.path.insert(1, str(Path(__file__).parent.parent))  # isaaclab directory (for other modules)

# ───────────────────────────────────────────────────────────────────────────── CLI ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser("sim_policy")
parser.add_argument("--key", type=str, default="demo_video", help="scene key (outputs/<key>)")
parser.add_argument("--robot", type=str, default="franka")
parser.add_argument("--num_envs", type=int, default=5)
parser.add_argument("--num_trials", type=int, default=1)
parser.add_argument("--teleop_device", type=str, default="keyboard")
parser.add_argument("--sensitivity", type=float, default=1.0)
parser.add_argument("--debug", action="store_true", default=True, help="Enable debug logging")
parser.add_argument("--target_successes", type=int, default=40, help="Minimum number of successful demonstrations to collect (will rerun until target is met)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.headless = True  # headless mode for batch execution
app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

# ───────────────────────────────────────────────────────────────────────────── Runtime imports ─────────────────────────────────────────────────────────────────────────────
import isaaclab.sim as sim_utils
from isaaclab.utils.math import subtract_frame_transforms

from graspnetAPI.grasp import GraspGroup

from typing import List
import imageio
from concurrent.futures import ThreadPoolExecutor
import time
import psutil

# ───────────────────────────────────────────────────────────────────────────── Simulation environments ─────────────────────────────────────────────────────────────────────────────
# Import AFTER AppLauncher has initialized Isaac Sim runtime
from sim_base import BaseSimulator, get_next_demo_id  # From local data_collection directory
from sim_env_factory import make_env  # Rest from parent isaaclab folder
from sim_preprocess.grasp_utils import get_best_grasp_with_hints
from sim_utils.transform_utils import pose_to_mat, mat_to_pose, grasp_to_world, grasp_approach_axis_batch
from sim_utils.sim_utils import load_sim_parameters

BASE_DIR   = Path.cwd()
img_folder = args_cli.key
out_dir    = BASE_DIR / "outputs"


# ───────────────────────────────────────────────────────────────────────────── Heuristic Manipulation ───────────────────────────────────────────────────────────────────────────── 
class HeuristicManipulation(BaseSimulator):
    """
    Physical trial-and-error grasping with approach-axis perturbation:
      â€¢ Multiple grasp proposals executed in parallel;
      â€¢ Every attempt does reset â†’ pre â†’ grasp â†’ close â†’ lift â†’ check;
      â€¢ Early stops when any env succeeds; then re-exec for logging.
    """
    def __init__(self, sim, scene, sim_cfgs: dict):
        robot_pose = torch.tensor(
            sim_cfgs["robot_cfg"]["robot_pose"],
            dtype=torch.float32,
            device=sim.device
        )
        super().__init__(
            sim=sim, sim_cfgs=sim_cfgs, scene=scene, args=args_cli,
            robot_pose=robot_pose, cam_dict=sim_cfgs["cam_cfg"],
            out_dir=out_dir, img_folder=img_folder,
            enable_motion_planning=True,
            set_physics_props=True, debug_level=0,
        )

        self.selected_object_id = sim_cfgs["demo_cfg"]["manip_object_id"]
        self.traj_key = sim_cfgs["demo_cfg"]["traj_key"]
        self.traj_path = sim_cfgs["demo_cfg"]["traj_path"]
        self.goal_offset = [0, 0, sim_cfgs["demo_cfg"]["goal_offset"]]
        self.grasp_path = sim_cfgs["demo_cfg"]["grasp_path"]
        self.grasp_idx = sim_cfgs["demo_cfg"]["grasp_idx"]
        self.grasp_pre = sim_cfgs["demo_cfg"]["grasp_pre"]
        self.grasp_delta = sim_cfgs["demo_cfg"]["grasp_delta"]
        self.verbose = args_cli.debug
        
        # Store random initial poses per environment (consistent throughout a single pipeline run)
        # Will be regenerated at the start of each new inference() call
        self.random_obj_poses = None  # Will store (B, 7) tensor: [x, y, z, qw, qx, qy, qz]
        self._generate_random_initial_poses()
        
        self.load_obj_goal_traj()
    
    def vprint(self, *args, **kwargs):
        """Print only if verbose flag is enabled"""
        if self.verbose:
            print(*args, **kwargs)
    
    def _generate_random_initial_poses(self, x_range=(-0.03, 0.03), y_range=(-0.03, 0.03), yaw_range=(-0.1, 0.1)):
        """
        Generate random initial poses for each environment.
        Random offsets are applied to x, y, and yaw (rotation around z-axis).
        These poses remain consistent during env resets within a single pipeline run.
        
        Args:
            x_range: (min, max) random offset range for x position in meters
            y_range: (min, max) random offset range for y position in meters  
            yaw_range: (min, max) random offset range for yaw rotation in radians
        """
        B = self.scene.num_envs
        device = self.sim.device
        
        # Generate random offsets
        rng = np.random.default_rng()
        x_offsets = rng.uniform(x_range[0], x_range[1], size=B).astype(np.float32)
        y_offsets = rng.uniform(y_range[0], y_range[1], size=B).astype(np.float32)
        yaw_offsets = rng.uniform(yaw_range[0], yaw_range[1], size=B).astype(np.float32)
        
        # Convert yaw to quaternion (rotation around z-axis)
        # For z-axis rotation: q = [cos(yaw/2), 0, 0, sin(yaw/2)] in wxyz format
        qw = np.cos(yaw_offsets / 2.0)
        qx = np.zeros(B, dtype=np.float32)
        qy = np.zeros(B, dtype=np.float32)
        qz = np.sin(yaw_offsets / 2.0)
        
        # Store as (B, 7) tensor: [x, y, z, qw, qx, qy, qz]
        # z offset is 0 (we don't randomize height)
        self.random_obj_poses = torch.zeros((B, 7), dtype=torch.float32, device=device)
        self.random_obj_poses[:, 0] = torch.from_numpy(x_offsets).to(device)
        self.random_obj_poses[:, 1] = torch.from_numpy(y_offsets).to(device)
        self.random_obj_poses[:, 2] = 0.0  # No z offset
        self.random_obj_poses[:, 3] = torch.from_numpy(qw).to(device)
        self.random_obj_poses[:, 4] = torch.from_numpy(qx).to(device)
        self.random_obj_poses[:, 5] = torch.from_numpy(qy).to(device)
        self.random_obj_poses[:, 6] = torch.from_numpy(qz).to(device)
        
        if self.verbose:
            print(f"[INFO] Generated random initial poses for {B} environments:")
            for b in range(min(B, 5)):  # Show first 5 envs
                print(f"  Env {b}: x_offset={x_offsets[b]:.4f}m, y_offset={y_offsets[b]:.4f}m, yaw_offset={yaw_offsets[b]:.4f}rad")
    
    def transform_grasp_to_randomized_pose(self, grasp_pos, grasp_quat, env_id):
        """
        Transform a grasp from the original object frame to the randomized object frame.
        Grasps are defined relative to the object at its original pose (0,0,0 with no rotation).
        We need to apply the random offset transform to move the grasp to the randomized pose.
        
        Args:
            grasp_pos: (3,) grasp position in original object frame
            grasp_quat: (4,) grasp quaternion (wxyz) in original object frame
            env_id: environment index
            
        Returns:
            transformed_pos: (3,) grasp position in randomized frame
            transformed_quat: (4,) grasp quaternion in randomized frame
        """
        # Get random offset for this env
        random_offset = self.random_obj_poses[env_id].cpu().numpy()  # (7,) [x, y, z, qw, qx, qy, qz]
        
        # Build transformation matrices
        T_original = pose_to_mat(grasp_pos, grasp_quat)  # Grasp in original frame
        T_offset = pose_to_mat(random_offset[:3], random_offset[3:7])  # Random offset transform
        
        # Apply transform: T_randomized = T_offset @ T_original
        T_randomized = T_offset @ T_original
        
        # Extract transformed pose
        transformed_pos, transformed_quat = mat_to_pose(T_randomized)
        
        return transformed_pos.astype(np.float32), transformed_quat.astype(np.float32)

    def load_obj_goal_traj(self):
        """
        Load the relative trajectory Δ_w (T,4,4) and precompute the absolute
        object goal trajectory for each env.
        
        IMPORTANT: We want randomized INITIAL poses but FIXED FINAL goal.
        Strategy:
        1. Compute trajectory using ORIGINAL pose (0,0,0) to get fixed final goal
        2. Initial pose in simulation will use random offsets (applied in reset())
        3. Planner will move from randomized initial pose to fixed final goal

        Sets:
        self.obj_rel_traj   : np.ndarray or None, shape (T,4,4)
        self.obj_goal_traj_w: np.ndarray or None, shape (B,T,4,4)
        """
        # ── 1) Load Δ_w (relative trajectory) ──
        rel = np.load(self.traj_path).astype(np.float32)
        self.obj_rel_traj = rel  # (T,4,4)

        # ── 2) Reset with random initial poses ──
        # This applies the random offsets stored in self.random_obj_poses to the object
        self.reset()

        B = self.scene.num_envs
        origins_gpu = self.scene.env_origins.detach()  # (B,3)
        origins = origins_gpu.cpu().numpy().astype(np.float32)  # (B,3)

        # ── 3) Compute trajectory goals using ORIGINAL (non-randomized) initial pose ──
        # This ensures all envs have the same final goal, regardless of random initial offsets
        # Original initial pose is at env_origin with identity quaternion
        original_init_pose = np.zeros(7, dtype=np.float32)
        original_init_pose[3] = 1.0  # wxyz = [1, 0, 0, 0]
        
        # Add goal offset (lift slightly to avoid collision at goal)
        offset_np = np.asarray(self.goal_offset, dtype=np.float32).reshape(3)
        original_init_pose[:3] += offset_np

        T = rel.shape[0]
        obj_goal = np.zeros((B, T, 4, 4), dtype=np.float32)
        
        # Compute trajectory for original pose (same for all envs)
        T_init_original = pose_to_mat(original_init_pose[:3], original_init_pose[3:7])  # (4,4)
        
        for b in range(B):
            for t in range(T):
                # Apply relative trajectory to ORIGINAL initial pose
                goal = rel[t] @ T_init_original
                # Convert to world frame for this env
                goal[:3, 3] += origins[b]
                obj_goal[b, t] = goal

        self.obj_goal_traj_w = obj_goal  # [B, T, 4, 4] - same final goal for all envs
        
        if self.verbose:
            # Show the common final goal position
            final_goal_pos = obj_goal[0, -1, :3, 3]  # Last timestep, translation
            print(f"[INFO] Fixed final goal position (env-local): [{final_goal_pos[0]:.3f}, {final_goal_pos[1]:.3f}, {final_goal_pos[2]:.3f}]")

    def follow_object_goals(self, start_joint_pos, sample_step=1, visualize=True, skip_envs=None):
        """
        Follow precomputed object absolute trajectory with automatic restart on failure.
        If motion planning fails, raises an exception to trigger re-grasp from step 0.
        
        Args:
            skip_envs: Boolean array [B] indicating which envs to skip from execution
        """
        B = self.scene.num_envs
        if skip_envs is None:
            skip_envs = np.zeros(B, dtype=bool)
        
        obj_goal_all = self.obj_goal_traj_w  # [B, T, 4, 4]
        T = obj_goal_all.shape[1]

        ee_w  = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]  # [B,7]
        obj_w = self.object_prim.data.root_state_w[:, :7]

        T_ee_in_obj = []
        for b in range(B):
            T_ee_w  = pose_to_mat(ee_w[b, :3],  ee_w[b, 3:7])
            T_obj_w = pose_to_mat(obj_w[b, :3], obj_w[b, 3:7])
            T_ee_in_obj.append((np.linalg.inv(T_obj_w) @ T_ee_w).astype(np.float32))
        
        # Store for is_success check
        self.T_ee_in_obj = T_ee_in_obj

        joint_pos = start_joint_pos
        root_w = self.robot.data.root_state_w[:, 0:7]

        t_iter = list(range(0, T, sample_step))
        t_iter = t_iter + [T-1] if t_iter[-1] != T-1 else t_iter

        # Calculate initial grasp distance (baseline) for continuous monitoring
        # We use the distance between EE and Object COM at the start of the trajectory
        ee_pos_initial = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:3]
        obj_pos_initial = self.object_prim.data.root_com_pos_w[:, 0:3]
        initial_grasp_dist = torch.norm(ee_pos_initial - obj_pos_initial, dim=1) # [B]
        
        # Store for is_success check
        self.initial_grasp_dist = initial_grasp_dist

        for t in t_iter:
            goal_pos_list, goal_quat_list = [], []
            if self.verbose:
                print(f"[INFO] follow object goal step {t}/{T}")
            # Preallocate arrays instead of building lists
            goal_pos_arr = np.zeros((B, 3), dtype=np.float32)
            goal_quat_arr = np.zeros((B, 4), dtype=np.float32)
            goal_quat_arr[:, 0] = 1.0  # Default identity quaternion
            
            for b in range(B):
                if skip_envs[b]:
                    # For skipped envs, use current EE pose to avoid planning to 0,0,0 (which crashes Curobo)
                    # We don't care about the result, but we need a valid input
                    T_ee_w = pose_to_mat(ee_w[b, :3].cpu().numpy(), ee_w[b, 3:7].cpu().numpy())
                    pos_b, quat_b = mat_to_pose(T_ee_w)
                    goal_pos_arr[b] = pos_b.astype(np.float32)
                    goal_quat_arr[b] = quat_b.astype(np.float32)
                else:
                    T_obj_goal = obj_goal_all[b, t]
                    T_ee_goal  = T_obj_goal @ T_ee_in_obj[b]
                    pos_b, quat_b = mat_to_pose(T_ee_goal)
                    goal_pos_arr[b] = pos_b.astype(np.float32)
                    goal_quat_arr[b] = quat_b.astype(np.float32)

            goal_pos  = torch.as_tensor(goal_pos_arr,  dtype=torch.float32, device=self.sim.device)
            goal_quat = torch.as_tensor(goal_quat_arr, dtype=torch.float32, device=self.sim.device)

            if visualize:
                self.show_goal(goal_pos, goal_quat)
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_w[:, :3], root_w[:, 3:7], goal_pos, goal_quat
            )
            
            # Execute motion planning for all environments
            # Note: move_to() executes all B environments; we record dummy data for skipped envs
            joint_pos, success = self.move_to(ee_pos_b, ee_quat_b, gripper_open=False)
            
            # Check for critical motion planning failure
            if joint_pos is None or success is None:
                print(f"[CRITICAL] Motion planning failed at step {t}/{T}")
                print("[CRITICAL] Object likely dropped - need to restart from grasp")
                raise RuntimeError("MotionPlanningFailure_RestartNeeded")
            
            if torch.any(success == False):
                print(f"[WARN] Some active envs failed motion planning at step {t}/{T}, but continuing...")
            
            # --- Continuous Grasp Maintenance Check ---
            # Verify object hasn't slipped/dropped by checking if distance to EE has increased significantly
            ee_pos_curr = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:3]
            obj_pos_curr = self.object_prim.data.root_com_pos_w[:, 0:3]
            current_grasp_dist = torch.norm(ee_pos_curr - obj_pos_curr, dim=1)
            
            # Check deviation: if current distance > initial distance + threshold (1cm)
            grasp_deviation = current_grasp_dist - initial_grasp_dist
            dropped_mask = grasp_deviation > 0.01  # 1cm threshold
            
            # Update skip_envs for dropped objects
            if torch.any(dropped_mask):
                new_drops = dropped_mask.cpu().numpy() & (~skip_envs)
                if np.any(new_drops):
                    drop_indices = np.where(new_drops)[0]
                    if self.verbose:
                        print(f"[WARN] Object dropped in envs {drop_indices.tolist()} at step {t}/{T} (dist change > 1cm)")
                    skip_envs[new_drops] = True
            # ------------------------------------------

            # Build action data: real data for active envs, dummy data for skipped envs
            # Batch CPU transfer to reduce synchronization
            ee_pos_b_cpu = ee_pos_b.cpu()
            ee_quat_b_cpu = ee_quat_b.cpu()
            ee_pos_b_data = ee_pos_b_cpu.numpy()
            ee_quat_b_data = ee_quat_b_cpu.numpy()
            for b in np.where(skip_envs)[0]:
                ee_pos_b_data[b] = np.zeros(3, dtype=np.float32)
                ee_quat_b_data[b] = np.array([1, 0, 0, 0], dtype=np.float32)
            
            self.save_dict["actions"].append(np.concatenate([ee_pos_b_data, ee_quat_b_data, np.ones((B, 1))], axis=1))

        # Open gripper at the end to release object and record final state
        joint_pos = self.wait(gripper_open=True, steps=10)

        return joint_pos, skip_envs


    def viz_object_goals(self, sample_step=1, hold_steps=20):
        self.reset()
        self.wait(gripper_open=True, steps=10)
        B = self.scene.num_envs
        env_ids = torch.arange(B, device=self.object_prim.device, dtype=torch.long)
        goals = self.obj_goal_traj_w
        t_iter = list(range(0, goals.shape[1], sample_step))
        t_iter = t_iter + [goals.shape[1]-1] if t_iter[-1] != goals.shape[1]-1 else t_iter
        for t in t_iter:
            if self.verbose:
                print(f"[INFO] viz object goal step {t}/{goals.shape[1]}")
            # Build numpy arrays directly instead of lists
            pos_arr = np.zeros((B, 3), dtype=np.float32)
            quat_arr = np.zeros((B, 4), dtype=np.float32)
            for b in range(B):
                pos, quat = mat_to_pose(goals[b, t])
                pos_arr[b] = pos.astype(np.float32)
                quat_arr[b] = quat.astype(np.float32)
            pose = self.object_prim.data.root_com_state_w[:, :7]  # ← CHANGED: Use COM
            pose[:, :3]   = torch.as_tensor(pos_arr,  dtype=torch.float32, device=pose.device)
            pose[:, 3:7]  = torch.as_tensor(quat_arr, dtype=torch.float32, device=pose.device)
            self.show_goal(pose[:, :3], pose[:, 3:7])

            for _ in range(hold_steps):
                self.object_prim.write_root_pose_to_sim(pose, env_ids=env_ids)
                self.object_prim.write_data_to_sim()
                self.step()

    # ---------- Helpers ----------
    def _to_base(self, pos_w: np.ndarray | torch.Tensor, quat_w: np.ndarray | torch.Tensor):
        """World â†’ robot base frame for all envs."""
        root = self.robot.data.root_state_w[:, 0:7]  # [B,7]
        p_w, q_w = self._ensure_batch_pose(pos_w, quat_w)
        pb, qb = subtract_frame_transforms(
            root[:, 0:3], root[:, 3:7], p_w, q_w
        )
        return pb, qb  # [B,3], [B,4]

    # ---------- Batched execution & lift-check ----------
    def execute_and_lift_once_batch(self, info: dict, lift_height=0.15, position_threshold=0.2, record=False) -> tuple[np.ndarray, np.ndarray]:
        """
        Reset → pre → grasp → close → lift → hold; return (success[B], score[B]).
        Now propagates motion planning failures by returning (None, None).
        
        Args:
            record: If False, skip recording observations (faster for grasp search)
        """
        B = self.scene.num_envs
        self.reset()
        
        # Capture initial orientation for stability check
        obj_quat_initial = self.object_prim.data.root_quat_w.clone()

        # open gripper buffer
        jp = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
        self.wait(gripper_open=True, steps=4, record=record)

        # pre-grasp
        jp, success = self.move_to(info["pre_p_b"], info["pre_q_b"], gripper_open=True, record=record)
        # Keep success tensor on GPU for now
        if torch.any(success==False):
            if self.verbose:
                failed_ids = torch.where(success==False)[0].cpu().numpy()
                print(f"[WARN] Pre-grasp motion planning failed for envs: {failed_ids.tolist()}")
        jp = self.wait(gripper_open=True, steps=3, record=record)

        # grasp
        jp, success_grasp = self.move_to(info["p_b"], info["q_b"], gripper_open=True, record=record)
        success = success & success_grasp  # Combine on GPU
        if torch.any(success_grasp==False):
            if self.verbose:
                newly_failed = torch.where(success_grasp==False)[0].cpu().numpy()
                print(f"[WARN] Grasp approach motion planning failed for envs: {newly_failed.tolist()}")
        jp = self.wait(gripper_open=True, steps=2, record=record)

        # close
        jp = self.wait(gripper_open=False, steps=6, record=record)

        # initial heights (USING COM)
        # initial heights (USING COM)
        obj0 = self.object_prim.data.root_com_pos_w[:, 0:3]  # ← CHANGED: Use COM
        ee_w = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        ee_p0 = ee_w[:, :3]
        # CRITICAL: Must clone z0 values because simulation state changes between here and z1 calculation
        # If we use views, the z0 values will be updated when robot/object moves, breaking the diff calculation
        robot_ee_z0 = ee_p0[:, 2].clone()
        obj_z0 = obj0[:, 2].clone()
        
        # Capture initial XY for vertical lift check
        obj_xy0 = obj0[:, :2].clone()
        
        if self.verbose:
            obj_z0_mean = obj_z0.mean()
            ee_z0_mean = robot_ee_z0.mean()
            print(f"[INFO] mean object z0={obj_z0_mean:.3f} m, mean EE z0={ee_z0_mean:.3f} m")

        # lift: keep orientation, add height
        ee_q = ee_w[:, 3:7]
        target_p = ee_p0.clone()
        target_p[:, 2] += lift_height

        root = self.robot.data.root_state_w[:, 0:7]
        p_lift_b, q_lift_b = subtract_frame_transforms(
            root[:, 0:3], root[:, 3:7],
            target_p, ee_q
        )
        jp, success_lift = self.move_to(p_lift_b, q_lift_b, gripper_open=False, record=record)
        success = success & success_lift  # Combine on GPU
        if torch.any(success_lift==False):
            if self.verbose:
                newly_failed = torch.where(success_lift==False)[0].cpu().numpy()
                print(f"[WARN] Lift motion planning failed for envs: {newly_failed.tolist()}")
        jp = self.wait(gripper_open=False, steps=8, record=record)

        # final heights and success checking (USING COM)
        obj1 = self.object_prim.data.root_com_pos_w[:, 0:3]  # ← CHANGED: Use COM
        obj_quat_final = self.object_prim.data.root_quat_w.clone() # Capture final orientation
        
        ee_w1 = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        robot_ee_z1 = ee_w1[:, 2]
        obj_z1 = obj1[:, 2]
        if self.verbose:
            obj_z1_mean = obj_z1.mean()
            ee_z1_mean = robot_ee_z1.mean()
            print(f"[INFO] mean object z1={obj_z1_mean:.3f} m, mean EE z1={ee_z1_mean:.3f} m")

        # Lift check
        ee_diff  = robot_ee_z1 - robot_ee_z0
        obj_diff = obj_z1 - obj_z0
        lift_check = (torch.abs(ee_diff - obj_diff) <= 0.01) & \
            (torch.abs(ee_diff) >= 0.5 * lift_height) & \
            (torch.abs(obj_diff) >= 0.5 * lift_height)

        # Orientation stability check (dot product of quaternions)
        # |q1 . q2| should be close to 1.0 for similar orientations
        quat_dot = torch.abs(torch.sum(obj_quat_initial * obj_quat_final, dim=1))
        # Threshold 0.95 corresponds to approx 18 degrees difference
        orientation_check = quat_dot >= 0.95
        
        # Vertical stability check (XY drift)
        # Object should not move significantly in XY plane during vertical lift
        obj_xy1 = obj1[:, :2]
        xy_drift = torch.norm(obj_xy1 - obj_xy0, dim=1)
        vertical_check = xy_drift <= 0.02  # 2cm tolerance for XY drift

        # Combined check - all on GPU
        # Removed proximity check as it is not relevant for vertical lift
        lifted = lift_check & orientation_check & vertical_check
        
        # Mark motion planning failures as not lifted - keep on GPU
        lifted = lifted & success

        # Score calculation - all on GPU
        score = torch.zeros_like(ee_diff)
        if torch.any(lifted):
            base_score = 1000.0
            # Use orientation stability as score component to favor stable grasps
            score[lifted] = base_score * quat_dot[lifted]
        
        if self.verbose:
            print(f"[INFO] Lift check passed: {lift_check.sum().item()}/{B}")
            print(f"[INFO] Orientation check passed: {orientation_check.sum().item()}/{B}")
            print(f"[INFO] Vertical check passed: {vertical_check.sum().item()}/{B}")
            print(f"[INFO] Final lifted flag (all checks): {lifted.sum().item()}/{B}")
            
            if B <= 10:  # Detailed output for small batches
                for b in range(B):
                    print(f"  Env {b}: lift={lift_check[b].item()}, orient={orientation_check[b].item()}, vert={vertical_check[b].item()}, "
                        f"quat_dot={quat_dot[b].item():.4f}, xy_drift={xy_drift[b].item():.4f}m, score={score[b].item():.2f}")
            else:
                print(f"[INFO] Orientation dot - mean: {quat_dot.mean().item():.4f}, "
                    f"min: {quat_dot.min().item():.4f}")
                print(f"[INFO] XY drift - mean: {xy_drift.mean().item():.4f}m, "
                    f"max: {xy_drift.max().item():.4f}m")
        
        # Single GPU→CPU transfer at the end
        # Batch GPU→CPU transfer for return values
        lifted_cpu = lifted.cpu()
        score_cpu = score.cpu()
        return lifted_cpu.numpy().astype(bool), score_cpu.numpy().astype(np.float32)

    def build_grasp_info(
        self,
        grasp_pos_w_batch: np.ndarray,   # (B,3)  GraspNet proposal in world frame
        grasp_quat_w_batch: np.ndarray,  # (B,4)  wxyz
        pre_dist_batch: np.ndarray,      # (B,)
        delta_batch: np.ndarray          # (B,)
    ) -> dict:
        """
        return grasp info dict for all envs in batch.
        """
        B = self.scene.num_envs
        p_w   = np.asarray(grasp_pos_w_batch,  dtype=np.float32).reshape(B, 3)
        q_w   = np.asarray(grasp_quat_w_batch, dtype=np.float32).reshape(B, 4)
        pre_d = np.asarray(pre_dist_batch,     dtype=np.float32).reshape(B)
        delt  = np.asarray(delta_batch,        dtype=np.float32).reshape(B)

        a_batch = grasp_approach_axis_batch(q_w)  # (B,3)

        pre_p_w = (p_w - pre_d[:, None] * a_batch).astype(np.float32)
        gra_p_w = (p_w + delt[:,  None] * a_batch).astype(np.float32)

        origins = self.scene.env_origins.detach().cpu().numpy().astype(np.float32)  # (B,3)
        pre_p_w = pre_p_w + origins
        gra_p_w = gra_p_w + origins

        pre_pb, pre_qb = self._to_base(pre_p_w, q_w)
        gra_pb, gra_qb = self._to_base(gra_p_w, q_w)

        return {
            "pre_p_w": pre_p_w, "p_w": gra_p_w, "q_w": q_w,
            "pre_p_b": pre_pb,  "pre_q_b": pre_qb,
            "p_b": gra_pb,      "q_b": gra_qb,
            "pre_dist": pre_d,  "delta": delt,
        }


    def grasp_trials(self, gg, std: float = 0.0005, max_retries: int = 3, existing_skip_mask: np.ndarray = None):
        """
        Async grasp trials: each env finds its own successful grasp independently.
        Environments that succeed early keep their grasp while failed envs continue searching.
        
        Args:
            gg: GraspGroup with grasp proposals
            std: Standard deviation for random perturbations
            max_retries: Number of retry attempts
            existing_skip_mask: Optional [B] boolean array of environments to skip from the start
        
        Returns dict with per-env success flags, chosen grasp parameters, and skip mask.
        Envs that exhaust retries are marked to skip in future processing.
        """
        B = self.scene.num_envs
        idx_all = list(range(len(gg)))
        if len(idx_all) == 0:
            print("[ERR] empty grasp list.")
            skip_mask = np.ones(B, dtype=bool) if existing_skip_mask is None else existing_skip_mask.copy()
            return {
                "success": np.zeros(B, dtype=bool),
                "chosen_pose_w": [None] * B,
                "chosen_pre": [None] * B,
                "chosen_delta": [None] * B,
                "skip_envs": skip_mask,
            }

        rng = np.random.default_rng()
        pre_dist_const = 0.12  # m

        # Track per-environment success and chosen grasps
        env_success = np.zeros(B, dtype=bool)
        env_chosen_pose_w = [None] * B  # List of (pos, quat) tuples per env
        env_chosen_pre = np.zeros(B, dtype=np.float32)
        env_chosen_delta = np.zeros(B, dtype=np.float32)
        env_best_score = np.zeros(B, dtype=np.float32)
        
        # Initialize skip mask from existing one or create new
        if existing_skip_mask is not None:
            env_skip = existing_skip_mask.copy()
            if self.verbose:
                print(f"[INFO] Starting grasp trials with {env_skip.sum()} pre-skipped environments: {np.where(env_skip)[0].tolist()}")
        else:
            env_skip = np.zeros(B, dtype=bool)  # Track envs to skip due to exhausted retries
        
        # Retry loop for grasp selection phase
        for retry_attempt in range(max_retries):
            try:
                print(f"\n{'='*60}")
                print(f"ASYNC GRASP SELECTION ATTEMPT {retry_attempt + 1}/{max_retries}")
                print(f"Successful envs: {env_success.sum()}/{B}")
                print(f"{'='*60}\n")
                
                # If all non-skipped envs succeeded, we're done
                active_mask = ~env_skip
                if np.all(env_success[active_mask]):
                    successful_envs = np.where(env_success & ~env_skip)[0]
                    skipped_envs = np.where(env_skip)[0]
                    if self.verbose:
                        print(f"[SUCCESS] All active environments found valid grasps!")
                    if len(skipped_envs) > 0 and self.verbose:
                        print(f"[INFO] Skipped {len(skipped_envs)} failed environments: {skipped_envs.tolist()}")
                    break
                
                # Get mask of envs that still need to find a grasp (not successful and not skipped)
                active_failed_mask = (~env_success) & (~env_skip)
                failed_env_ids = np.where(active_failed_mask)[0]
                n_failed = len(failed_env_ids)
                
                if n_failed == 0:
                    # All remaining envs are either successful or skipped
                    break
                
                if self.verbose:
                    print(f"[INFO] {n_failed} environments still searching for grasps: {failed_env_ids.tolist()}")
                
                # Iterate through grasp proposals
                for start in range(0, len(idx_all), B):
                    block = idx_all[start : start + B]
                    if len(block) < B:
                        block = block + [block[-1]] * (B - len(block))
                    
                    # Log progress for this block
                    print(f"[SEARCH] Processing grasp block [{start}:{min(start+B, len(idx_all))}] / {len(idx_all)}, "
                          f"Current successes: {env_success.sum()}/{B}")

                    # Build grasp proposals for ALL envs (even successful ones get dummy data)
                    # Pre-allocate arrays to avoid list+append overhead
                    grasp_pos_w_batch = np.zeros((B, 3), dtype=np.float32)
                    grasp_quat_w_batch = np.zeros((B, 4), dtype=np.float32)
                    for b in range(B):
                        if env_success[b] or env_skip[b]:
                            # Use existing successful grasp or dummy for skipped env
                            if env_success[b]:
                                p_w, q_w = env_chosen_pose_w[b]
                            else:
                                # Dummy grasp for skipped env
                                p_w, q_w = grasp_to_world(gg[0])
                                # Transform to randomized pose even for dummy
                                p_w, q_w = self.transform_grasp_to_randomized_pose(p_w, q_w, b)
                            grasp_pos_w_batch[b] = p_w.astype(np.float32)
                            grasp_quat_w_batch[b] = q_w.astype(np.float32)
                        else:
                            # New grasp proposal for failed env
                            idx = block[b % len(block)]
                            p_w, q_w = grasp_to_world(gg[int(idx)])
                            # CRITICAL: Transform grasp to account for random object pose
                            p_w, q_w = self.transform_grasp_to_randomized_pose(p_w, q_w, b)
                            grasp_pos_w_batch[b] = p_w.astype(np.float32)
                            grasp_quat_w_batch[b] = q_w.astype(np.float32)
                    
                    self.show_goal(grasp_pos_w_batch, grasp_quat_w_batch)
                    
                    # Random disturbance along approach axis
                    pre_dist_batch = np.full((B,), pre_dist_const, dtype=np.float32)
                    delta_batch = rng.normal(0.0, std, size=(B,)).astype(np.float32)

                    info = self.build_grasp_info(grasp_pos_w_batch, grasp_quat_w_batch,
                                                pre_dist_batch, delta_batch)

                    # Execute grasp trial for ALL envs (no recording during search for speed)
                    ok_batch, score_batch = self.execute_and_lift_once_batch(info, record=False)
                    
                    # Track newly successful envs in this iteration
                    newly_successful_count = 0
                    
                    # Update success status ONLY for envs that were still searching
                    for b in range(B):
                        if not env_success[b] and not env_skip[b]:  # Only update if active and not already successful
                            if ok_batch[b] and score_batch[b] > env_best_score[b]:
                                # This env found a better grasp
                                env_success[b] = True
                                env_chosen_pose_w[b] = (grasp_pos_w_batch[b], grasp_quat_w_batch[b])
                                env_chosen_pre[b] = pre_dist_batch[b]
                                env_chosen_delta[b] = delta_batch[b]
                                env_best_score[b] = score_batch[b]
                                newly_successful_count += 1
                                if self.verbose:
                                    print(f"[SUCCESS] Env {b} found valid grasp (score: {score_batch[b]:.2f})")
                    
                    # Check if all active envs now succeeded
                    active_mask = ~env_skip
                    if self.verbose and newly_successful_count > 0:
                        print(f"[SEARCH] block[{start}:{start+B}] -> {newly_successful_count} new successes")
                    
                    # Early termination: all active envs succeeded
                    if np.all(env_success[active_mask]):
                        successful_envs = np.where(env_success & ~env_skip)[0]
                        skipped_envs = np.where(env_skip)[0]
                        if self.verbose:
                            print(f"[SUCCESS] All active environments found valid grasps! Breaking early.")
                        if len(skipped_envs) > 0 and self.verbose:
                            print(f"[INFO] {len(skipped_envs)} environments skipped: {skipped_envs.tolist()}")
                        return {
                            "success": env_success,
                            "chosen_pose_w": env_chosen_pose_w,
                            "chosen_pre": env_chosen_pre,
                            "chosen_delta": env_chosen_delta,
                            "skip_envs": env_skip,
                        }
                
                # End of this attempt - check status
                active_mask = ~env_skip
                if not np.all(env_success[active_mask]):
                    still_failed = np.where((~env_success) & (~env_skip))[0]
                    if self.verbose:
                        print(f"[WARN] {len(still_failed)} envs still need grasps after attempt {retry_attempt + 1}: {still_failed.tolist()}")
                    if retry_attempt < max_retries - 1:
                        if self.verbose:
                            print("[INFO] Retrying grasp selection for failed environments...")
                        continue
                    else:
                        # Last attempt exhausted - mark remaining failed envs as skipped
                        if self.verbose:
                            print(f"[WARN] Exhausted all {max_retries} grasp attempts")
                            print(f"[INFO] Marking {len(still_failed)} failed environments to skip: {still_failed.tolist()}")
                        env_skip[still_failed] = True
                        
                        successful_envs = np.where(env_success)[0]
                        if self.verbose:
                            print(f"[INFO] Proceeding with {len(successful_envs)} successful environments: {successful_envs.tolist()}")
                        
                        return {
                            "success": env_success,
                            "chosen_pose_w": env_chosen_pose_w,
                            "chosen_pre": env_chosen_pre,
                            "chosen_delta": env_chosen_delta,
                            "skip_envs": env_skip,
                        }
                        
            except RuntimeError as e:
                if "GraspTrialMotionPlanningFailure" in str(e):
                    print(f"\n[RESTART] Motion planning failed during grasp trial on attempt {retry_attempt + 1}")
                    print(f"[RESTART] Remaining grasp trial attempts: {max_retries - retry_attempt - 1}\n")
                    
                    # Clear corrupted data and continue to next retry
                    self.clear_data()
                    
                    if retry_attempt < max_retries - 1:
                        continue
                    else:
                        print("[ERR] Grasp trial failed after all retry attempts")
                        # Mark all currently failed envs as skipped
                        failed_envs = np.where((~env_success) & (~env_skip))[0]
                        if len(failed_envs) > 0 and self.verbose:
                            print(f"[INFO] Marking {len(failed_envs)} failed environments to skip: {failed_envs.tolist()}")
                        if len(failed_envs) > 0:
                            env_skip[failed_envs] = True
                        return {
                            "success": env_success,
                            "chosen_pose_w": env_chosen_pose_w,
                            "chosen_pre": env_chosen_pre,
                            "chosen_delta": env_chosen_delta,
                            "skip_envs": env_skip,
                        }
                else:
                    raise
                    
            except Exception as e:
                print(f"[ERROR] Unexpected error during grasp trial: {type(e).__name__}: {e}")
                if retry_attempt < max_retries - 1:
                    print(f"[ERROR] Attempting grasp trial retry {retry_attempt + 1}/{max_retries}...")
                    self.clear_data()
                    continue
                else:
                    # Mark all currently failed envs as skipped
                    failed_envs = np.where((~env_success) & (~env_skip))[0]
                    if len(failed_envs) > 0 and self.verbose:
                        print(f"[INFO] Marking {len(failed_envs)} failed environments to skip: {failed_envs.tolist()}")
                    if len(failed_envs) > 0:
                        env_skip[failed_envs] = True
                    
                    # Replace None values with dummy poses for skipped envs to avoid downstream NoneType errors
                    for b in np.where(env_skip)[0]:
                        if env_chosen_pose_w[b] is None:
                            dummy_pos = np.array([0, 0, 0], dtype=np.float32)
                            dummy_quat = np.array([1, 0, 0, 0], dtype=np.float32)
                            env_chosen_pose_w[b] = (dummy_pos, dummy_quat)
                    
                    return {
                        "success": env_success,
                        "chosen_pose_w": env_chosen_pose_w,
                        "chosen_pre": env_chosen_pre,
                        "chosen_delta": env_chosen_delta,
                        "skip_envs": env_skip,
                    }
        
        # Replace None values with dummy poses for skipped envs to avoid downstream NoneType errors
        for b in np.where(env_skip)[0]:
            if env_chosen_pose_w[b] is None:
                dummy_pos = np.array([0, 0, 0], dtype=np.float32)
                dummy_quat = np.array([1, 0, 0, 0], dtype=np.float32)
                env_chosen_pose_w[b] = (dummy_pos, dummy_quat)
        
        # Return final status
        return {
            "success": env_success,
            "chosen_pose_w": env_chosen_pose_w,
            "chosen_pre": env_chosen_pre,
            "chosen_delta": env_chosen_delta,
            "skip_envs": env_skip,
        }

    
    def is_success(self, position_threshold: float = 0.10, skip_envs: np.ndarray = None) -> torch.Tensor:
        """
        Verify if the manipulation task succeeded by checking:
        1. Object is at Goal (Distance < 10cm)
        2. Gripper is at Goal (Distance < 10cm) - Explicit check using T_ee_in_obj
        3. Object is in Gripper (Deviation < 2cm)
        
        Args:
            position_threshold: Distance threshold for Object-Goal check (default: 0.10m = 10cm)
            skip_envs: Boolean array [B] indicating which envs to skip from verification
        
        Returns:
            torch.Tensor: Boolean tensor [B] indicating success for each environment
        """
        B = self.scene.num_envs
        if skip_envs is None:
            skip_envs = np.zeros(B, dtype=bool)
        
        # --- 1. Object Goal Check ---
        final_goal_matrices = self.obj_goal_traj_w[:, -1, :, :]  # [B, 4, 4]
        goal_positions_np = final_goal_matrices[:, :3, 3]
        goal_positions = torch.tensor(goal_positions_np, dtype=torch.float32, device=self.sim.device)
        
        # Current Root and COM
        current_root_state = self.object_prim.data.root_state_w
        current_root_pos = current_root_state[:, :3]
        current_root_quat = current_root_state[:, 3:7]
        current_com_pos = self.object_prim.data.root_com_pos_w[:, :3]
        
        # Calculate Goal COM positions
        # The COM offset is constant in the object's local frame.
        # We need to: (1) Get the current COM offset in local frame, (2) Apply to goal pose
        goal_com_positions_list = []
        for b in range(B):
            # Current state
            root_pos_cur = current_root_pos[b].cpu().numpy()
            root_quat_cur = current_root_quat[b].cpu().numpy()  # wxyz format
            com_pos_cur = current_com_pos[b].cpu().numpy()
            
            # COM offset in world frame
            com_offset_world = com_pos_cur - root_pos_cur  # [3]
            
            # Convert COM offset to object's local frame
            # R_cur^T @ com_offset_world gives the offset in local coords (assuming no scale)
            R_cur = transforms3d.quaternions.quat2mat(root_quat_cur)  # Convert wxyz to rotation matrix
            com_offset_local = R_cur.T @ com_offset_world  # Rotate to local frame
            
            # Goal state
            T_goal_root = final_goal_matrices[b]  # [4, 4] numpy array
            goal_root_pos = T_goal_root[:3, 3]
            R_goal = T_goal_root[:3, :3]
            
            # Apply local offset to goal pose
            # goal_com_pos = goal_root_pos + R_goal @ com_offset_local
            com_offset_world_goal = R_goal @ com_offset_local
            goal_com_pos = goal_root_pos + com_offset_world_goal
            
            goal_com_positions_list.append(goal_com_pos)
            
        goal_com_positions = torch.tensor(np.array(goal_com_positions_list), dtype=torch.float32, device=self.sim.device)
        
        # Calculate Distances
        root_dist = torch.norm(current_root_pos - goal_positions, dim=1)
        com_dist = torch.norm(current_com_pos - goal_com_positions, dim=1)
        
        # Success requires BOTH Root and COM to be within threshold
        obj_success = (root_dist <= position_threshold) & (com_dist <= position_threshold)

        # --- 2. Gripper Goal Check ---
        # Calculate Target Gripper Pose: T_ee_goal = T_obj_goal @ T_ee_in_obj
        # We use the stored T_ee_in_obj from the start of the trajectory
        ee_pos_final = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:3]
        
        if hasattr(self, 'T_ee_in_obj') and self.T_ee_in_obj is not None:
            target_ee_pos_list = []
            for b in range(B):
                T_obj_goal = final_goal_matrices[b]
                T_ee_goal = T_obj_goal @ self.T_ee_in_obj[b]
                target_ee_pos_list.append(T_ee_goal[:3, 3])
            
            target_ee_pos = torch.tensor(np.array(target_ee_pos_list), dtype=torch.float32, device=self.sim.device)
            grip_goal_dist = torch.norm(ee_pos_final - target_ee_pos, dim=1)
            gripper_success = grip_goal_dist <= 0.10
        else:
            # Fallback if T_ee_in_obj not available (should not happen if follow_object_goals ran)
            print("[WARN] T_ee_in_obj not found, skipping explicit Gripper Goal Check")
            gripper_success = torch.ones(B, dtype=torch.bool, device=self.sim.device)
            grip_goal_dist = torch.zeros(B, dtype=torch.float32, device=self.sim.device)

        # --- 3. Holding Check (Object in Gripper) ---
        # Check if the distance between Gripper and Object COM has remained stable.
        # We compare the final distance to the initial grasp distance.
        obj_com_final = self.object_prim.data.root_com_pos_w[:, 0:3]
        current_grip_com_dist = torch.norm(ee_pos_final - obj_com_final, dim=1)
        
        if hasattr(self, 'initial_grasp_dist') and self.initial_grasp_dist is not None:
            # Check for deviation from initial grasp configuration
            # If the object slipped or was dropped, this distance will change.
            # We use a strict 2cm threshold. 
            # - 0.00m is impossible due to physics engine noise/soft contacts.
            # - 0.02m allows for micro-settling but catches any real slip or drag.
            grasp_deviation = torch.abs(current_grip_com_dist - self.initial_grasp_dist)
            holding_success = grasp_deviation <= 0.02
            
            holding_metric = grasp_deviation
            holding_threshold = 0.02
            holding_metric_name = "Grip-COM Deviation"
        else:
            # Fallback to absolute check if initial dist missing (shouldn't happen)
            # Relaxed to 15cm to account for tool handles if we don't have initial baseline
            holding_success = current_grip_com_dist <= 0.15
            holding_metric = current_grip_com_dist
            holding_threshold = 0.15
            holding_metric_name = "Grip-COM Dist"

        # Combined Success
        success_mask = obj_success & gripper_success & holding_success
        
        # Print results
        print("\n" + "="*50)
        print("TASK VERIFICATION RESULTS (Strict Multi-Check)")
        print("="*50)
        for b in range(B):
            if skip_envs[b]:
                print(f"Env {b}: SKIPPED (environment was skipped during grasp phase)")
            else:
                status = "SUCCESS" if success_mask[b].item() else "FAILURE"
                fail_reason = ""
                if not obj_success[b].item():
                    fail_reason += f"[Obj-Goal Dist Root:{root_dist[b].item():.3f}m/COM:{com_dist[b].item():.3f}m > {position_threshold}m] "
                if not gripper_success[b].item():
                    fail_reason += f"[Grip-Goal Dist {grip_goal_dist[b].item():.3f}m > 0.10m] "
                if not holding_success[b].item():
                    fail_reason += f"[Not Held ({holding_metric_name} {holding_metric[b].item():.3f}m > {holding_threshold}m)]"
                
                print(f"Env {b}: {status} {fail_reason}")
                print(f"  Obj-Goal Root Dist: {root_dist[b].item():.4f}m")
                print(f"  Obj-Goal COM Dist:  {com_dist[b].item():.4f}m")
                print(f"  Grip-Goal Dist:     {grip_goal_dist[b].item():.4f}m")
                print(f"  {holding_metric_name}: {holding_metric[b].item():.4f}m (Initial: {self.initial_grasp_dist[b].item():.4f}m)" if hasattr(self, 'initial_grasp_dist') else f"  {holding_metric_name}: {holding_metric[b].item():.4f}m")
        
        # Overall result - only check active (non-skipped) environments
        skip_envs_np = skip_envs  # Already numpy
        success_mask_np = success_mask.cpu().numpy()  # Convert torch to numpy
        
        active_mask = ~skip_envs_np
        active_success_mask = success_mask_np[active_mask]
        all_success = np.all(active_success_mask)
        
        print("="*50)
        if all_success:
            active_count = np.sum(active_mask)
            print(f"TASK VERIFIER: SUCCESS - All {active_count} active environments completed successfully!")
        else:
            active_success_count = np.sum(active_success_mask)
            active_count = np.sum(active_mask)
            print(f"TASK VERIFIER: FAILURE - {active_success_count}/{active_count} active environments succeeded")
        print("="*50 + "\n")
        
        return success_mask
    
    def save_data_selective(self, skip_envs: np.ndarray, ignore_keys: List[str] = None, 
                           demo_id: int = None, env_offset: int = 0, successful_env_data: dict = None):
        """
        Save data for successful environments.
        
        Args:
            skip_envs: Boolean array indicating which envs to skip
            ignore_keys: Keys to not save
            demo_id: Optional demo ID to use (if None, generates new ID)
            env_offset: Offset to add to environment IDs when saving (for multi-run collection)
            successful_env_data: Dict mapping env_id to {key: np.array} with actual trajectory data (NO padding!)
        """
        print("\n" + "="*60)
        print("[DEBUG] save_data_selective() called")
        print(f"[DEBUG] skip_envs shape: {skip_envs.shape}, sum: {skip_envs.sum()}")
        print(f"[DEBUG] demo_id: {demo_id}, env_offset: {env_offset}")
        print("="*60 + "\n")
        
        if ignore_keys is None:
            ignore_keys = []
            
        print("[DEBUG] Creating save directory...")
        save_root = self._demo_dir()
        if demo_id is not None:
            # Use provided demo_id instead of auto-generated one
            save_root = self.out_dir / img_folder / "demos" / f"demo_{demo_id}"
        save_root.mkdir(parents=True, exist_ok=True)
        print(f"[DEBUG] Save root: {save_root}")
        
        # Use successful_env_data if provided (NO PADDING!), otherwise fall back to save_dict
        use_direct_data = (successful_env_data is not None and len(successful_env_data) > 0)
        
        if use_direct_data:
            print(f"[DEBUG] Using successful_env_data directly (NO padding, each env keeps actual trajectory length)")
            for env_id, env_data in successful_env_data.items():
                first_key = list(env_data.keys())[0]
                traj_len = env_data[first_key].shape[0]
                print(f"[DEBUG]   Env {env_id}: {traj_len} frames")
        else:
            print(f"[DEBUG] Falling back to save_dict (legacy mode)")

        # Stack data with proper shape handling (ONLY if not using direct data)
        stacked = {}
        if not use_direct_data:
            print("[DEBUG] Starting data stacking phase...")
            print(f"[DEBUG] save_dict has {len(self.save_dict)} keys: {list(self.save_dict.keys())}")
            
            for k, v in self.save_dict.items():
                print(f"[DEBUG] Processing key '{k}': type={type(v)}, len={len(v) if hasattr(v, '__len__') else 'N/A'}")
                try:
                    arr = np.array(v)
                    print(f"[DEBUG]   Initial array shape: {arr.shape}, dtype: {arr.dtype}")
                    # Check if we got an object array (inconsistent shapes)
                    if arr.dtype == object:
                        # Try to stack with explicit dtype
                        if len(v) > 0 and hasattr(v[0], 'shape'):
                            print(f"[DEBUG]   Attempting explicit stack for object array...")
                            arr = np.stack(v, axis=0)
                            print(f"[DEBUG]   Stack successful: shape={arr.shape}")
                    stacked[k] = arr
                    print(f"[DEBUG]   Final stacked shape for '{k}': {arr.shape}")
                except Exception as e:
                    print(f"[WARN] Failed to stack {k}: {e}. Saving as list.")
                    stacked[k] = v
            
            print(f"[DEBUG] Data stacking complete. Stacked {len(stacked)} keys")
        
        active_envs = np.where(~skip_envs)[0]
        print(f"[INFO] Saving data for {len(active_envs)} active environments: {active_envs.tolist()}")

        def save_env_data(b):
            """Save data for a single environment"""
            print(f"\n[DEBUG] save_env_data() called for env {b}")
            start_time = time.time()
            # Add env_offset to the saved environment ID
            save_env_id = b + env_offset
            print(f"[DEBUG] Env {b}: save_env_id={save_env_id}, creating directory...")
            env_dir = self._env_dir(save_root, save_env_id)
            print(f"[DEBUG] Env {b}: directory created at {env_dir}")
            
            # Get data source: either direct successful_env_data or legacy stacked
            if use_direct_data:
                if b not in successful_env_data:
                    print(f"[WARN] Env {b} not in successful_env_data, skipping")
                    return
                data_dict = successful_env_data[b]
                traj_len = len(next(iter(data_dict.values())))
                print(f"[DEBUG] Env {b}: Using direct data, trajectory length = {traj_len} frames")
            else:
                data_dict = stacked
                print(f"[DEBUG] Env {b}: Using legacy stacked data, processing {len(stacked)} keys...")
            
            for key_idx, (key, data_source) in enumerate(data_dict.items()):
                # Skip ignored keys
                if key in (ignore_keys or []):
                    print(f"[DEBUG] Env {b}: Skipping '{key}' (in ignore list)")
                    continue
                
                # Extract env data based on mode
                if use_direct_data:
                    # Direct mode: data is already per-env, no slicing needed!
                    env_data = data_source
                    print(f"[DEBUG] Env {b}: Processing key '{key}' (shape: {env_data.shape}, dtype: {env_data.dtype})")
                else:
                    # Legacy mode: slice from stacked array
                    print(f"[DEBUG] Env {b}: Processing key {key_idx+1}/{len(stacked)}: '{key}' (shape: {data_source.shape}, dtype: {data_source.dtype})")
                    
                    if isinstance(data_source, list):
                        print(f"[DEBUG] Env {b}: Skipping '{key}' (list type)")
                        continue
                    
                    if data_source.ndim < 2:
                        print(f"[DEBUG] Env {b}: Skipping '{key}' (insufficient dimensions, ndim={data_source.ndim})")
                        continue
                    
                    # Extract this env's data slice
                    try:
                        print(f"[DEBUG] Env {b}: Attempting to extract slice [:, {b}] from shape {data_source.shape}...")
                        # Slice creates new array, no need for .copy()
                        env_data = data_source[:, b]
                        print(f"[DEBUG] Env {b}: Extracted '{key}' slice successfully, shape={env_data.shape}")
                    except Exception as e:
                        print(f"[ERROR] Env {b}: Failed to extract slice for '{key}': {type(e).__name__}: {e}")
                        print(f"[ERROR] Env {b}: Array shape was {data_source.shape}, tried to access [:, {b}]")
                        continue
                
                # Now process env_data (same for both modes)
                if key in (ignore_keys or []):
                    print(f"[DEBUG] Env {b}: Skipping '{key}' (in ignore list, double-check)")
                    continue
                    print(f"[DEBUG] Env {b}: Skipping '{key}' (in ignore list)")
                    continue
                
                if key == "rgb":
                    print(f"[DEBUG] Env {b}: Encoding RGB video ({env_data.shape[0]} frames)...")
                    video_path = env_dir / "sim_video.mp4"
                    writer = imageio.get_writer(
                        video_path, fps=50, codec='libx264', quality=7,
                        pixelformat='yuv420p', macro_block_size=None
                    )
                    # Write frames and explicitly delete to free memory during encoding
                    for t in range(env_data.shape[0]):
                        writer.append_data(env_data[t])
                    writer.close()
                    del writer  # Close and delete writer to flush buffers
                    del env_data  # Free memory immediately
                    import gc
                    gc.collect()  # Force garbage collection
                    print(f"[DEBUG] Env {b}: RGB video encoding complete, memory freed")
                elif key == "segmask":
                    print(f"[DEBUG] Env {b}: Encoding segmask video ({env_data.shape[0]} frames)...")
                    video_path = env_dir / "mask_video.mp4"
                    writer = imageio.get_writer(
                        video_path, fps=50, codec='libx264', quality=7,
                        pixelformat='yuv420p', macro_block_size=None
                    )
                    for t in range(env_data.shape[0]):
                        writer.append_data((env_data[t].astype(np.uint8) * 255))
                    writer.close()
                    del writer
                    del env_data  # Free memory immediately
                    import gc
                    gc.collect()
                    print(f"[DEBUG] Env {b}: Segmask video encoding complete, memory freed")
                elif key == "depth":
                    print(f"[DEBUG] Env {b}: Processing depth data...")
                    # Save original depth first (float16, small)
                    print(f"[DEBUG] Env {b}: Saving depth .npy first...")
                    np.save(env_dir / f"{key}.npy", env_data)
                    
                    # Now process for video - work with minimal memory
                    print(f"[DEBUG] Env {b}: Converting to float32 for percentile...")
                    # Only keep positive values for percentile (much smaller array)
                    if env_data.dtype == np.float16:
                        flat = env_data[env_data > 0].astype(np.float32)
                    else:
                        flat = env_data[env_data > 0]
                    max_depth = np.percentile(flat, 99) if flat.size > 0 else 1.0
                    del flat  # Free immediately
                    
                    print(f"[DEBUG] Env {b}: Encoding depth video in chunks (max={max_depth:.3f})...")
                    video_path = env_dir / "depth_video.mp4"
                    writer = imageio.get_writer(
                        video_path, fps=50, codec='libx264', quality=7,
                        pixelformat='yuv420p', macro_block_size=None
                    )
                    # Process in chunks of 50 frames to balance speed and memory
                    chunk_size = 50
                    for chunk_start in range(0, env_data.shape[0], chunk_size):
                        chunk_end = min(chunk_start + chunk_size, env_data.shape[0])
                        chunk = env_data[chunk_start:chunk_end]
                        chunk_f32 = chunk.astype(np.float32) if chunk.dtype == np.float16 else chunk
                        chunk_norm = np.clip(chunk_f32 / max_depth * 255.0, 0, 255).astype(np.uint8)
                        for t in range(chunk_norm.shape[0]):
                            writer.append_data(chunk_norm[t])
                        del chunk, chunk_f32, chunk_norm  # Free chunk memory
                    writer.close()
                    del writer
                    del env_data  # Free original depth data
                    import gc
                    gc.collect()
                    print(f"[DEBUG] Env {b}: Depth processing complete, memory freed")
                else:
                    print(f"[DEBUG] Env {b}: Saving '{key}' as .npy (shape: {env_data.shape})...")
                    np.save(env_dir / f"{key}.npy", env_data)
                    del env_data  # Free memory
                    print(f"[DEBUG] Env {b}: '{key}' .npy saved, memory freed")
            
            print(f"[DEBUG] Env {b}: Saving config.json...")
            json.dump(self.sim_cfgs, open(env_dir / "config.json", "w"), indent=2)
            elapsed = time.time() - start_time
            print(f"[INFO] Env {save_env_id} saved in {elapsed:.1f}s")
            return b

        # Save environments with parallel encoding and extensive debug logging
        print(f"[DEBUG] Starting save process for {len(active_envs)} environments")
        print(f"[DEBUG] Active environment IDs: {active_envs.tolist()}")
        print(f"[DEBUG] Total memory keys in stacked dict: {list(stacked.keys())}")
        
        # Log memory usage if psutil available
        try:
            import psutil
            import gc
            process = psutil.Process()
            mem_info = process.memory_info()
            print(f"[DEBUG] Memory before encoding: RSS={mem_info.rss / 1024**3:.2f} GB, VMS={mem_info.vms / 1024**3:.2f} GB")
            print(f"[WARN] Memory usage is very high! Running garbage collection...")
            gc.collect()
            mem_info = process.memory_info()
            print(f"[DEBUG] Memory after GC: RSS={mem_info.rss / 1024**3:.2f} GB, VMS={mem_info.vms / 1024**3:.2f} GB")
        except:
            print("[DEBUG] psutil not available, skipping memory logging")
        
        print("[DEBUG] Starting parallel video encoding with ThreadPoolExecutor...")
        print("[DEBUG] NOTE: Processing only active environment data to reduce memory")
        with ThreadPoolExecutor(max_workers=min(len(active_envs), 4)) as executor:
            print(f"[DEBUG] ThreadPoolExecutor created with max_workers={min(len(active_envs), 4)}")
            print("[DEBUG] Submitting save tasks to executor...")
            futures = {executor.submit(save_env_data, env_id): env_id for env_id in active_envs}
            print(f"[DEBUG] Submitted {len(futures)} tasks to executor")
            
            print("[DEBUG] Waiting for encoding tasks to complete...")
            completed = 0
            for future in futures:
                env_id = futures[future]
                print(f"[DEBUG] Waiting for env {env_id} to complete (task {completed+1}/{len(futures)})...")
                try:
                    result = future.result()
                    completed += 1
                    print(f"[DEBUG] Env {env_id} completed successfully ({completed}/{len(futures)})")
                    
                    # CRITICAL: Free this env's data immediately after saving
                    if use_direct_data and env_id in successful_env_data:
                        print(f"[DEBUG] Freeing successful_env_data for env {env_id}")
                        del successful_env_data[env_id]
                        import gc
                        gc.collect()
                    
                    # Log memory after each completion
                    try:
                        process = psutil.Process()
                        mem_info = process.memory_info()
                        print(f"[DEBUG] Memory after env {env_id}: RSS={mem_info.rss / 1024**3:.2f} GB")
                    except:
                        pass
                except Exception as e:
                    print(f"[ERROR] Env {env_id} failed: {type(e).__name__}: {e}")
                    raise
        
        print("[DEBUG] All video encoding tasks completed successfully")
        print("[INFO]: Demonstration is saved at: ", save_root)
        
        # Compose real videos with parallel processing and debug logging
        print("\n[DEBUG] Starting real video composition phase...")
        print(f"[DEBUG] Will compose videos for {len(active_envs)} environments")
        
        def compose_video(b):
            save_env_id = b + env_offset
            print(f"[DEBUG] Starting composition for env {save_env_id} (offset env {b})...")
            success = self.compose_real_video(env_id=save_env_id, demo_path=save_root)
            print(f"[DEBUG] Composition for env {save_env_id} {'succeeded' if success else 'failed'}")
            return (save_env_id, success)
        
        with ThreadPoolExecutor(max_workers=min(len(active_envs), 4)) as executor:
            print(f"[DEBUG] ThreadPoolExecutor created for composition with max_workers={min(len(active_envs), 4)}")
            print("[DEBUG] Submitting composition tasks...")
            results = list(executor.map(compose_video, active_envs))
            print(f"[DEBUG] All {len(results)} composition tasks completed")
        
        print("\n[DEBUG] Real video composition results:")
        for save_env_id, success in results:
            if success:
                print(f"[INFO] Real video composed successfully for env {save_env_id}")
            else:
                print(f"[WARN] Failed to compose real video for env {save_env_id}")

        print("\n[DEBUG] Creating all_demos directory structure...")
        demo_root = self.out_dir / "all_demos"
        demo_root.mkdir(parents=True, exist_ok=True)
        
        # Get starting demo ID ONCE before loop
        print(f"[DEBUG] Getting next demo ID from {demo_root}...")
        starting_demo_id = get_next_demo_id(demo_root)
        print(f"[DEBUG] Starting demo ID: {starting_demo_id}")
        
        # Create separate demo folder for EACH successful environment
        demos_created = 0
        for idx, b in enumerate(active_envs):
            save_env_id = b + env_offset
            
            # Use sequential IDs: starting_demo_id + idx
            total_demo_id = starting_demo_id + idx
            print(f"[DEBUG] Creating demo {total_demo_id} for env {save_env_id}...")
            demo_save_path = demo_root / f"demo_{total_demo_id}"
            print(f"[DEBUG] Creating demo directory: {demo_save_path}")
            demo_save_path.mkdir(parents=True, exist_ok=True)
            
            print(f"[DEBUG] Creating meta_info.json for env {save_env_id}...")
            meta_info = {
                "path": str(save_root),
                "fps": 50,
                "env_id": int(save_env_id),  # Convert numpy int64 to Python int
                "source_demo_id": int(demo_id) if demo_id is not None else "N/A",
            }
            with open(demo_save_path / "meta_info.json", "w") as f:
                json.dump(meta_info, f)
            print("[DEBUG] meta_info.json saved")
            
            # Copy this specific env's data (env directories use 3-digit format: env_000)
            env_source = save_root / f"env_{save_env_id:03d}"
            print(f"[DEBUG] Copying files from {env_source} to {demo_save_path}...")
            if env_source.exists():
                os.system(f"cp -r {env_source}/* {demo_save_path}")
                print(f"[DEBUG] File copy complete for env {save_env_id}")
                print(f"[INFO]: Demo {total_demo_id} saved at: {demo_save_path}")
                demos_created += 1
            else:
                print(f"[ERROR] Source directory {env_source} does not exist!")
        
        print(f"\n[DEBUG] save_data_selective() complete. Created {demos_created} demo folders")
        # Return number of demo folders actually created
        return demos_created


    def inference(self, std: float = 0.0, max_grasp_retries: int = 1, 
              max_traj_retries: int = 3, position_threshold: float = 0.1,
              demo_id: int = None, env_offset: int = 0) -> int:
        """
        Main function with decoupled grasp selection and trajectory execution.
        
        Phase 1: Grasp selection - try max_grasp_retries times, mark failures as skipped
        Phase 2: Trajectory execution - try max_traj_retries times per env, mark failures as skipped
        Phase 3: Save only successful environments
        
        NO re-grasping during trajectory phase.
        
        Args:
            demo_id: Optional demo ID to use for saving (for multi-run collection)
            env_offset: Offset to add to environment IDs when saving
            
        Returns:
            Number of successful environments saved
        """
        # Generate fresh random poses for this pipeline run
        # Each inference() call (new demo collection run) gets new random poses
        self._generate_random_initial_poses()
        if self.verbose:
            print("[INFO] Generated new random initial poses for this pipeline run")
        
        # Reload trajectory with new random initial poses
        self.load_obj_goal_traj()
        if self.verbose:
            print("[INFO] Recomputed object goal trajectory with randomized initial poses")
        
        # Memory logging helper
        def log_memory(label):
            try:
                import psutil
                process = psutil.Process()
                mem_info = process.memory_info()
                print(f"[MEMORY] {label}: RSS={mem_info.rss / 1024**3:.2f} GB, VMS={mem_info.vms / 1024**3:.2f} GB")
            except:
                pass
        
        log_memory("INFERENCE START")
        
        B = self.scene.num_envs
        
        # Read grasp proposals (only once)
        npy_path = self.grasp_path
        if npy_path is None or (not os.path.exists(npy_path)):
            print(f"[ERR] grasps npy not found: {npy_path}")
            return False
        gg = GraspGroup().from_npy(npy_file_path=npy_path)
        gg = get_best_grasp_with_hints(gg, point=None, direction=[0, 0, -1])
        
        log_memory("After loading grasps")

        # Track per-environment state
        env_skip = np.zeros(B, dtype=bool)  # Environments to skip (failed grasp or trajectory)
        env_traj_success = np.zeros(B, dtype=bool)  # Track which envs succeeded
        env_traj_attempts = np.zeros(B, dtype=int)  # Track trajectory attempts per env
        
        # ========== PHASE 1: GRASP SELECTION (ONE TIME ONLY) ==========
        print("\n" + "#"*60)
        print("PHASE 1: GRASP SELECTION")
        print("#"*60 + "\n")
        
        log_memory("Before grasp selection")
        
        if self.grasp_idx >= 0:
            # Fixed grasp index mode - all envs use same grasp
            if self.grasp_idx >= len(gg):
                print(f"[ERR] grasp_idx {self.grasp_idx} out of range [0,{len(gg)})")
                return 0
            if self.verbose:
                print(f"[INFO] using fixed grasp index {self.grasp_idx} for all envs.")
            p_w, q_w = grasp_to_world(gg[int(self.grasp_idx)])
            
            # CRITICAL: Transform grasp to each env's randomized pose
            env_chosen_pose_w = []
            for b in range(B):
                p_b, q_b = self.transform_grasp_to_randomized_pose(p_w, q_w, b)
                env_chosen_pose_w.append((p_b, q_b))
            
            env_chosen_pre = np.full(B, self.grasp_pre if self.grasp_pre is not None else 0.12, dtype=np.float32)
            env_chosen_delta = np.full(B, self.grasp_delta if self.grasp_delta is not None else 0.0, dtype=np.float32)
            # No envs skipped in fixed mode
            env_skip = np.zeros(B, dtype=bool)
        else:
            # Automatic async grasp selection
            ret = self.grasp_trials(gg, std=std, max_retries=max_grasp_retries, existing_skip_mask=None)
            
            env_chosen_pose_w = ret["chosen_pose_w"]
            env_chosen_pre = ret["chosen_pre"]
            env_chosen_delta = ret["chosen_delta"]
            env_skip = ret["skip_envs"]  # Envs that failed grasp selection
        
        # Check if any envs succeeded in grasp selection
        active_envs = np.where(~env_skip)[0]
        if len(active_envs) == 0:
            print("[ERR] All environments failed grasp selection")
            return 0
        
        if self.verbose:
            print(f"\n[GRASP SELECTION COMPLETE]")
            print(f"  Active envs: {active_envs.tolist()}")
        if env_skip.sum() > 0 and self.verbose:
            print(f"  Skipped envs (failed grasp): {np.where(env_skip)[0].tolist()}")
        
        log_memory("After grasp selection")
        
        # ========== PHASE 2: TRAJECTORY EXECUTION (WITH PER-ENV RETRY) ==========
        print("\n" + "#"*60)
        print("PHASE 2: TRAJECTORY EXECUTION")
        print("#"*60 + "\n")

        # Track successful env data: each env saves data from the attempt where it succeeded
        successful_env_data = {}  # {env_id: {key: np.array}} - NO PADDING!
        
        log_memory("Before trajectory loop")

        for traj_attempt in range(max_traj_retries):
            try:
                log_memory(f"Trajectory attempt {traj_attempt+1} START")
                print(f"[MEMORY] save_dict keys: {list(self.save_dict.keys())}")
                print(f"[MEMORY] save_dict lengths: {[(k, len(v) if hasattr(v, '__len__') else 'N/A') for k, v in self.save_dict.items()]}")
                
                # Get envs that still need to succeed (not skipped AND not yet successful)
                needs_attempt = (~env_skip) & (~env_traj_success)
                active_envs = np.where(needs_attempt)[0]
                
                if len(active_envs) == 0:
                    # All non-skipped envs have succeeded!
                    successful_envs = np.where(env_traj_success)[0]
                    if self.verbose:
                        print(f"\n[SUCCESS] All {len(successful_envs)} environments completed!")
                        print(f"  Successful envs: {successful_envs.tolist()}")
                    if env_skip.sum() > 0 and self.verbose:
                        print(f"  Skipped envs: {np.where(env_skip)[0].tolist()}")
                    
                    log_memory("Before save_data_selective")
                    num_saved = self.save_data_selective(env_skip, demo_id=demo_id, env_offset=env_offset, 
                                                        successful_env_data=successful_env_data)
                    log_memory("After save_data_selective")
                    
                    # Free successful_env_data after saving
                    successful_env_data.clear()
                    log_memory("After clearing successful_env_data")
                    
                    return num_saved
                
                print(f"\n{'='*60}")
                print(f"TRAJECTORY ATTEMPT {traj_attempt + 1}/{max_traj_retries}")
                if self.verbose:
                    print(f"Envs needing attempt: {active_envs.tolist()}")
                already_successful = np.where(env_traj_success & ~env_skip)[0]
                if len(already_successful) > 0 and self.verbose:
                    print(f"Envs already successful (skipping): {already_successful.tolist()}")
                if self.verbose:
                    print(f"Skipped envs: {np.where(env_skip)[0].tolist()}")
                print(f"{'='*60}\n")
                
                # Prepare grasp info for ALL envs
                # Active envs get real data, skipped/successful envs get dummy data
                p_all = np.zeros((B, 3), dtype=np.float32)
                q_all = np.zeros((B, 4), dtype=np.float32)
                # Direct assignment (will be modified per-env below, so needs to be mutable)
                pre_all = env_chosen_pre
                del_all = env_chosen_delta
                
                for b in range(B):
                    if needs_attempt[b] and env_chosen_pose_w[b] is not None:
                        # Active env needing attempt
                        p_all[b] = env_chosen_pose_w[b][0]
                        q_all[b] = env_chosen_pose_w[b][1]
                    else:
                        # Dummy data for skipped or already successful envs
                        p_all[b] = np.zeros(3, dtype=np.float32)
                        q_all[b] = np.array([1, 0, 0, 0], dtype=np.float32)
                
                info_all = self.build_grasp_info(p_all, q_all, pre_all, del_all)

                # Reset and execute
                log_memory("Before reset()")
                self.reset()
                log_memory("After reset()")
                
                # Capture object position for stability check (XY only)
                obj_pos_start = self.object_prim.data.root_state_w[:, :2].clone()

                # Save grasp poses relative to camera
                cam_p = self.camera.data.pos_w
                cam_q = self.camera.data.quat_w_ros
                gp_w  = torch.as_tensor(info_all["p_w"],     dtype=torch.float32, device=self.sim.device)
                gq_w  = torch.as_tensor(info_all["q_w"],     dtype=torch.float32, device=self.sim.device)
                pre_w = torch.as_tensor(info_all["pre_p_w"], dtype=torch.float32, device=self.sim.device)
                gp_cam,  gq_cam  = subtract_frame_transforms(cam_p, cam_q, gp_w,  gq_w)
                pre_cam, pre_qcm = subtract_frame_transforms(cam_p, cam_q, pre_w, gq_w)
                self.save_dict["grasp_pose_cam"]    = torch.cat([gp_cam,  gq_cam],  dim=1).unsqueeze(0).cpu().numpy()
                self.save_dict["pregrasp_pose_cam"] = torch.cat([pre_cam, pre_qcm], dim=1).unsqueeze(0).cpu().numpy()

                jp = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
                self.wait(gripper_open=True, steps=4)
                
                log_memory("After initial wait")

                # Pre-grasp
                if self.verbose:
                    print("[INFO] Moving to pre-grasp positions...")
                jp, success = self.move_to(info_all["pre_p_b"], info_all["pre_q_b"], gripper_open=True)
                if jp is None or success is None:
                    print("[WARN] Pre-grasp motion planning failed")
                    raise RuntimeError("PreGraspMotionFailure")
                
                log_memory("After pre-grasp")
                
                # Mark which envs to skip in action recording (skipped + already successful)
                envs_to_skip_recording = env_skip | env_traj_success
                
                pre_p_b_data = info_all["pre_p_b"].cpu().numpy()
                pre_q_b_data = info_all["pre_q_b"].cpu().numpy()
                for b in np.where(envs_to_skip_recording)[0]:
                    pre_p_b_data[b] = np.zeros(3, dtype=np.float32)
                    pre_q_b_data[b] = np.array([1, 0, 0, 0], dtype=np.float32)
                self.save_dict["actions"].append(np.concatenate([pre_p_b_data, pre_q_b_data, np.zeros((B, 1))], axis=1))
                jp = self.wait(gripper_open=True, steps=3)

                # Grasp
                if self.verbose:
                    print("[INFO] Moving to grasp positions...")
                jp, success = self.move_to(info_all["p_b"], info_all["q_b"], gripper_open=True)
                if jp is None or success is None:
                    print("[WARN] Grasp motion planning failed")
                    raise RuntimeError("GraspMotionFailure")
                
                p_b_data = info_all["p_b"].cpu().numpy()
                q_b_data = info_all["q_b"].cpu().numpy()
                for b in np.where(envs_to_skip_recording)[0]:
                    p_b_data[b] = np.zeros(3, dtype=np.float32)
                    q_b_data[b] = np.array([1, 0, 0, 0], dtype=np.float32)
                self.save_dict["actions"].append(np.concatenate([p_b_data, q_b_data, np.zeros((B, 1))], axis=1))

                # Close gripper
                if self.verbose:
                    print("[INFO] Closing grippers...")
                jp = self.wait(gripper_open=False, steps=50)
                
                # Check for jerky grasp (XY displacement > 5cm)
                obj_pos_post_grasp = self.object_prim.data.root_state_w[:, :2]
                grasp_displacement = torch.norm(obj_pos_post_grasp - obj_pos_start, dim=1)
                jerky_mask = grasp_displacement > 0.05
                
                if torch.any(jerky_mask):
                    new_jerky = jerky_mask.cpu().numpy() & (~env_skip)
                    if np.any(new_jerky):
                        jerky_indices = np.where(new_jerky)[0]
                        if self.verbose:
                            print(f"[WARN] Jerky grasp detected in envs {jerky_indices.tolist()} (displacement > 5cm)")
                        env_skip[new_jerky] = True
                        
                        # Update recording mask to include these new failures
                        envs_to_skip_recording = env_skip | env_traj_success
                
                log_memory("After gripper close")
                
                p_close_data = info_all["p_b"].cpu().numpy()
                q_close_data = info_all["q_b"].cpu().numpy()
                for b in np.where(envs_to_skip_recording)[0]:
                    p_close_data[b] = np.zeros(3, dtype=np.float32)
                    q_close_data[b] = np.array([1, 0, 0, 0], dtype=np.float32)
                self.save_dict["actions"].append(np.concatenate([p_close_data, q_close_data, np.ones((B, 1))], axis=1))

                # Follow object trajectory (skip already successful + skipped envs)
                if self.verbose:
                    print("[INFO] Following object trajectories...")
                skip_for_traj = env_skip | env_traj_success
                
                log_memory("Before follow_object_goals")
                jp, updated_skip_envs = self.follow_object_goals(jp, sample_step=5, visualize=True, skip_envs=skip_for_traj)
                
                # Update env_skip with any new failures from trajectory execution (e.g. drops)
                # We only care about new failures in active envs
                new_failures = updated_skip_envs & (~skip_for_traj)
                if np.any(new_failures):
                    if self.verbose:
                        print(f"[INFO] Updating env_skip with {np.sum(new_failures)} new failures from trajectory execution")
                    env_skip = env_skip | new_failures
                
                log_memory("After follow_object_goals")
                
                # Trajectory execution completed - verify which envs succeeded
                if self.verbose:
                    print("[INFO] Trajectory execution completed, verifying goal positions...")
                
                # Use the robust is_success check
                # Note: is_success returns a torch tensor of booleans
                success_mask_tensor = self.is_success(position_threshold=position_threshold, skip_envs=env_skip)
                per_env_success = success_mask_tensor.cpu().numpy()
                
                # Check results ONLY for envs that attempted this round
                if self.verbose:
                    print("\n" + "="*50)
                    print("PER-ENVIRONMENT TRAJECTORY RESULTS")
                    print("="*50)
                
                # Check success for each env
                for b in active_envs:  # Only check envs that attempted
                    env_traj_attempts[b] += 1
                    if per_env_success[b]:
                        env_traj_success[b] = True
                        if self.verbose:
                            print(f"Env {b}: SUCCESS")
                        
                        # Save THIS env's data from THIS successful attempt ONLY if not already saved
                        if b not in successful_env_data:
                            if self.verbose:
                                print(f"[INFO] Capturing successful trajectory data for Env {b}")
                            log_memory(f"Before capturing env {b} data")
                            successful_env_data[b] = {}
                            
                            # Capture data for this env without creating full stacked array
                            # This saves memory by only processing one timestep at a time
                            for key in self.save_dict.keys():
                                if isinstance(self.save_dict[key], list) and len(self.save_dict[key]) > 0:
                                    # Extract env b's data across timesteps without full stack
                                    # Process timestep-by-timestep to minimize memory
                                    env_data = []
                                    for timestep_data in self.save_dict[key]:
                                        if timestep_data.ndim >= 2:
                                            env_data.append(timestep_data[b])  # Extract just this env's data
                                    if env_data:
                                        successful_env_data[b][key] = np.array(env_data)  # Shape: (T, ...)
                                elif isinstance(self.save_dict[key], np.ndarray):
                                    # Already numpy (like grasp_pose_cam)
                                    arr = self.save_dict[key]
                                    if arr.ndim >= 2:
                                        # Slice already creates new array, no need for .copy()
                                        successful_env_data[b][key] = arr[:, b]
                            
                            log_memory(f"After capturing env {b} data")
                        else:
                            if self.verbose:
                                print(f"[INFO] Env {b} already has successful data from previous attempt - not re-capturing")
                    else:
                        if self.verbose:
                            print(f"Env {b}: FAILED (attempt {env_traj_attempts[b]}/{max_traj_retries})")
                        if env_traj_attempts[b] >= max_traj_retries:
                            env_skip[b] = True
                            if self.verbose:
                                print(f"  → Marking Env {b} as SKIPPED (exhausted {max_traj_retries} trajectory attempts)")
                
                if self.verbose:
                    print("="*50 + "\n")
                
                # Clear save_dict IMMEDIATELY after capturing all successful env data
                # This frees memory before the next iteration
                if self.verbose:
                    print("[DEBUG] Clearing save_dict after capturing successful envs")
                log_memory("Before clearing save_dict (after capture)")
                self.clear_data()
                import gc
                gc.collect()
                log_memory("After clearing save_dict and GC")
                
                # Check if all non-skipped envs succeeded
                remaining_active = np.where(~env_skip)[0]
                if len(remaining_active) == 0:
                    print("[ERR] All environments have been skipped")
                    return 0
                
                all_remaining_succeeded = np.all(env_traj_success[remaining_active])
                
                if all_remaining_succeeded:
                    if self.verbose:
                        print(f"[SUCCESS] All {len(remaining_active)} remaining environments succeeded!")
                        print(f"  Successful envs: {remaining_active.tolist()}")
                    if env_skip.sum() > 0 and self.verbose:
                        print(f"  Skipped envs: {np.where(env_skip)[0].tolist()}")
                    
                    # Save data for successful environments only
                    num_saved = self.save_data_selective(env_skip, demo_id=demo_id, env_offset=env_offset,
                                                        successful_env_data=successful_env_data)
                    
                    # Free successful_env_data after saving
                    successful_env_data.clear()
                    log_memory("After clearing successful_env_data")
                    
                    return num_saved
                else:
                    # Some envs still need retry
                    still_need_attempt = np.where((~env_skip) & (~env_traj_success))[0]
                    if self.verbose:
                        print(f"[INFO] {len(still_need_attempt)} environments still need attempts: {still_need_attempt.tolist()}")
                    
                    if traj_attempt < max_traj_retries - 1:
                        if self.verbose:
                            print(f"[INFO] Retrying trajectory execution (attempt {traj_attempt + 2}/{max_traj_retries})...")
                        # Note: save_dict already cleared above after capturing successful envs
                        continue
                    else:
                        # Last attempt - mark remaining failures as skipped
                        for b in still_need_attempt:
                            env_skip[b] = True
                            if self.verbose:
                                print(f"[INFO] Marking Env {b} as SKIPPED (failed all {max_traj_retries} trajectory attempts)")
                        
                        # Check if any succeeded
                        final_successful = np.where(env_traj_success)[0]
                        if len(final_successful) > 0:
                            if self.verbose:
                                print(f"\n[PARTIAL SUCCESS] Saving data for {len(final_successful)} successful environments: {final_successful.tolist()}")
                            
                            log_memory("Before save_data_selective (partial success)")
                            num_saved = self.save_data_selective(env_skip, demo_id=demo_id, env_offset=env_offset,
                                                                successful_env_data=successful_env_data)
                            log_memory("After save_data_selective (partial success)")
                            
                            successful_env_data.clear()
                            
                            return num_saved
                        else:
                            print("[ERR] All environments failed")
                            return 0
            except RuntimeError as e:
                error_msg = str(e)
                print(f"\n[ERROR] Runtime error during trajectory execution: {error_msg}")
                
                # Increment attempt counter for active envs
                for b in active_envs:
                    env_traj_attempts[b] += 1
                
                # Mark envs that exhausted attempts as skipped
                for b in active_envs:
                    if env_traj_attempts[b] >= max_traj_retries:
                        env_skip[b] = True
                        if self.verbose:
                            print(f"[INFO] Marking Env {b} as SKIPPED (exhausted attempts after error)")
                
                log_memory("After marking skipped envs in error handler")
                self.clear_data()
                
                log_memory("After error clear_data()")
                
                # Check if any active envs remain
                remaining_active = np.where(~env_skip)[0]
                if len(remaining_active) == 0:
                    print("[ERR] All environments have been skipped after errors")
                    return 0
                
                if traj_attempt < max_traj_retries - 1:
                    if self.verbose:
                        print(f"[INFO] Retrying with {len(remaining_active)} remaining environments...")
                    continue
                else:
                    if self.verbose:
                        print(f"[PARTIAL SUCCESS] Saving {len(remaining_active)} environments that didn't encounter errors")
                    
                    num_saved = self.save_data_selective(env_skip, demo_id=demo_id, env_offset=env_offset,
                                                        successful_env_data=successful_env_data)
                    successful_env_data.clear()
                    return num_saved
                    
            except Exception as e:
                print(f"[ERROR] Unexpected error: {type(e).__name__}: {e}")
                
                # Increment attempt counter for active envs
                for b in active_envs:
                    env_traj_attempts[b] += 1
                
                # Mark envs that exhausted attempts as skipped
                for b in active_envs:
                    if env_traj_attempts[b] >= max_traj_retries:
                        env_skip[b] = True
                        print(f"[INFO] Marking Env {b} as SKIPPED (exhausted attempts after error)")
                
                self.clear_data()
                
                # Check if any active envs remain
                remaining_active = np.where(~env_skip)[0]
                if len(remaining_active) == 0:
                    print("[ERR] All environments have been skipped after errors")
                    return 0
                
                if traj_attempt < max_traj_retries - 1:
                    print(f"[INFO] Retrying with {len(remaining_active)} remaining environments...")
                    continue
                else:
                    # Save whatever succeeded
                    final_successful = np.where(env_traj_success)[0]
                    if len(final_successful) > 0:
                        print(f"[PARTIAL SUCCESS] Saving {len(final_successful)} environments")
                        
                        num_saved = self.save_data_selective(env_skip, demo_id=demo_id, env_offset=env_offset,
                                                            successful_env_data=successful_env_data)
                        successful_env_data.clear()
                        return num_saved
                    else:
                        print("[ERR] All environments failed")
                        return 0
# ───────────────────────────────────────────────────────────────────────────── Entry Point ─────────────────────────────────────────────────────────────────────────────
def update_max_rss():
    global max_rss
    process = psutil.Process()
    current_rss = process.memory_info().rss
    max_rss = max(max_rss, current_rss)

def main():

    #add timer
    global start_time, end_time, max_rss
    start_time = time.time()
    max_rss = 0
    sim_cfgs = load_sim_parameters(BASE_DIR, args_cli.key)
    env, _ = make_env(
        cfgs=sim_cfgs, num_envs=args_cli.num_envs,
        device=args_cli.device,
        bg_simplify=False,
    )
    sim, scene = env.sim, env.scene

    my_sim = HeuristicManipulation(sim, scene, sim_cfgs=sim_cfgs)

    demo_root = (out_dir / img_folder / "demos").resolve()
    
    # If target_successes is specified, collect data until we reach that target
    if args_cli.target_successes is not None:
        target = args_cli.target_successes
        total_successes = 0
        env_offset = 0
        
        # Get or create a demo_id for this collection session
        demo_id = get_next_demo_id(demo_root)
        
        print(f"\n{'='*80}")
        print(f"TARGET SUCCESS MODE: Collecting {target} successful demonstrations")
        print(f"Saving all successful envs to demo_{demo_id}")
        print(f"{'='*80}\n")
        
        run_count = 0
        while total_successes < target:
            run_count += 1
            print(f"\n{'#'*80}")
            print(f"RUN {run_count}: Current total successes: {total_successes}/{target}")
            print(f"{'#'*80}\n")
            
            robot_pose = torch.tensor(sim_cfgs["robot_cfg"]["robot_pose"], dtype=torch.float32, device=my_sim.sim.device)
            my_sim.set_robot_pose(robot_pose)
            my_sim.reset()
            
            # CRITICAL: Clear save_dict before each run to prevent memory accumulation
            # NOTE: This only clears save_dict (trajectory data), NOT state variables like demo_id, env_offset, etc.
            if run_count > 1:  # Don't clear before first run (variables not initialized yet)
                print(f"[MEMORY] Clearing save_dict before run {run_count}...")
                my_sim.clear_data()
            
            # Log memory state
            try:
                import psutil
                import gc
                gc.collect()  # Force garbage collection
                process = psutil.Process()
                mem_info = process.memory_info()
                print(f"[MEMORY] Before run {run_count}: RSS={mem_info.rss / 1024**3:.2f} GB, VMS={mem_info.vms / 1024**3:.2f} GB")
                update_max_rss()
            except:
                pass
            
            # Run inference with shared demo_id and cumulative env_offset
            num_saved = my_sim.inference(demo_id=demo_id, env_offset=env_offset)
            
            total_successes += num_saved
            # CRITICAL: Increment env_offset by num_envs (not num_saved) to avoid ID collisions
            # If we have 4 envs and save 3, next run should start at env_4, not env_3
            env_offset += args_cli.num_envs
            
            # CRITICAL: Clear data after each inference run
            print(f"[MEMORY] Clearing save_dict after run {run_count}...")
            my_sim.clear_data()
            
            # Force garbage collection
            try:
                import gc
                gc.collect()
                print(f"[MEMORY] Garbage collection complete")
                update_max_rss()
            except:
                pass
            
            print(f"\n[RUN {run_count} COMPLETE] Saved {num_saved} successful environments")
            print(f"[PROGRESS] Total successes so far: {total_successes}/{target}\n")
            
            if total_successes >= target:
                print(f"\n{'='*80}")
                print(f"TARGET REACHED! Collected {total_successes} successful demonstrations")
                print(f"All data saved in demo_{demo_id}")
                print(f"{'='*80}\n")
                break
                
            if num_saved == 0:
                print(f"[WARN] No successful environments in run {run_count}")
                # Continue trying - don't give up
        
        # Timer for multi-run collection
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n[INFO] Total elapsed time: {elapsed_time / 60:.2f} minutes")
        print(f"[INFO] Maximum RSS usage: {max_rss / 1024**3:.2f} GB")
        
    else:
        # Original behavior: single trial per num_trials
        for _ in range(args_cli.num_trials):

            robot_pose = torch.tensor(sim_cfgs["robot_cfg"]["robot_pose"], dtype=torch.float32, device=my_sim.sim.device)
            my_sim.set_robot_pose(robot_pose)
            my_sim.demo_id = get_next_demo_id(demo_root)
            my_sim.reset()
            print(f"[INFO] start simulation demo_{my_sim.demo_id}")
            my_sim.inference()
            update_max_rss()
        
        # Timer for single-run mode
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n[INFO] Total elapsed time: {elapsed_time / 60:.2f} minutes")
        print(f"[INFO] Maximum RSS usage: {max_rss / 1024**3:.2f} GB")
    
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
    elapsed_time = end_time - start_time
    print(f"[INFO] Total elapsed time: {elapsed_time / 60:.2f} minutes")
    print(f"[INFO] Maximum RSS usage: {max_rss / 1024**3:.2f} GB")
    os.system("quit()")
    simulation_app.close()