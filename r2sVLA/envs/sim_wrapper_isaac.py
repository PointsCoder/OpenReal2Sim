"""
Policy evaluation wrapper for Isaac Lab simulation.
Provides interface for evaluating policies in batched simulation environments.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Union, Any, Tuple
import numpy as np
import torch
import argparse

from isaaclab.utils.math import subtract_frame_transforms, transform_points, combine_frame_transforms
# Note: AppLauncher should be initialized by the caller, not at module level
# This prevents duplicate initialization when importing this module

# Local imports
import sys
import os
file_path = Path(__file__).resolve()
sys.path.append(str(file_path.parent))
sys.path.append(str(file_path.parent.parent))

from sim_base_isaac import BaseSimulator
from sim_utils import pose_to_mat, mat_to_pose, pose_distance
from envs.cfgs.task_cfg import TaskCfg, TrajectoryCfg, SuccessMetric, SuccessMetricType, load_task_cfg
from envs.cfgs.eval_cfg import EvaluationConfig
from envs.cfgs.policy_interface import Action, BasePolicy
from envs.cfgs.randomizer import Randomizer
from envs.cfgs.randomizer_cfg import RandomizerConfig
from sim_base_isaac import BaseSimulator
from envs.make_env_isaac import make_env, get_prim_name_from_oid
# Constants
BASE_DIR = Path.cwd()

# ──────────────────────────── Policy Evaluation Wrapper ────────────────────────────
class PolicyEvaluationWrapper(BaseSimulator):
    """
    Policy evaluation wrapper for Isaac Lab simulation environments.
    
    Provides a clean interface for evaluating policies in batched simulation environments.
    Supports both joint position (qpos) and end-effector (ee) action types.
    
    Features:
      • Batch environment support with per-environment state tracking
      • Flexible action interface (qpos or ee control)
      • Episode termination checking
      • Policy integration via BasePolicy interface
      • Observation formatting for policy consumption
    """
    def __init__(self, sim, scene, task_cfg: TaskCfg, eval_cfg: EvaluationConfig, num_envs: int):
        super().__init__(
            sim=sim, scene=scene,
            out_dir=eval_cfg.video_save_dir,
            enable_motion_planning=True,
            set_physics_props=True, debug_level=0,
            task_cfg=task_cfg,
            eval_cfg=eval_cfg,
        )

        self.selected_object_id = task_cfg.manipulated_oid
        self._selected_object_id = str(self.selected_object_id)  # Store as string for mapping
        self._update_object_prim()  # Update object_prim based on selected_object_id
        self.record = eval_cfg.record_video  # Store whether to record data
        assert self.record, "Record must be True for evaluation"
        #self.traj_cfg_list = traj_cfg_list
       
        self.task_type = task_cfg.task_type
       
        # Evaluation configuration
        self.eval_cfg = eval_cfg
        
        # Store task_cfg for trajectory loading
        self.task_cfg = task_cfg
        
        # Initialize randomizer (lazy initialization)
        self.randomizer: Optional[Randomizer] = None
        self.randomizer_cfg: Optional[RandomizerConfig] = None
        
        # Trajectory tracking
        self.traj_cfg_list: Optional[List[TrajectoryCfg]] = None
        self.current_trajectory_idx = 0  # For verified randomization mode
        
        # Success metric list - per environment [B]
        self.success_metric_list: List[Optional[SuccessMetric]] = [None] * num_envs
        
        # Episode tracking - per environment [B]
        device = self.robot.device
        self.step_count = torch.zeros(self.num_envs, dtype=torch.long, device=device)  # [B]
        self.episode_done = torch.zeros(self.num_envs, dtype=torch.bool, device=device)  # [B]
        self.eval_success = torch.zeros(self.num_envs, dtype=torch.bool, device=device)  # [B]
        self._is_success = torch.zeros(self.num_envs, dtype=torch.bool, device=device)  # [B] - Current success state
        
        # Grasping success tracking - per environment [B]
        self.initial_object_height = torch.zeros(self.num_envs, dtype=torch.float32, device=device)  # [B] - Initial object height (z coordinate)
        self.grasping_success_achieved = torch.zeros(self.num_envs, dtype=torch.bool, device=device)  # [B] - Whether grasping success has been achieved at least once
        
        # Gripper state tracking - per environment [B]
        # True = open, False = closed (updated from actions)
        self.gripper_state_list = torch.ones(self.num_envs, dtype=torch.bool, device=device)  # [B] - Current gripper state (True=open, False=closed)
        
        # Policy (will be set externally)
        self.policy: Optional[BasePolicy] = None

    def _compute_poses_from_traj_cfg(self, traj_cfg_list: List[TrajectoryCfg]):
        """
        Extract poses and trajectories from a list of TrajectoryCfg objects.
        Reference: sim_randomize_rollout.py compute_poses_from_traj_cfg
        
        Args:
            traj_cfg_list: List of TrajectoryCfg objects
            
        Returns:
            robot_poses_list: List of robot poses [7] for each trajectory
            object_poses_dict: Dict mapping oid -> list of (pos, quat) tuples
            object_trajectory_list: List of object trajectories
            final_gripper_state_list: List of final gripper states
            pregrasp_pose_list: List of pregrasp poses
            grasp_pose_list: List of grasp poses
            end_pose_list: List of end poses from success_metric
            success_metric_list: List of SuccessMetric objects
        """
        robot_poses_list = []
        object_poses_dict = {}  # {oid: [(pos, quat), ...]}
        object_trajectory_list = []
        final_gripper_state_list = []
        pregrasp_pose_list = []
        grasp_pose_list = []
        end_pose_list = []
        success_metric_list = []

        for traj_cfg in traj_cfg_list:
            robot_poses_list.append(traj_cfg.robot_pose)
            
            # Extract object poses: traj_cfg.object_poses is a dict mapping oid -> pose
            for oid in traj_cfg.object_poses.keys():
                pose = traj_cfg.object_poses[oid]
                oid_str = str(oid)
                if oid_str not in object_poses_dict:
                    object_poses_dict[oid_str] = []
                object_poses_dict[oid_str].append(np.array(pose, dtype=np.float32))
            
            # Extract object trajectory
            traj = []
            for i in range(len(traj_cfg.object_trajectory)):
                mat = pose_to_mat(traj_cfg.object_trajectory[i][:3], traj_cfg.object_trajectory[i][3:7])
                traj.append(mat)
            object_trajectory_list.append(np.array(traj, dtype=np.float32))
            
            final_gripper_state_list.append(traj_cfg.final_gripper_close)
            pregrasp_pose_list.append(np.array(traj_cfg.pregrasp_pose, dtype=np.float32) if traj_cfg.pregrasp_pose is not None else None)
            grasp_pose_list.append(np.array(traj_cfg.grasp_pose, dtype=np.float32) if traj_cfg.grasp_pose is not None else None)
            
            # Extract success metric
            success_metric_list.append(traj_cfg.success_metric)
            if traj_cfg.success_metric.end_pose is not None:
                end_pose_list.append(np.array(traj_cfg.success_metric.end_pose, dtype=np.float32))
            else:
                end_pose_list.append(None)
      
        return robot_poses_list, object_poses_dict, object_trajectory_list, final_gripper_state_list, pregrasp_pose_list, grasp_pose_list, end_pose_list, success_metric_list

    def reset(self, env_ids=None, use_randomization: bool = False, use_verified_randomization: bool = False):
        super().reset(env_ids)
        device = self.object_prim.device
        if env_ids is None:
            env_ids_t = self._all_env_ids.to(device)  # (B,)
        else:
            env_ids_t = torch.as_tensor(env_ids, device=device, dtype=torch.long).view(-1)  # (M,)
        M = int(env_ids_t.shape[0])

        # ──────────── Load trajectory configurations ────────────
        if not use_randomization:
            print("[INFO] Using reference trajectory")
            self.traj_cfg_list = self.task_cfg.reference_trajectory
        else:
            if use_verified_randomization:
                # Mode 1: Load from generated_trajectories (verified randomization)
                if self.task_cfg.generated_trajectories is None or len(self.task_cfg.generated_trajectories) == 0:
                    raise ValueError("No generated_trajectories available in task_cfg for verified randomization mode")
                
                # Load the last num_envs trajectories from generated_trajectories
                num_available = len(self.task_cfg.generated_trajectories)
                start_idx = max(0, num_available - self.num_envs)
                end_idx = num_available
                
                # If we need more trajectories than available, cycle through them
                if end_idx - start_idx < M:
                    # Repeat trajectories if needed
                    traj_cfg_list = []
                    for i in range(M):
                        idx = (start_idx + i) % num_available
                        traj_cfg_list.append(self.task_cfg.generated_trajectories[idx])
                else:
                    traj_cfg_list = self.task_cfg.generated_trajectories[start_idx:end_idx][:M]
                
                self.traj_cfg_list = traj_cfg_list
            else:
                # Mode 2: Generate new trajectories using randomizer
                if self.randomizer is None:
                    from envs.cfgs.randomizer import Randomizer
                    self.randomizer = Randomizer(self.task_cfg)
                
                # Get randomizer config (use default if not set)
                if self.randomizer_cfg is None:
                    # Try to load from config file or use defaults
                    try:
                        from envs.cfgs.randomizer_cfg import RandomizerConfig
                        config_path = BASE_DIR / "config" / "randomizer_cfg.yaml"
                        if config_path.exists():
                            self.randomizer_cfg = RandomizerConfig.from_yaml(config_path, task_name=self.task_cfg.task_key)
                        else:
                            self.randomizer_cfg = RandomizerConfig()  # Use defaults
                    except Exception as e:
                        print(f"[WARN] Failed to load randomizer config, using defaults: {e}")
                        from envs.cfgs.randomizer_cfg import RandomizerConfig
                        self.randomizer_cfg = RandomizerConfig()
                
                # Generate randomized trajectories
                # Calculate traj_randomize_num to get at least M trajectories
                total_per_traj = self.randomizer_cfg.scene_randomize_num * self.randomizer_cfg.robot_pose_randomize_num
                traj_randomize_num = max(1, (M + total_per_traj - 1) // total_per_traj)  # Ceiling division
                
                random_traj_cfg_list = self.randomizer.generate_randomized_scene_cfg(
                    grid_dist=self.randomizer_cfg.grid_dist,
                    grid_num=self.randomizer_cfg.grid_num,
                    angle_random_range=self.randomizer_cfg.angle_random_range,
                    angle_random_num=self.randomizer_cfg.angle_random_num,
                    traj_randomize_num=traj_randomize_num,
                    scene_randomize_num=self.randomizer_cfg.scene_randomize_num,
                    robot_pose_randomize_range=self.randomizer_cfg.robot_pose_randomize_range,
                    robot_pose_randomize_angle=self.randomizer_cfg.robot_pose_randomize_angle,
                    robot_pose_randomize_num=self.randomizer_cfg.robot_pose_randomize_num,
                    fix_end_pose=self.randomizer_cfg.fix_end_pose,
                )
                # Take first M trajectories
                self.traj_cfg_list = random_traj_cfg_list[:M]
        
        # ──────────── Compute poses and update success metrics ────────────
        (self.robot_poses_list, self.object_poses_dict, self.object_trajectory_list,
         self.final_gripper_state_list, self.pregrasp_pose_list, self.grasp_pose_list,
         self.end_pose_list, success_metrics) = self._compute_poses_from_traj_cfg(self.traj_cfg_list)
        
        # Update success_metric_list for the environments being reset
        for i, env_idx in enumerate(env_ids_t.cpu().numpy()):
            if i < len(success_metrics):
                self.success_metric_list[env_idx] = success_metrics[i]

        # ──────────── Set object poses ────────────
        env_origins = self.scene.env_origins.to(device)[env_ids_t]  # (M,3)
        
        # Set poses for all objects from object_poses_dict
        for oid in self.object_poses_dict.keys():
            # Get prim name from oid
            prim_name = get_prim_name_from_oid(str(oid))
            
            object_prim = self.scene[prim_name]
            
            # Get pose for this object
            if len(self.object_poses_dict[oid]) == 0:
                continue
            
            # Extract poses for the environments being reset
            poses_array = np.array(self.object_poses_dict[oid], dtype=np.float32)
            if poses_array.shape[0] < M:
                # Pad with last pose if needed
                last_pose = poses_array[-1]
                poses_array = np.vstack([poses_array, np.tile(last_pose, (M - poses_array.shape[0], 1))])
            
            pos = poses_array[env_ids_t.cpu().numpy(), :3]
            quat = poses_array[env_ids_t.cpu().numpy(), 3:7]
            
            object_pose = torch.zeros((M, 7), device=device, dtype=torch.float32)
            object_pose[:, :3] = env_origins + torch.tensor(pos, dtype=torch.float32, device=device)
            object_pose[:, 3:7] = torch.tensor(quat, dtype=torch.float32, device=device)  # wxyz
            
            object_prim.write_root_pose_to_sim(object_pose, env_ids=env_ids_t)
            object_prim.write_root_velocity_to_sim(
                torch.zeros((M, 6), device=device, dtype=torch.float32), env_ids=env_ids_t
            )
            object_prim.write_data_to_sim()
        
        # ──────────── Set robot poses ────────────
        rp_local = np.array(self.robot_poses_list, dtype=np.float32)
        env_origins_robot = self.scene.env_origins.to(device)[env_ids_t]
        import copy
        robot_pose_world = copy.deepcopy(rp_local)
        
        # Ensure we have enough robot poses
        if robot_pose_world.shape[0] < M:
            last_pose = robot_pose_world[-1]
            robot_pose_world = np.vstack([robot_pose_world, np.tile(last_pose, (M - robot_pose_world.shape[0], 1))])
        
        robot_pose_world[:, :3] = env_origins_robot.cpu().numpy() + robot_pose_world[env_ids_t.cpu().numpy(), :3]
        self.robot.write_root_pose_to_sim(torch.tensor(robot_pose_world, dtype=torch.float32, device=device), env_ids=env_ids_t)
        self.robot.write_root_velocity_to_sim(
            torch.zeros((M, 6), device=device, dtype=torch.float32), env_ids=env_ids_t
        )

        joint_pos = self.robot.data.default_joint_pos.to(self.robot.device)[env_ids_t]  # (M,7)
        joint_vel = self.robot.data.default_joint_vel.to(self.robot.device)[env_ids_t]  # (M,7)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_t)
        self.robot.write_data_to_sim()

        self.clear_data()

        # Reset episode tracking for specified environments
        device = self.object_prim.device
        if env_ids is None:
            env_ids_t = self._all_env_ids.to(device)
        else:
            env_ids_t = torch.as_tensor(env_ids, device=device, dtype=torch.long).view(-1)
        
        self.step_count[env_ids_t] = 0
        self.episode_done[env_ids_t] = False
        self.eval_success[env_ids_t] = False
        self._is_success[env_ids_t] = False
        
        # Record initial object height for grasping success check
        # Need to step simulation once to get accurate positions after reset
        initial_object_pos = self.object_prim.data.root_com_pos_w[:, :3]  # [B, 3]
        initial_object_pos -= self.scene.env_origins.to(device)[env_ids_t]
        print("current robot pose", self.robot.data.root_state_w[:, :7])
        print("scene.env_origins", self.scene.env_origins.to(device)[env_ids_t])
        print("initial_object_pos", initial_object_pos)
        self.initial_object_height[env_ids_t] = initial_object_pos[env_ids_t, 2]  # Store z coordinate (height)
        
        # Reset grasping success tracking
        self.grasping_success_achieved[env_ids_t] = False
        
        # Reset gripper state (default to open)
        self.gripper_state_list[env_ids_t] = True
        
        # Reset policy if available
        if self.policy is not None:
            self.policy.reset()

    # ---------- Policy Evaluation Functions ----------
    def get_obs(self) -> Dict[str, Any]:
        """
        Get observation from environment.
        Returns a dictionary with observation data compatible with policy interface.
        
        Returns:
            Dictionary containing:
                - 'rgb': RGB image [B, H, W, 3]
                - 'depth': Depth image [B, H, W]
                - 'joint_pos': Joint positions [B, n_joints]
                - 'joint_vel': Joint velocities [B, n_joints]
                - 'ee_pose': End-effector pose [B, 7] (pos + quat)
                - 'gripper_pos': Gripper position [B, 1]
                - Other observation keys as needed
        """
        # Use actual gripper state for gripper_cmd (use first env as representative, or could use per-env logic)
        # FIXME: Currently we DO NOT actually ALLOW PARALLELISM IN EVALUATION.
        obs = self.get_observation(gripper_open=self.gripper_state_list[0].item() if self.num_envs > 0 else True)
        
        # Convert to policy-friendly format
        observation = {
            'rgb': obs['rgb'],  # [B, H, W, 3]
            'depth': obs['depth'],  # [B, H, W]
            'joint_pos': obs['joint_pos'],  # [B, n_joints]
            'joint_vel': obs.get('joint_vel', torch.zeros_like(obs['joint_pos'])),  # [B, n_joints]
            'ee_pose': obs['ee_pose_w'],  # [B, 7] (pos + quat)
            'gripper_pos': obs['gripper_pos'],  # [B, 1] or [B, 3]
            'gripper_cmd': obs.get('gripper_cmd', torch.zeros((self.num_envs, 1), device=self.robot.device)),
        }
        
        return observation
    
    def take_action(self, action: Action, record: bool = True) -> Dict[str, Any]:
        """
        Execute an action in the environment.
        
        Args:
            action: Action object (either qpos or ee type)
            record: Whether to record data
        
        Returns:
            Updated observation dictionary
        """
        device = self.robot.device
        
        # Find active (not done) environments
        active_mask = ~self.episode_done  # [B]
        active_env_ids = self._all_env_ids[active_mask].to(device)  # [M] where M <= B
        
        # If all environments are done, return observation
        if len(active_env_ids) == 0:
            return self.get_obs()
        
        # Check step limit for active environments
        step_limit_reached = self.step_count[active_env_ids] >= self.eval_cfg.max_steps
        self.episode_done[active_env_ids[step_limit_reached]] = True
        
        # Update active mask after step limit check
        active_mask = ~self.episode_done
        active_env_ids = self._all_env_ids[active_mask].to(device)
        
        if len(active_env_ids) == 0:
            return self.get_obs()
        
        # Extract gripper state from action (for updating gripper_state_list after action execution)
        gripper_open_from_action = action.gripper_open
        if isinstance(gripper_open_from_action, torch.Tensor):
            if gripper_open_from_action.ndim == 0:
                # Scalar: apply to all active environments
                gripper_open_value = gripper_open_from_action.item()
                gripper_open_tensor = torch.full((len(active_env_ids),), gripper_open_value, dtype=torch.bool, device=device)
            else:
                # Per-env tensor
                if gripper_open_from_action.shape[0] == len(active_env_ids):
                    gripper_open_tensor = gripper_open_from_action.to(device).bool()
                elif gripper_open_from_action.shape[0] == self.num_envs:
                    gripper_open_tensor = gripper_open_from_action[active_env_ids].to(device).bool()
                else:
                    gripper_open_value = gripper_open_from_action[0].item() if gripper_open_from_action.shape[0] > 0 else True
                    gripper_open_tensor = torch.full((len(active_env_ids),), gripper_open_value, dtype=torch.bool, device=device)
        else:
            # Scalar (bool or float): apply to all active environments
            gripper_open_value = bool(gripper_open_from_action) if not isinstance(gripper_open_from_action, bool) else gripper_open_from_action
            gripper_open_tensor = torch.full((len(active_env_ids),), gripper_open_value, dtype=torch.bool, device=device)
        
        # Execute action based on type
        if action.action_type == 'qpos':
            # Joint position control
            joint_pos_des = action.qpos.to(self.robot.device)
            
            # Ensure correct shape: [B, n_joints]
            if joint_pos_des.ndim == 1:
                joint_pos_des = joint_pos_des.unsqueeze(0).repeat(self.num_envs, 1)
            
            # Use scalar gripper_open for apply_actions (it handles per-env internally)
            if isinstance(gripper_open_from_action, torch.Tensor) and gripper_open_from_action.ndim > 0:
                gripper_open_scalar = gripper_open_from_action[0].item() if gripper_open_from_action.shape[0] > 0 else True
            else:
                gripper_open_scalar = gripper_open_from_action.item() if isinstance(gripper_open_from_action, torch.Tensor) else gripper_open_from_action
            
            self.apply_actions(joint_pos_des, gripper_open=gripper_open_scalar)
            
        elif action.action_type == 'ee_direct':
            # End-effector control using motion planning
            ee_pose = action.ee_pose.to(self.robot.device)
            
            # Ensure correct shape: [B, 7]
            if ee_pose.ndim == 1:
                ee_pose = ee_pose.unsqueeze(0).repeat(self.num_envs, 1)
            
            # Split into position and quaternion
            position = ee_pose[:, :3]
            quaternion = ee_pose[:, 3:7]  # [B, 4] (wxyz)
            
            # Handle gripper
            gripper_open = action.gripper_open
            if isinstance(gripper_open, torch.Tensor):
                if gripper_open.ndim == 0:
                    gripper_open = gripper_open.item()
                else:
                    gripper_open = gripper_open[0].item() if gripper_open.shape[0] > 0 else True
            
            # Use motion planning to move to target pose
            self.move_to_motion_planning(
                position=position,
                quaternion=quaternion,
                gripper_open=gripper_open,
                record=record
            )
            self.wait(gripper_open=gripper_open, steps=3)
        
        elif action.action_type == 'ee_l':
            # End-effector control using motion planning
            ee_pose = action.ee_pose.to(self.robot.device)
            
            # Ensure correct shape: [B, 7]
            if ee_pose.ndim == 1:
                ee_pose = ee_pose.unsqueeze(0).repeat(self.num_envs, 1)
            
            # Split into position and quaternion
            position = ee_pose[:, :3] + self.scene.env_origins.to(self.robot.device)[active_env_ids]
            quaternion = ee_pose[:, 3:7]  # [B, 4] (wxyz)
            
            # Handle gripper
            gripper_open = action.gripper_open
            if isinstance(gripper_open, torch.Tensor):
                if gripper_open.ndim == 0:
                    gripper_open = gripper_open.item()
                else:
                    gripper_open = gripper_open[0].item() if gripper_open.shape[0] > 0 else True
            
            robot_root_pose_w = self.robot.data.root_state_w[:, :7]
            ee_pos_l, ee_quat_l = subtract_frame_transforms(robot_root_pose_w[:, :3], robot_root_pose_w[:, 3:7], position, quaternion)
            # Use motion planning to move to target pose
            # self.move_to_motion_planning(
            #     position=ee_pos_l,
            #     quaternion=ee_quat_l,
            #     gripper_open=gripper_open,
            #     record=record
            # )
            self.move_to_ik(
                position=ee_pos_l,
                quaternion=ee_quat_l,
                steps=3,
                gripper_open=gripper_open,
                record=record
            )
            #self.wait(gripper_open=gripper_open, steps=3)
    
        elif action.action_type == 'ee_cam':
            # End-effector control in camera frame - convert to world frame first
            ee_pose_cam = action.ee_pose.to(self.robot.device)
            
            # Ensure correct shape: [B, 7]
            if ee_pose_cam.ndim == 1:
                ee_pose_cam = ee_pose_cam.unsqueeze(0).repeat(self.num_envs, 1)
            
            # Get camera pose in world frame
            cam_pos_w = self.camera.data.pos_w  # [B, 3]
            cam_quat_w = self.camera.data.quat_w_ros  # [B, 4] (wxyz)
            
            cam_pos_w -= self.scene.env_origins.to(self.robot.device)[active_env_ids]
            # Extract camera frame pose
            ee_pos_cam = ee_pose_cam[:, :3]  # [B, 3]
            ee_quat_cam = ee_pose_cam[:, 3:7]  # [B, 4] (wxyz)
            
            # Convert from camera frame to world frame
            # Use combine_frame_transforms to batch transform both position and quaternion
            # t01, q01: camera pose in world frame (frame 1 w.r.t. frame 0)
            # t12, q12: ee pose in camera frame (frame 2 w.r.t. frame 1)
            # Returns: ee pose in world frame (frame 2 w.r.t. frame 0)
            ee_pos_w, ee_quat_w = combine_frame_transforms(
                t01=cam_pos_w,  # [B, 3] camera position in world
                q01=cam_quat_w,  # [B, 4] camera quaternion in world (wxyz)
                t12=ee_pos_cam,  # [B, 3] ee position in camera
                q12=ee_quat_cam  # [B, 4] ee quaternion in camera (wxyz)
            )  # Returns: [B, 3], [B, 4]
            ee_pos_w += self.scene.env_origins.to(self.robot.device)[active_env_ids]
            
            # Handle gripper
            gripper_open = action.gripper_open
            if isinstance(gripper_open, torch.Tensor):
                if gripper_open.ndim == 0:
                    gripper_open = gripper_open.item()
                else:
                    gripper_open = gripper_open[0].item() if gripper_open.shape[0] > 0 else True
            
            robot_root_pose_w = self.robot.data.root_state_w[:, :7]
            ee_pos_l, ee_quat_l = subtract_frame_transforms(robot_root_pose_w[:, :3], robot_root_pose_w[:, 3:7], ee_pos_w, ee_quat_w)
            self.move_to_motion_planning(
                position=ee_pos_l,
                quaternion=ee_quat_l,
                gripper_open=gripper_open,
                record=record
            )
        else:
            
            raise ValueError(f"Unknown action type: {action.action_type}")
        
        # Update gripper_state_list from action (after action execution)
        self.gripper_state_list[active_env_ids] = gripper_open_tensor
        
        # Update step count for active environments (task step counter)
        # Each take_action() call corresponds to one task step
        self.step_count[active_env_ids] += 1
        
        # Update success state for active environments
        self.is_success(env_ids=active_env_ids)
        
        # Check termination for active environments
        termination_mask = self.check_termination(env_ids=active_env_ids)
        self.episode_done[active_env_ids] = termination_mask
        
        # Get updated observation
        obs = self.get_obs()
        
        # Update policy observation if needed
        if self.policy is not None:
            self.policy.update_obs(obs)
        
        return obs
    
    def check_termination(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Check if episode should terminate for specified environments.
        This is a placeholder - user will implement their own termination logic.
        
        Args:
            env_ids: Optional tensor of environment IDs to check. If None, checks all environments.
        
        Returns:
            Boolean tensor [M] indicating which environments should terminate
        """
        device = self.robot.device
        if env_ids is None:
            env_ids_t = self._all_env_ids.to(device)
        else:
            env_ids_t = torch.as_tensor(env_ids, device=device, dtype=torch.long).view(-1)
        
        M = len(env_ids_t)
        termination = torch.zeros(M, dtype=torch.bool, device=device)
        
        # Check step limit
        step_limit_reached = self.step_count[env_ids_t] >= self.eval_cfg.max_steps
        termination |= step_limit_reached
        
        termination |= self._is_success[env_ids_t]
        
        return termination
    
    def is_success(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Check success based on eval_cfg.success_keys.
        Combines different success metrics based on configuration.
        
        Args:
            env_ids: Optional tensor of environment IDs to check. If None, checks all environments.
        
        Returns:
            Boolean tensor [M] or [B] indicating which environments have achieved success
        """
        device = self.object_prim.device
        if env_ids is None:
            env_ids_t = self._all_env_ids.to(device)
        else:
            env_ids_t = torch.as_tensor(env_ids, device=device, dtype=torch.long).view(-1)
        
        M = len(env_ids_t)
        
        # If no success keys specified, return False for all
        if not self.eval_cfg.success_keys:
            self._is_success[env_ids_t] = False
            return self._is_success[env_ids_t]
        
        # Initialize success mask (all True, will be ANDed with each check)
        success_mask = torch.ones(M, dtype=torch.bool, device=device)
        
        # Check each success key
        for key in self.eval_cfg.success_keys:
            if key == "grasping":
                success_mask = success_mask & self.is_grasping_success(env_ids=env_ids_t)
            elif key == "strict":
                success_mask = success_mask & self.is_strict_success(env_ids=env_ids_t)
            elif key == "metric":
                success_mask = success_mask & self.is_metric_success(env_ids=env_ids_t)
            else:
                print(f"[WARN] Unknown success key: {key}, ignoring")
        
        # Update internal success state
        self._is_success[env_ids_t] = success_mask
        
        return success_mask
    
    def is_grasping_success(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = self.object_prim.device
        if env_ids is None:
            env_ids_t = self._all_env_ids.to(device)
        else:
            env_ids_t = torch.as_tensor(env_ids, device=device, dtype=torch.long).view(-1)
        
        # If already achieved for these environments, return True
        already_achieved = self.grasping_success_achieved[env_ids_t]
        if torch.all(already_achieved):
            return already_achieved
        
        # Check gripper is closed using gripper_state_list (True=open, False=closed)
        is_gripper_closed = ~self.gripper_state_list[env_ids_t]  # [M]
        
        # Get current and initial object heights
        current_object_pos = self.object_prim.data.root_com_pos_w[:, :3]  # [B, 3]

        current_object_height = current_object_pos[env_ids_t, 2]  # [M] - z coordinate
        initial_height = self.initial_object_height[env_ids_t]  # [M]
        
        # Check if object is lifted above threshold
        height_lifted = current_object_height - initial_height  # [M]
        is_object_lifted = height_lifted >= self.eval_cfg.lift_height_threshold
        
        # Grasping success: gripper closed AND object lifted
        current_success = is_gripper_closed & is_object_lifted  # [M]
        
        # Update tracking: once achieved, it stays True
        self.grasping_success_achieved[env_ids_t] = self.grasping_success_achieved[env_ids_t] | current_success
        
        return self.grasping_success_achieved[env_ids_t]
    

    def _check_gripper_state_match(self, env_ids_t: torch.Tensor) -> torch.Tensor:
        """Check if gripper state matches final_gripper_close requirement for given environments."""
        device = env_ids_t.device
        gripper_match_list = []
        
        for env_idx in env_ids_t.cpu().numpy():
            success_metric = self.success_metric_list[env_idx]
            
            if success_metric is None:
                gripper_match_list.append(torch.tensor(False, device=device))
                continue
            
            # Get current gripper state (True=open, False=closed)
            current_gripper_open = self.gripper_state_list[env_idx]
            
            # final_gripper_close=True means should be closed (gripper_open=False)
            # final_gripper_close=False means should be open (gripper_open=True)
            required_gripper_open = not success_metric.final_gripper_close
            gripper_match = (current_gripper_open == required_gripper_open)
            gripper_match_list.append(torch.tensor(gripper_match, device=device))
        
        return torch.stack(gripper_match_list)  # [M]
    
    def _compute_pose_distance_to_end_pose(self, env_ids_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute pose distance between current object pose and end_pose from SuccessMetric.
        
        Returns:
            trans_dist: Translation distance [M]
            angle_dist: Rotation angle distance [M]
            valid_mask: Boolean mask indicating which envs have valid end_pose [M]
        """
        device = env_ids_t.device
        
        # Get current object pose in world frame and convert to local frame
        obj_w = self.object_prim.data.root_state_w[:, :7]  # [B, 7]
        origins = self.scene.env_origins.to(device)  # [B, 3]
        obj_local = obj_w.clone()
        obj_local[:, :3] = obj_w[:, :3] - origins  # [B, 7] in local frame
        
        trans_dist_list = []
        angle_list = []
        valid_mask_list = []
        
        for env_idx in env_ids_t.cpu().numpy():
            success_metric = self.success_metric_list[env_idx]
            
            if success_metric is None or success_metric.end_pose is None:
                trans_dist_list.append(torch.tensor(float('inf'), device=device))
                angle_list.append(torch.tensor(float('inf'), device=device))
                valid_mask_list.append(False)
                continue
            
            valid_mask_list.append(True)
            
            # Get current and target poses
            obj_pose_local = obj_local[env_idx]  # [7]
            end_pose = np.array(success_metric.end_pose, dtype=np.float32)  # [7]
            
            # Convert to 4x4 transformation matrices
            obj_mat = pose_to_mat(obj_pose_local[:3].cpu().numpy(), obj_pose_local[3:7].cpu().numpy())
            end_mat = pose_to_mat(end_pose[:3], end_pose[3:7])
            
            # Compute pose distance
            obj_mat_t = torch.tensor(obj_mat, dtype=torch.float32, device=device)
            end_mat_t = torch.tensor(end_mat, dtype=torch.float32, device=device)
            trans_dist, angle = pose_distance(obj_mat_t.unsqueeze(0), end_mat_t.unsqueeze(0))
            
            trans_dist_list.append(trans_dist.squeeze(0))
            angle_list.append(angle.squeeze(0))
        
        trans_dist = torch.stack(trans_dist_list)  # [M]
        angle_dist = torch.stack(angle_list)  # [M]
        valid_mask = torch.tensor(valid_mask_list, dtype=torch.bool, device=device)  # [M]
        
        return trans_dist, angle_dist, valid_mask
    
    def is_strict_success(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Check if object pose is within threshold distance of the end_pose from SuccessMetric.
        Compares current object pose with the target end_pose using pose distance (translation + rotation).
        
        Args:
            env_ids: Optional tensor of environment IDs to check. If None, checks all environments.
        
        Returns:
            Boolean tensor [M] or [B] indicating which environments have achieved strict success
        """
        device = self.object_prim.device
        if env_ids is None:
            env_ids_t = self._all_env_ids.to(device)
        else:
            env_ids_t = torch.as_tensor(env_ids, device=device, dtype=torch.long).view(-1)
        
        # Compute pose distances
        trans_dist, angle_dist, valid_mask = self._compute_pose_distance_to_end_pose(env_ids_t)
        
        # Check pose distance thresholds
        trans_success = trans_dist <= self.eval_cfg.pose_dist_threshold
        angle_success = angle_dist <= self.eval_cfg.angle_dist_threshold
        
        # Check gripper state matches requirement
        gripper_state_success = self._check_gripper_state_match(env_ids_t)
        
        # All conditions must be satisfied
        is_success = trans_success & angle_success & gripper_state_success & valid_mask
        
        return is_success
    
    def is_metric_success(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Check success based on SuccessMetric type for each environment.
        Uses different criteria based on success_metric_type:
        - SIMPLE_LIFT: Check lift_height threshold
        - TARGET_POINT: Check end_pose distance (same as is_strict_success)
        - TARGET_PLANE: Check ground_value (object height above ground)
        
        Args:
            env_ids: Optional tensor of environment IDs to check. If None, checks all environments.
        
        Returns:
            Boolean tensor [M] or [B] indicating which environments have achieved metric success
        """
        device = self.object_prim.device
        if env_ids is None:
            env_ids_t = self._all_env_ids.to(device)
        else:
            env_ids_t = torch.as_tensor(env_ids, device=device, dtype=torch.long).view(-1)
        
        success_list = []
        
        for env_idx in env_ids_t.cpu().numpy():
            success_metric = self.success_metric_list[env_idx]
            
            if success_metric is None:
                success_list.append(torch.tensor(False, device=device))
                continue
            
            # Check gripper state first (required for all metric types)
            current_gripper_open = self.gripper_state_list[env_idx]
            required_gripper_open = not success_metric.final_gripper_close
            gripper_match = (current_gripper_open == required_gripper_open)
            
            if not gripper_match:
                success_list.append(torch.tensor(False, device=device))
                continue
            
            # Check based on metric type
            if success_metric.success_metric_type == SuccessMetricType.SIMPLE_LIFT:
                # Check lift height
                current_object_pos = self.object_prim.data.root_com_pos_w[env_idx, :3]  # [3]
                current_height = current_object_pos[2]
                initial_height = self.initial_object_height[env_idx]
                height_lifted = current_height - initial_height
                
                lift_threshold = success_metric.lift_height if success_metric.lift_height is not None else self.eval_cfg.lift_height_threshold
                is_success = height_lifted >= lift_threshold
                
            elif success_metric.success_metric_type == SuccessMetricType.TARGET_POINT:
                # Check end_pose distance (same as is_strict_success)
                env_idx_t = torch.tensor([env_idx], device=device, dtype=torch.long)
                trans_dist, angle_dist, valid_mask = self._compute_pose_distance_to_end_pose(env_idx_t)
                
                if not valid_mask[0] or success_metric.end_pose is None:
                    is_success = False
                else:
                    trans_success = trans_dist[0] <= self.eval_cfg.pose_dist_threshold
                    angle_success = angle_dist[0] <= self.eval_cfg.angle_dist_threshold
                    is_success = trans_success & angle_success
                    
            elif success_metric.success_metric_type == SuccessMetricType.TARGET_PLANE:
                # Check ground value (object height above ground plane)
                current_object_pos = self.object_prim.data.root_com_pos_w[env_idx, :3]  # [3]
                current_height = current_object_pos[2]
                
                if success_metric.ground_value is not None:
                    is_success = current_height >= success_metric.ground_value
                else:
                    # Fallback: check if lifted above initial height
                    initial_height = self.initial_object_height[env_idx]
                    height_lifted = current_height - initial_height
                    is_success = height_lifted >= self.eval_cfg.lift_height_threshold
            else:
                # Unknown metric type
                is_success = False
            
            success_list.append(torch.tensor(is_success, device=device))
        
        return torch.stack(success_list)  # [M]
    

    def set_policy(self, policy: BasePolicy) -> None:
        """Set the policy for evaluation."""
        self.policy = policy
    
    def evaluate_episode(self, reset_env: bool = True, env_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Run a complete evaluation episode.
        
        Args:
            reset_env: Whether to reset the environment at the start
            env_ids: Optional list of environment IDs to evaluate. If None, evaluates all.
        
        Returns:
            Dictionary with episode results:
                - 'success': bool tensor [B] or [M]
                - 'steps': int tensor [B] or [M]
                - 'episode_done': bool tensor [B] or [M]
        """
        if reset_env:
            self.reset(env_ids=env_ids)
        
        if self.policy is None:
            raise ValueError("Policy not set. Call set_policy() first.")
        
        # Reset policy
        self.policy.reset()
        
        # Note: physics_step_counter removed - step counting is now handled by base class
        
        # Convert env_ids to tensor if provided
        device = self.robot.device
        if env_ids is not None:
            env_ids_t = torch.as_tensor(env_ids, device=device, dtype=torch.long)
        else:
            env_ids_t = None
        
        # Run episode until all specified environments are done
        while True:
            # Check if all environments are done
            if env_ids_t is not None:
                all_done = torch.all(self.episode_done[env_ids_t])
            else:
                all_done = torch.all(self.episode_done)
            
            if all_done:
                break
            
            obs = self.get_obs()
            # Preprocess observation (e.g., resize images to policy's required resolution)
            obs = self.policy.preprocess_observation(obs)
            actions = self.policy.get_action(obs)
            
            # Handle both single Action and action sequence (List[Action])
            # This matches RoboTwin's pattern where policies return action chunks
            if isinstance(actions, list):
                # Action sequence: execute each action sequentially
                # Note: take_action() already updates policy observation internally
                for action in actions:
                    obs = self.take_action(action, record=self.record)
                    # Check if all environments are done after each action
                    if env_ids_t is not None:
                        all_done = torch.all(self.episode_done[env_ids_t])
                    else:
                        all_done = torch.all(self.episode_done)
                    if all_done:
                        break
            else:
                # Single Action: execute directly
                obs = self.take_action(actions, record=self.record)
        
        # Return results for specified environments
        if env_ids_t is not None:
            return {
                'success': self._is_success[env_ids_t].cpu(),
                'steps': self.step_count[env_ids_t].cpu(),
                'episode_done': self.episode_done[env_ids_t].cpu(),
            }
        else:
            return {
                'success': self._is_success.cpu(),
                'steps': self.step_count.cpu(),
                'episode_done': self.episode_done.cpu(),
            }

    # ---------- Helpers ----------
    def _to_base(self, pos_w: np.ndarray | torch.Tensor, quat_w: np.ndarray | torch.Tensor):
        """World → robot base frame for all envs."""
        root = self.robot.data.root_state_w[:, 0:7]  # [B,7]
        p_w, q_w = self._ensure_batch_pose(pos_w, quat_w)
        pb, qb = subtract_frame_transforms(
            root[:, 0:3], root[:, 3:7], p_w, q_w
        )
        return pb, qb  # [B,3], [B,4]


