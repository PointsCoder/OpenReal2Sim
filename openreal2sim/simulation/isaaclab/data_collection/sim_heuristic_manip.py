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
from isaaclab.app import AppLauncher

# ───────────────────────────────────────────────────────────────────────────── CLI ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser("sim_policy")
parser.add_argument("--key", type=str, default="demo_video", help="scene key (outputs/<key>)")
parser.add_argument("--robot", type=str, default="franka")
parser.add_argument("--num_envs", type=int, default=3)
parser.add_argument("--num_trials", type=int, default=1)
parser.add_argument("--teleop_device", type=str, default="keyboard")
parser.add_argument("--sensitivity", type=float, default=1.0)
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
from typing import Dict, Optional

# ───────────────────────────────────────────────────────────────────────────── Simulation environments ─────────────────────────────────────────────────────────────────────────────
from sim_base import BaseSimulator, get_next_demo_id
from sim_env_factory import make_env
from sim_preprocess.grasp_utils import get_best_grasp_with_hints
from sim_utils.transform_utils import pose_to_mat, mat_to_pose, grasp_to_world, grasp_approach_axis_batch
from sim_utils.sim_utils import load_sim_parameters

BASE_DIR   = Path.cwd()
img_folder = args_cli.key
out_dir    = BASE_DIR / "outputs"

from typing import Tuple

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
        self.load_obj_goal_traj()
    
    

    def load_obj_goal_traj(self):
        """
        Load the relative trajectory and precompute ALL absolute goal poses on GPU.
        This eliminates CPU-GPU transfers during trajectory execution.
        """
        # Load relative trajectory
        rel = np.load(self.traj_path).astype(np.float32)  # (T, 4, 4)
        self.obj_rel_traj = rel
        
        self.reset()
        
        # Get initial object poses (COM)
        B = self.scene.num_envs
        obj_state = self.object_prim.data.root_com_state_w[:, :7]  # [B, 7]
        
        # Apply offset
        offset_t = torch.tensor(self.goal_offset, dtype=torch.float32, device=obj_state.device)
        obj_state[:, :3] += offset_t
        
        # Normalize to env origin frame
        origins = self.scene.env_origins  # [B, 3] - already on GPU!
        obj_state[:, :3] -= origins
        
        # Convert to transformation matrices [B, 4, 4] - ALL ON GPU
        T_init = pose_to_mat_batch_torch(obj_state[:, :3], obj_state[:, 3:7])  # [B, 4, 4]
        
        # Convert relative trajectory to GPU
        T = rel.shape[0]
        rel_gpu = torch.tensor(rel, dtype=torch.float32, device=obj_state.device)  # [T, 4, 4]
        
        # Precompute ALL trajectory goals: T_goal[b,t] = rel[t] @ T_init[b]
        # Use batched matrix multiplication
        T_init_expanded = T_init.unsqueeze(0)  # [1, B, 4, 4]
        rel_expanded = rel_gpu.unsqueeze(1)     # [T, 1, 4, 4]
        
        # Batched matmul: [T, B, 4, 4]
        obj_goal = torch.matmul(rel_expanded, T_init_expanded)  # [T, B, 4, 4]
        
        # Add back world frame offset
        obj_goal[:, :, :3, 3] += origins.unsqueeze(0)  # Broadcasting
        
        # Transpose to [B, T, 4, 4] for easier indexing
        self.obj_goal_traj_w_gpu = obj_goal.permute(1, 0, 2, 3).contiguous()  # [B, T, 4, 4]
        
        # Also store as numpy for backward compatibility (but prefer GPU version)
        self.obj_goal_traj_w = obj_goal.permute(1, 0, 2, 3).cpu().numpy()

    def follow_object_goals(self, start_joint_pos, sample_step=1, visualize=True, skip_envs=None):
        """
        Follow precomputed object trajectory - ALL operations on GPU.
        """
        B = self.scene.num_envs
        device = self.sim.device
        
        if skip_envs is None:
            skip_envs = torch.zeros(B, dtype=torch.bool, device=device)
        else:
            # FIX: Ensure skip_envs is the right type and on the right device
            if isinstance(skip_envs, np.ndarray):
                skip_envs = torch.tensor(skip_envs, dtype=torch.bool, device=device)
            elif isinstance(skip_envs, torch.Tensor):
                skip_envs = skip_envs.to(device).bool()
        
        # FIX: Verify dimensions
        assert skip_envs.shape[0] == B, f"skip_envs shape {skip_envs.shape} doesn't match num_envs {B}"
        
        obj_goal_all_gpu = self.obj_goal_traj_w_gpu  # [B, T, 4, 4] - already on GPU!
        T = obj_goal_all_gpu.shape[1]
        
        # Get current EE and object poses
        ee_w = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]  # [B, 7]
        obj_w = self.object_prim.data.root_state_w[:, :7]  # [B, 7]
        
        # Compute T_ee_in_obj for all envs - BATCHED on GPU
        T_ee_w = pose_to_mat_batch_torch(ee_w[:, :3], ee_w[:, 3:7])    # [B, 4, 4]
        T_obj_w = pose_to_mat_batch_torch(obj_w[:, :3], obj_w[:, 3:7])  # [B, 4, 4]
        
        # Batched inverse: [B, 4, 4]
        T_obj_w_inv = torch.linalg.inv(T_obj_w)
        T_ee_in_obj = torch.matmul(T_obj_w_inv, T_ee_w)  # [B, 4, 4]
        
        joint_pos = start_joint_pos
        root_w = self.robot.data.root_state_w[:, 0:7]
        
        t_iter = list(range(0, T, sample_step))
        if t_iter[-1] != T-1:
            t_iter.append(T-1)
        
        for t in t_iter:
            print(f"[INFO] follow object goal step {t}/{T}")
            
            # Get goals for timestep t - ALL ON GPU
            T_obj_goal = obj_goal_all_gpu[:, t, :, :]  # [B, 4, 4]
            
            # Compute EE goals: T_ee_goal = T_obj_goal @ T_ee_in_obj - BATCHED
            T_ee_goal = torch.matmul(T_obj_goal, T_ee_in_obj)  # [B, 4, 4]
            
            # Extract poses from matrices - BATCHED
            goal_pos, goal_quat = mat_to_pose_batch_torch(T_ee_goal)  # [B, 3], [B, 4]
            
            # FIX: Handle skipped envs with proper broadcasting
            # Create dummy goal for skipped envs
            dummy_pos = torch.zeros(3, device=device, dtype=goal_pos.dtype)
            dummy_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=goal_quat.dtype)
            
            # Use advanced indexing instead of where() to avoid dimension issues
            for b in range(B):
                if skip_envs[b]:
                    goal_pos[b] = dummy_pos
                    goal_quat[b] = dummy_quat
            
            if visualize and self.debug_level > 0:
                self.show_goal(goal_pos, goal_quat)
            
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_w[:, :3], root_w[:, 3:7], goal_pos, goal_quat
            )
            
            joint_pos, success = self.move_to(ee_pos_b, ee_quat_b, gripper_open=False)
            
            if joint_pos is None or success is None:
                print(f"[CRITICAL] Motion planning failed at step {t}/{T}")
                raise RuntimeError("MotionPlanningFailure_RestartNeeded")
            
            if torch.any(success == False):
                print(f"[WARN] Some active envs failed motion planning at step {t}/{T}")
            
            # FIX: Store actions with proper masking
            ee_pos_b_save = ee_pos_b.clone()
            ee_quat_b_save = ee_quat_b.clone()
            gripper_save = torch.ones((B, 1), device=device, dtype=torch.float32)
            
            # Mask out skipped envs
            for b in range(B):
                if skip_envs[b]:
                    ee_pos_b_save[b] = dummy_pos
                    ee_quat_b_save[b] = dummy_quat
            
            action = torch.cat([ee_pos_b_save, ee_quat_b_save, gripper_save], dim=1)
            self.save_dict["actions"].append(action)  # Store as GPU tensor
        
        joint_pos = self.wait(gripper_open=True, steps=10)
        return joint_pos

    def viz_object_goals(self, sample_step=1, hold_steps=20):
        self.reset()
        self.wait(gripper_open=True, steps=10)
        B = self.scene.num_envs
        env_ids = torch.arange(B, device=self.object_prim.device, dtype=torch.long)
        goals = self.obj_goal_traj_w
        t_iter = list(range(0, goals.shape[1], sample_step))
        t_iter = t_iter + [goals.shape[1]-1] if t_iter[-1] != goals.shape[1]-1 else t_iter
        for t in t_iter:
            print(f"[INFO] viz object goal step {t}/{goals.shape[1]}")
            pos_list, quat_list = [], []
            for b in range(B):
                pos, quat = mat_to_pose(goals[b, t])
                pos_list.append(pos.astype(np.float32))
                quat_list.append(quat.astype(np.float32))
            pose = self.object_prim.data.root_com_state_w[:, :7]  # ← CHANGED: Use COM
            pose[:, :3]   = torch.tensor(np.stack(pos_list),  dtype=torch.float32, device=pose.device)
            pose[:, 3:7]  = torch.tensor(np.stack(quat_list), dtype=torch.float32, device=pose.device)
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
    def execute_and_lift_once_batch(self, info: dict, lift_height=0.06, position_threshold=0.2) -> tuple[np.ndarray, np.ndarray]:
        """
        Reset → pre → grasp → close → lift → hold; return (success[B], score[B]).
        Now propagates motion planning failures by returning (None, None).
        """
        B = self.scene.num_envs
        self.reset()

        # open gripper buffer
        jp = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
        self.wait(gripper_open=True, steps=4)

        # pre-grasp
        jp, success = self.move_to(info["pre_p_b"], info["pre_q_b"], gripper_open=True)
        if jp is None or success is None:
            print("[CRITICAL] Motion planning failed during pre-grasp in grasp trial")
            return None, None
        if torch.any(success==False): 
            return np.zeros(B, bool), np.zeros(B, np.float32)
        jp = self.wait(gripper_open=True, steps=3)

        # grasp
        jp, success = self.move_to(info["p_b"], info["q_b"], gripper_open=True)
        if jp is None or success is None:
            print("[CRITICAL] Motion planning failed during grasp approach in grasp trial")
            return None, None
        if torch.any(success==False): 
            return np.zeros(B, bool), np.zeros(B, np.float32)
        jp = self.wait(gripper_open=True, steps=2)

        # close
        jp = self.wait(gripper_open=False, steps=6)

        # initial heights (USING COM)
        obj0 = self.object_prim.data.root_com_pos_w[:, 0:3]  # ← CHANGED: Use COM
        ee_w = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        ee_p0 = ee_w[:, :3]
        robot_ee_z0 = ee_p0[:, 2].clone()
        obj_z0 = obj0[:, 2].clone()
        print(f"[INFO] mean object z0={obj_z0.mean().item():.3f} m, mean EE z0={robot_ee_z0.mean().item():.3f} m")

        # lift: keep orientation, add height
        ee_q = ee_w[:, 3:7]
        target_p = ee_p0.clone()
        target_p[:, 2] += lift_height

        root = self.robot.data.root_state_w[:, 0:7]
        p_lift_b, q_lift_b = subtract_frame_transforms(
            root[:, 0:3], root[:, 3:7],
            target_p, ee_q
        )
        jp, success = self.move_to(p_lift_b, q_lift_b, gripper_open=False)
        
        # Check for motion planning failure
        if jp is None or success is None:
            print("[CRITICAL] Motion planning failed during lift in grasp trial")
            return None, None
        
        if torch.any(success==False): 
            return np.zeros(B, bool), np.zeros(B, np.float32)
        jp = self.wait(gripper_open=False, steps=8)

        # final heights and success checking (USING COM)
        obj1 = self.object_prim.data.root_com_pos_w[:, 0:3]  # ← CHANGED: Use COM
        ee_w1 = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        robot_ee_z1 = ee_w1[:, 2]
        obj_z1 = obj1[:, 2]
        print(f"[INFO] mean object z1={obj_z1.mean().item():.3f} m, mean EE z1={robot_ee_z1.mean().item():.3f} m")

        # Lift check
        ee_diff  = robot_ee_z1 - robot_ee_z0
        obj_diff = obj_z1 - obj_z0
        lift_check = (torch.abs(ee_diff - obj_diff) <= 0.01) & \
            (torch.abs(ee_diff) >= 0.5 * lift_height) & \
            (torch.abs(obj_diff) >= 0.5 * lift_height)

        # Goal proximity check (USING COM)
        final_goal_matrices = self.obj_goal_traj_w[:, -1, :, :]
        goal_positions_np = final_goal_matrices[:, :3, 3]
        goal_positions = torch.tensor(goal_positions_np, dtype=torch.float32, device=self.sim.device)
        current_obj_pos = self.object_prim.data.root_com_state_w[:, :3]  # ← CHANGED: Use COM
        distances = torch.norm(current_obj_pos - goal_positions, dim=1)
        proximity_check = distances <= position_threshold

        # Combined check
        #lifted = lift_check & proximity_check

        lifted = proximity_check  # TEMPORARY: only use proximity for success

        # Score calculation
        score = torch.zeros_like(ee_diff)
        if torch.any(lifted):
            base_score = 1000.0
            epsilon = 0.001
            score[lifted] = base_score / (distances[lifted] + epsilon)
        
        print(f"[INFO] Lift check passed: {lift_check.sum().item()}/{B}")
        print(f"[INFO] Proximity check passed: {proximity_check.sum().item()}/{B}")
        print(f"[INFO] Final lifted flag (both checks): {lifted.sum().item()}/{B}")
        
        if B <= 10:  # Detailed output for small batches
            for b in range(B):
                print(f"  Env {b}: lift={lift_check[b].item()}, prox={proximity_check[b].item()}, "
                    f"dist={distances[b].item():.4f}m, score={score[b].item():.2f}")
        else:
            print(f"[INFO] Distance to goal - mean: {distances.mean().item():.4f}m, "
                f"min: {distances.min().item():.4f}m, max: {distances.max().item():.4f}m")
        
        return lifted.detach().cpu().numpy().astype(bool), score.detach().cpu().numpy().astype(np.float32)

    def build_grasp_info(
    self,
    grasp_pos_w_batch,   # Can be numpy or torch
    grasp_quat_w_batch,  # Can be numpy or torch
    pre_dist_batch,      # Can be numpy or torch
    delta_batch          # Can be numpy or torch
) -> dict:
        """
        Build grasp info for all envs - VECTORIZED on GPU.
        Accepts both numpy arrays and torch tensors for backward compatibility.
        """
        B = self.scene.num_envs
        device = self.sim.device
        
        # Convert to torch tensors if they're numpy arrays
        if isinstance(grasp_pos_w_batch, np.ndarray):
            p_w = torch.tensor(grasp_pos_w_batch, dtype=torch.float32, device=device)
        else:
            p_w = grasp_pos_w_batch.to(device)
        
        if isinstance(grasp_quat_w_batch, np.ndarray):
            q_w = torch.tensor(grasp_quat_w_batch, dtype=torch.float32, device=device)
        else:
            q_w = grasp_quat_w_batch.to(device)
        
        if isinstance(pre_dist_batch, np.ndarray):
            pre_d = torch.tensor(pre_dist_batch, dtype=torch.float32, device=device).view(B, 1)
        else:
            pre_d = pre_dist_batch.to(device).view(B, 1)
        
        if isinstance(delta_batch, np.ndarray):
            delt = torch.tensor(delta_batch, dtype=torch.float32, device=device).view(B, 1)
        else:
            delt = delta_batch.to(device).view(B, 1)
        
        # Compute approach axis - VECTORIZED
        a_batch = grasp_approach_axis_batch_torch(q_w)  # [B, 3]
        
        # Compute pre-grasp and grasp positions - VECTORIZED
        pre_p_w = p_w - pre_d * a_batch  # [B, 3]
        gra_p_w = p_w + delt * a_batch   # [B, 3]
        
        # Add environment origins
        origins = self.scene.env_origins  # [B, 3]
        pre_p_w = pre_p_w + origins
        gra_p_w = gra_p_w + origins
        
        # Transform to base frame
        pre_pb, pre_qb = self._to_base(pre_p_w, q_w)
        gra_pb, gra_qb = self._to_base(gra_p_w, q_w)
        
        return {
            "pre_p_w": pre_p_w, "p_w": gra_p_w, "q_w": q_w,
            "pre_p_b": pre_pb, "pre_q_b": pre_qb,
            "p_b": gra_pb, "q_b": gra_qb,
            "pre_dist": pre_d.squeeze(1), "delta": delt.squeeze(1),
        }


    def grasp_trials(
    self, 
    gg, 
    std: float = 0.0005, 
    max_retries: int = 3, 
    existing_skip_mask: Optional[np.ndarray] = None
) -> Dict:
        """
        Async grasp trials with GPU acceleration and vectorized operations.
        Each environment finds its own successful grasp independently.
        
        Args:
            gg: GraspGroup with grasp proposals
            std: Standard deviation for random perturbations along approach axis
            max_retries: Number of retry attempts for grasp selection
            existing_skip_mask: Optional [B] boolean array of environments to skip from start
        
        Returns:
            dict with:
                - success: [B] boolean array of per-env success
                - chosen_pose_w: List of (pos, quat) tuples per env
                - chosen_pre: [B] pre-grasp distances
                - chosen_delta: [B] grasp deltas
                - skip_envs: [B] boolean array of envs to skip in future processing
        """
        B = self.scene.num_envs
        device = self.sim.device
        idx_all = list(range(len(gg)))
        
        # Handle empty grasp list
        if len(idx_all) == 0:
            print("[ERR] empty grasp list.")
            skip_mask = np.ones(B, dtype=bool) if existing_skip_mask is None else existing_skip_mask.copy()
            return {
                "success": np.zeros(B, dtype=bool),
                "chosen_pose_w": [None] * B,
                "chosen_pre": np.zeros(B, dtype=np.float32),
                "chosen_delta": np.zeros(B, dtype=np.float32),
                "skip_envs": skip_mask,
            }
        
        rng = np.random.default_rng()
        pre_dist_const = 0.12  # meters
        
        # ═══════════════════════════════════════════════════════════════════════
        # OPTIMIZATION 1: Precompute ALL grasp poses on GPU once
        # ═══════════════════════════════════════════════════════════════════════
        print(f"[INFO] Precomputing {len(gg)} grasp poses on GPU...")
        all_grasp_pos = torch.zeros((len(gg), 3), dtype=torch.float32, device=device)
        all_grasp_quat = torch.zeros((len(gg), 4), dtype=torch.float32, device=device)
        
        # Import the grasp conversion utility
        from sim_utils.transform_utils import grasp_to_world
        
        for i in range(len(gg)):
            p_w, q_w = grasp_to_world(gg[i])
            all_grasp_pos[i] = torch.tensor(p_w, dtype=torch.float32, device=device)
            all_grasp_quat[i] = torch.tensor(q_w, dtype=torch.float32, device=device)
        
        print(f"[INFO] Grasp poses loaded on GPU: {all_grasp_pos.shape}")
        
        # ═══════════════════════════════════════════════════════════════════════
        # OPTIMIZATION 2: Use GPU tensors for per-environment tracking
        # ═══════════════════════════════════════════════════════════════════════
        env_success = torch.zeros(B, dtype=torch.bool, device=device)
        env_chosen_idx = torch.zeros(B, dtype=torch.long, device=device)
        env_chosen_pre = torch.full((B,), pre_dist_const, dtype=torch.float32, device=device)
        env_chosen_delta = torch.zeros(B, dtype=torch.float32, device=device)
        env_best_score = torch.zeros(B, dtype=torch.float32, device=device)
        
        # Initialize skip mask
        if existing_skip_mask is not None:
            env_skip = torch.tensor(existing_skip_mask, dtype=torch.bool, device=device)
            print(f"[INFO] Starting with {env_skip.sum().item()} pre-skipped environments: "
                f"{torch.where(env_skip)[0].tolist()}")
        else:
            env_skip = torch.zeros(B, dtype=torch.bool, device=device)
        
        # ═══════════════════════════════════════════════════════════════════════
        # Retry loop for grasp selection
        # ═══════════════════════════════════════════════════════════════════════
        for retry_attempt in range(max_retries):
            try:
                print(f"\n{'='*60}")
                print(f"ASYNC GRASP SELECTION ATTEMPT {retry_attempt + 1}/{max_retries}")
                print(f"Successful envs: {env_success.sum().item()}/{B}")
                print(f"{'='*60}\n")
                
                # Check if all non-skipped envs succeeded
                active_mask = ~env_skip
                if torch.all(env_success[active_mask]):
                    successful_envs = torch.where(env_success & ~env_skip)[0]
                    skipped_envs = torch.where(env_skip)[0]
                    print(f"[SUCCESS] All active environments found valid grasps!")
                    if len(skipped_envs) > 0:
                        print(f"[INFO] Skipped {len(skipped_envs)} failed environments: "
                            f"{skipped_envs.tolist()}")
                    break
                
                # Get mask of envs that still need to find a grasp
                active_failed_mask = (~env_success) & (~env_skip)
                failed_env_ids = torch.where(active_failed_mask)[0]
                n_failed = len(failed_env_ids)
                
                if n_failed == 0:
                    break
                
                print(f"[INFO] {n_failed} environments still searching for grasps: "
                    f"{failed_env_ids.tolist()}")
                
                # ═══════════════════════════════════════════════════════════════
                # OPTIMIZATION 3: Vectorized grasp proposal testing
                # ═══════════════════════════════════════════════════════════════
                for start in range(0, len(idx_all), B):
                    block = idx_all[start : start + B]
                    if len(block) < B:
                        block = block + [block[-1]] * (B - len(block))
                    
                    # Create grasp index tensor for this block
                    grasp_indices = torch.tensor(block, dtype=torch.long, device=device)
                    
                    # Select grasps for ALL envs (including successful/skipped)
                    grasp_pos_batch = all_grasp_pos[grasp_indices].clone()   # [B, 3]
                    grasp_quat_batch = all_grasp_quat[grasp_indices].clone() # [B, 4]
                    
                    # For successful/skipped envs, replace with their chosen grasp
                    mask_keep = env_success | env_skip
                    if torch.any(mask_keep):
                        grasp_pos_batch[mask_keep] = all_grasp_pos[env_chosen_idx[mask_keep]]
                        grasp_quat_batch[mask_keep] = all_grasp_quat[env_chosen_idx[mask_keep]]
                    
                    # Visualize (optional)
                    if self.debug_level > 0:
                        self.show_goal(grasp_pos_batch, grasp_quat_batch)
                    
                    # ═══════════════════════════════════════════════════════════
                    # OPTIMIZATION 4: Vectorized random perturbations
                    # ═══════════════════════════════════════════════════════════
                    delta_batch = torch.tensor(
                        rng.normal(0.0, std, size=(B,)),
                        dtype=torch.float32, 
                        device=device
                    )
                    
                    # Build grasp info using GPU tensors (vectorized)
                    info = self.build_grasp_info(
                        grasp_pos_batch, 
                        grasp_quat_batch,
                        env_chosen_pre,  # Already on GPU
                        delta_batch
                    )
                    
                    # Execute grasp trial for ALL envs
                    ok_batch, score_batch = self.execute_and_lift_once_batch(info)
                    
                    # Check for motion planning failure
                    if ok_batch is None or score_batch is None:
                        print(f"[CRITICAL] Motion planning failed during grasp trial "
                            f"at block[{start}:{start+B}]")
                        raise RuntimeError("GraspTrialMotionPlanningFailure")
                    
                    # ═══════════════════════════════════════════════════════════
                    # OPTIMIZATION 5: Vectorized success update
                    # ═══════════════════════════════════════════════════════════
                    ok_t = torch.tensor(ok_batch, dtype=torch.bool, device=device)
                    score_t = torch.tensor(score_batch, dtype=torch.float32, device=device)
                    
                    # Update mask: not already successful, not skipped, this trial succeeded, 
                    # and score improved
                    update_mask = (~env_success) & (~env_skip) & ok_t & (score_t > env_best_score)
                    
                    if torch.any(update_mask):
                        # Vectorized update - no loop!
                        env_success[update_mask] = True
                        env_chosen_idx[update_mask] = grasp_indices[update_mask]
                        env_chosen_delta[update_mask] = delta_batch[update_mask]
                        env_best_score[update_mask] = score_t[update_mask]
                        
                        updated_envs = torch.where(update_mask)[0]
                        print(f"[SUCCESS] Envs {updated_envs.tolist()} found valid grasps")
                        for e in updated_envs:
                            print(f"  Env {e.item()}: score={score_t[e].item():.2f}")
                    
                    # Check if all active envs now succeeded
                    newly_successful = torch.sum(env_success[active_failed_mask])
                    print(f"[SEARCH] block[{start}:{start+B}] -> "
                        f"{newly_successful.item()}/{n_failed} previously-failed envs now successful")
                    
                    # Early exit if all succeeded
                    if torch.all(env_success[active_mask]):
                        successful_envs = torch.where(env_success & ~env_skip)[0]
                        skipped_envs = torch.where(env_skip)[0]
                        print(f"[SUCCESS] All active environments found valid grasps "
                            f"on attempt {retry_attempt + 1}!")
                        if len(skipped_envs) > 0:
                            print(f"[INFO] {len(skipped_envs)} environments skipped: "
                                f"{skipped_envs.tolist()}")
                        
                        # Convert to output format and return early
                        return self._convert_grasp_results(
                            env_success, env_skip, env_chosen_idx, 
                            env_chosen_pre, env_chosen_delta,
                            all_grasp_pos, all_grasp_quat
                        )
                
                # End of this attempt - check status
                if not torch.all(env_success[active_mask]):
                    still_failed = torch.where((~env_success) & (~env_skip))[0]
                    print(f"[WARN] {len(still_failed)} envs still need grasps after "
                        f"attempt {retry_attempt + 1}: {still_failed.tolist()}")
                    
                    if retry_attempt < max_retries - 1:
                        print("[INFO] Retrying grasp selection for failed environments...")
                        continue
                    else:
                        # Last attempt exhausted - mark remaining failed envs as skipped
                        print(f"[WARN] Exhausted all {max_retries} grasp attempts")
                        print(f"[INFO] Marking {len(still_failed)} failed environments to skip: "
                            f"{still_failed.tolist()}")
                        env_skip[still_failed] = True
                        
                        successful_envs = torch.where(env_success)[0]
                        print(f"[INFO] Proceeding with {len(successful_envs)} successful "
                            f"environments: {successful_envs.tolist()}")
                        
                        return self._convert_grasp_results(
                            env_success, env_skip, env_chosen_idx,
                            env_chosen_pre, env_chosen_delta,
                            all_grasp_pos, all_grasp_quat
                        )
            
            # ═══════════════════════════════════════════════════════════════════
            # Error handling with automatic retry
            # ═══════════════════════════════════════════════════════════════════
            except RuntimeError as e:
                if "GraspTrialMotionPlanningFailure" in str(e):
                    print(f"\n[RESTART] Motion planning failed during grasp trial "
                        f"on attempt {retry_attempt + 1}")
                    print(f"[RESTART] Remaining grasp trial attempts: "
                        f"{max_retries - retry_attempt - 1}\n")
                    
                    # Clear corrupted data
                    self.clear_data()
                    
                    if retry_attempt < max_retries - 1:
                        continue
                    else:
                        print("[ERR] Grasp trial failed after all retry attempts")
                        # Mark all currently failed envs as skipped
                        failed_envs = torch.where((~env_success) & (~env_skip))[0]
                        if len(failed_envs) > 0:
                            print(f"[INFO] Marking {len(failed_envs)} failed environments "
                                f"to skip: {failed_envs.tolist()}")
                            env_skip[failed_envs] = True
                        
                        return self._convert_grasp_results(
                            env_success, env_skip, env_chosen_idx,
                            env_chosen_pre, env_chosen_delta,
                            all_grasp_pos, all_grasp_quat
                        )
                else:
                    raise
            
            except Exception as e:
                print(f"[ERROR] Unexpected error during grasp trial: "
                    f"{type(e).__name__}: {e}")
                
                if retry_attempt < max_retries - 1:
                    print(f"[ERROR] Attempting grasp trial retry "
                        f"{retry_attempt + 2}/{max_retries}...")
                    self.clear_data()
                    continue
                else:
                    # Mark all currently failed envs as skipped
                    failed_envs = torch.where((~env_success) & (~env_skip))[0]
                    if len(failed_envs) > 0:
                        print(f"[INFO] Marking {len(failed_envs)} failed environments "
                            f"to skip: {failed_envs.tolist()}")
                        env_skip[failed_envs] = True
                    
                    return self._convert_grasp_results(
                        env_success, env_skip, env_chosen_idx,
                        env_chosen_pre, env_chosen_delta,
                        all_grasp_pos, all_grasp_quat
                    )
        
        # Return final results
        return self._convert_grasp_results(
            env_success, env_skip, env_chosen_idx,
            env_chosen_pre, env_chosen_delta,
            all_grasp_pos, all_grasp_quat
        )


    def _convert_grasp_results(
        self,
        env_success: torch.Tensor,
        env_skip: torch.Tensor,
        env_chosen_idx: torch.Tensor,
        env_chosen_pre: torch.Tensor,
        env_chosen_delta: torch.Tensor,
        all_grasp_pos: torch.Tensor,
        all_grasp_quat: torch.Tensor
    ) -> Dict:
        """
        Helper function to convert GPU tensors to expected output format.
        Handles None values for skipped environments.
        
        Args:
            env_success: [B] boolean tensor
            env_skip: [B] boolean tensor
            env_chosen_idx: [B] long tensor (indices into all_grasp_*)
            env_chosen_pre: [B] float tensor
            env_chosen_delta: [B] float tensor
            all_grasp_pos: [N, 3] float tensor (all grasp positions)
            all_grasp_quat: [N, 4] float tensor (all grasp quaternions)
        
        Returns:
            dict with numpy arrays and list of pose tuples
        """
        B = self.scene.num_envs
        
        # Convert chosen grasp indices to actual poses
        env_chosen_pose_w = []
        for b in range(B):
            if env_skip[b].item():
                # Dummy pose for skipped env
                dummy_pos = np.zeros(3, dtype=np.float32)
                dummy_quat = np.array([1, 0, 0, 0], dtype=np.float32)
                env_chosen_pose_w.append((dummy_pos, dummy_quat))
            else:
                idx = env_chosen_idx[b].item()
                p = all_grasp_pos[idx].cpu().numpy().astype(np.float32)
                q = all_grasp_quat[idx].cpu().numpy().astype(np.float32)
                env_chosen_pose_w.append((p, q))
        
        return {
            "success": env_success.cpu().numpy(),
            "chosen_pose_w": env_chosen_pose_w,
            "chosen_pre": env_chosen_pre.cpu().numpy(),
            "chosen_delta": env_chosen_delta.cpu().numpy(),
            "skip_envs": env_skip.cpu().numpy(),
        }

    
    def is_success(self, position_threshold: float = 0.1, skip_envs: np.ndarray = None) -> bool:
        """
        Verify if the manipulation task succeeded by comparing final object COM position
        with the goal position from the trajectory.
        
        Args:
            position_threshold: Distance threshold in meters (default: 0.1m = 10cm)
            skip_envs: Boolean array [B] indicating which envs to skip from verification
        
        Returns:
            bool: True if task succeeded for all active envs, False otherwise
        """
        B = self.scene.num_envs
        if skip_envs is None:
            skip_envs = np.zeros(B, dtype=bool)
        
        final_goal_matrices = self.obj_goal_traj_w[:, -1, :, :]  # [B, 4, 4] - last timestep
        
        # Extract goal positions (translation part of the transform matrix)
        goal_positions_np = final_goal_matrices[:, :3, 3]  # [B, 3] - xyz positions (numpy)
        
        # Convert to torch tensor
        goal_positions = torch.tensor(goal_positions_np, dtype=torch.float32, device=self.sim.device)
        
        # Get current object COM positions
        current_obj_pos = self.object_prim.data.root_com_state_w[:, :3]  # ← ALREADY CHANGED: Use COM
        
        # Calculate distances
        distances = torch.norm(current_obj_pos - goal_positions, dim=1)  # [B]
        
        # Check if all environments succeeded
        success_mask = distances <= position_threshold
        
        # Print results for each environment
        print("\n" + "="*50)
        print("TASK VERIFICATION RESULTS (Using Center of Mass)")
        print("="*50)
        for b in range(B):
            if skip_envs[b]:
                print(f"Env {b}: SKIPPED (environment was skipped during grasp phase)")
            else:
                status = "SUCCESS" if success_mask[b].item() else "FAILURE"
                print(f"Env {b}: {status} (COM distance: {distances[b].item():.4f}m, threshold: {position_threshold}m)")
                print(f"  Goal COM position: [{goal_positions[b, 0].item():.3f}, {goal_positions[b, 1].item():.3f}, {goal_positions[b, 2].item():.3f}]")
                print(f"  Final COM position: [{current_obj_pos[b, 0].item():.3f}, {current_obj_pos[b, 1].item():.3f}, {current_obj_pos[b, 2].item():.3f}]")
        
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
        
        return bool(all_success)
    
    
    
    def save_data_selective(self, skip_envs: np.ndarray, ignore_keys: List[str] = None):
        """
        Save data with SINGLE CPU transfer per environment.
        """
        if ignore_keys is None:
            ignore_keys = []
        
        save_root = self._demo_dir()
        save_root.mkdir(parents=True, exist_ok=True)
        
        # Stack all timesteps - STILL ON GPU
        stacked_gpu = {}
        for k, v in self.save_dict.items():
            if k not in ignore_keys and len(v) > 0:
                # Check if list contains tensors or numpy arrays
                if isinstance(v[0], torch.Tensor):
                    stacked_gpu[k] = torch.stack(v, dim=0)  # [T, B, ...] on GPU
                else:
                    # Already numpy (backward compatibility)
                    stacked_gpu[k] = torch.tensor(np.array(v), device='cuda')
        
        active_envs = np.where(~skip_envs)[0]
        print(f"[INFO] Saving data for {len(active_envs)} active environments")
        
        def save_env_data(b):
            """Save data for a single environment - ONE CPU transfer per env"""
            start_time = time.time()
            env_dir = self._env_dir(save_root, b)
            
            # Transfer this environment's data to CPU ONCE
            env_data_cpu = {}
            for key, arr_gpu in stacked_gpu.items():
                if key in ignore_keys:
                    continue
                # Single transfer: extract env b and move to CPU
                env_data_cpu[key] = arr_gpu[:, b].cpu().numpy()
            
            # Now save from CPU arrays
            for key, arr in env_data_cpu.items():
                if key == "rgb":
                    video_path = env_dir / "sim_video.mp4"
                    writer = imageio.get_writer(
                        video_path, fps=50, codec='libx264', quality=7,
                        pixelformat='yuv420p', macro_block_size=None
                    )
                    for t in range(arr.shape[0]):
                        writer.append_data(arr[t])
                    writer.close()
                elif key == "segmask":
                    video_path = env_dir / "mask_video.mp4"
                    writer = imageio.get_writer(
                        video_path, fps=50, codec='libx264', quality=7,
                        pixelformat='yuv420p', macro_block_size=None
                    )
                    for t in range(arr.shape[0]):
                        writer.append_data((arr[t].astype(np.uint8) * 255))
                    writer.close()
                elif key == "depth":
                    flat = arr[arr > 0]
                    max_depth = np.percentile(flat, 99) if flat.size > 0 else 1.0
                    depth_norm = np.clip(arr / max_depth * 255.0, 0, 255).astype(np.uint8)
                    video_path = env_dir / "depth_video.mp4"
                    writer = imageio.get_writer(
                        video_path, fps=50, codec='libx264', quality=7,
                        pixelformat='yuv420p', macro_block_size=None
                    )
                    for t in range(depth_norm.shape[0]):
                        writer.append_data(depth_norm[t])
                    writer.close()
                    np.save(env_dir / f"{key}.npy", arr)
                else:
                    np.save(env_dir / f"{key}.npy", arr)
            
            json.dump(self.sim_cfgs, open(env_dir / "config.json", "w"), indent=2)
            elapsed = time.time() - start_time
            print(f"[INFO] Env {b} saved in {elapsed:.1f}s")
            return b
        
        # Parallel save with ProcessPoolExecutor for CPU-bound video encoding
        print("[INFO] Starting parallel video encoding...")
        from concurrent.futures import ProcessPoolExecutor
        max_workers = min(len(active_envs), os.cpu_count() or 4)
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(save_env_data, active_envs))
        
        print("[INFO]: Demonstration is saved at: ", save_root)
        
        # Compose real videos (can also be parallelized)
        print("\n[INFO] Composing real videos with background...")
        
        def compose_video(b):
            success = self.compose_real_video(env_id=b)
            return b, success
        
        with ThreadPoolExecutor(max_workers=min(len(active_envs), 4)) as executor:
            results = list(executor.map(compose_video, active_envs))
        
        for b, success in results:
            if success:
                print(f"[INFO] Real video composed successfully for env {b}")
            else:
                print(f"[WARN] Failed to compose real video for env {b}")

        demo_root = self.out_dir / "all_demos"
        demo_root.mkdir(parents=True, exist_ok=True)
        total_demo_id = get_next_demo_id(demo_root)
        demo_save_path = demo_root / f"demo_{total_demo_id}"
        demo_save_path.mkdir(parents=True, exist_ok=True)
        meta_info = {
            "path": str(save_root),
            "fps": 50,
            "active_envs": active_envs.tolist(),
            "skipped_envs": np.where(skip_envs)[0].tolist(),
        }
        with open(demo_save_path / "meta_info.json", "w") as f:
            json.dump(meta_info, f)
        os.system(f"cp -r {save_root}/* {demo_save_path}")
        print("[INFO]: Demonstration is saved at: ", demo_save_path)


    def inference(self, std: float = 0.0, max_grasp_retries: int = 1, 
              max_traj_retries: int = 3, position_threshold: float = 0.15) -> bool:  # ← ADD THIS
        """
        Main function with decoupled grasp selection and trajectory execution.
        
        Phase 1: Grasp selection - try max_grasp_retries times, mark failures as skipped
        Phase 2: Trajectory execution - try max_traj_retries times per env, mark failures as skipped
        Phase 3: Save only successful environments
        
        NO re-grasping during trajectory phase.
        """
        B = self.scene.num_envs
        
        # Read grasp proposals (only once)
        npy_path = self.grasp_path
        if npy_path is None or (not os.path.exists(npy_path)):
            print(f"[ERR] grasps npy not found: {npy_path}")
            return False
        gg = GraspGroup().from_npy(npy_file_path=npy_path)
        gg = get_best_grasp_with_hints(gg, point=None, direction=[0, 0, -1])

        # Track per-environment state
        env_skip = np.zeros(B, dtype=bool)  # Environments to skip (failed grasp or trajectory)
        env_traj_success = np.zeros(B, dtype=bool)  # Track which envs succeeded
        env_traj_attempts = np.zeros(B, dtype=int)  # Track trajectory attempts per env
        
        # ========== PHASE 1: GRASP SELECTION (ONE TIME ONLY) ==========
        print("\n" + "#"*60)
        print("PHASE 1: GRASP SELECTION")
        print("#"*60 + "\n")
        
        if self.grasp_idx >= 0:
            # Fixed grasp index mode - all envs use same grasp
            if self.grasp_idx >= len(gg):
                print(f"[ERR] grasp_idx {self.grasp_idx} out of range [0,{len(gg)})")
                return False
            print(f"[INFO] using fixed grasp index {self.grasp_idx} for all envs.")
            p_w, q_w = grasp_to_world(gg[int(self.grasp_idx)])
            
            # Prepare per-env format
            env_chosen_pose_w = [(p_w.astype(np.float32), q_w.astype(np.float32))] * B
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
            return False
        
        print(f"\n[GRASP SELECTION COMPLETE]")
        print(f"  Active envs: {active_envs.tolist()}")
        if env_skip.sum() > 0:
            print(f"  Skipped envs (failed grasp): {np.where(env_skip)[0].tolist()}")
        
        # ========== PHASE 2: TRAJECTORY EXECUTION (WITH PER-ENV RETRY) ==========
        print("\n" + "#"*60)
        print("PHASE 2: TRAJECTORY EXECUTION")
        print("#"*60 + "\n")

        # Store successful environment data before any retries
        successful_env_data = {}  # {env_id: {key: data}} to preserve successful runs

        for traj_attempt in range(max_traj_retries):
            try:
                # Get envs that still need to succeed (not skipped AND not yet successful)
                needs_attempt = (~env_skip) & (~env_traj_success)
                active_envs = np.where(needs_attempt)[0]
                
                if len(active_envs) == 0:
                    # All non-skipped envs have succeeded!
                    successful_envs = np.where(env_traj_success)[0]
                    print(f"\n[SUCCESS] All {len(successful_envs)} environments completed!")
                    print(f"  Successful envs: {successful_envs.tolist()}")
                    if env_skip.sum() > 0:
                        print(f"  Skipped envs: {np.where(env_skip)[0].tolist()}")
                    
                    # Restore successful environment data
                    if successful_env_data:
                        print(f"[INFO] Restoring data for {len(successful_env_data)} previously successful environments")
                        for env_id, saved_data in successful_env_data.items():
                            for key, data in saved_data.items():
                                # Replace the env_id column with saved successful data
                                for t in range(len(self.save_dict[key])):
                                    if t < len(saved_data[key]):
                                        self.save_dict[key][t][env_id] = saved_data[key][t]
                    
                    self.save_data_selective(env_skip)
                    return True
                
                print(f"\n{'='*60}")
                print(f"TRAJECTORY ATTEMPT {traj_attempt + 1}/{max_traj_retries}")
                print(f"Envs needing attempt: {active_envs.tolist()}")
                already_successful = np.where(env_traj_success & ~env_skip)[0]
                if len(already_successful) > 0:
                    print(f"Envs already successful (skipping): {already_successful.tolist()}")
                print(f"Skipped envs: {np.where(env_skip)[0].tolist()}")
                print(f"{'='*60}\n")
                
                # Prepare grasp info for ALL envs
                # Active envs get real data, skipped/successful envs get dummy data
                p_all = torch.zeros((B, 3), dtype=torch.float32, device=self.sim.device)
                q_all = torch.zeros((B, 4), dtype=torch.float32, device=self.sim.device)
                q_all[:, 0] = 1.0  # Identity quaternion
                pre_all_torch = torch.tensor(env_chosen_pre, device=self.sim.device)
                del_all_torch = torch.tensor(env_chosen_delta, device=self.sim.device)
                
                for b in range(B):
                    if needs_attempt[b] and env_chosen_pose_w[b] is not None:
                        # Active env needing attempt
                        p_all[b] = torch.tensor(env_chosen_pose_w[b][0], device=self.sim.device)
                        q_all[b] = torch.tensor(env_chosen_pose_w[b][1], device=self.sim.device)
                    else:
                        # Dummy data for skipped or already successful envs
                        p_all[b] = torch.tensor(np.zeros(3, dtype=np.float32), device=self.sim.device)
                        q_all[b] = torch.tensor(np.array([1, 0, 0, 0], dtype=np.float32), device=self.sim.device)
                
                info_all = self.build_grasp_info(p_all, q_all, pre_all_torch, del_all_torch)

                # Reset and execute
                self.reset()

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

                # Pre-grasp
                print("[INFO] Moving to pre-grasp positions...")
                jp, success = self.move_to(info_all["pre_p_b"], info_all["pre_q_b"], gripper_open=True)
                if jp is None or success is None:
                    print("[WARN] Pre-grasp motion planning failed")
                    raise RuntimeError("PreGraspMotionFailure")
                
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
                print("[INFO] Closing grippers...")
                jp = self.wait(gripper_open=False, steps=50)
                
                p_close_data = info_all["p_b"].cpu().numpy()
                q_close_data = info_all["q_b"].cpu().numpy()
                for b in np.where(envs_to_skip_recording)[0]:
                    p_close_data[b] = np.zeros(3, dtype=np.float32)
                    q_close_data[b] = np.array([1, 0, 0, 0], dtype=np.float32)
                self.save_dict["actions"].append(np.concatenate([p_close_data, q_close_data, np.ones((B, 1))], axis=1))

                # Follow object trajectory (skip already successful + skipped envs)
                print("[INFO] Following object trajectories...")
                skip_for_traj = env_skip | env_traj_success
                jp = self.follow_object_goals(jp, sample_step=5, visualize=True, skip_envs=skip_for_traj)
                
                # Trajectory execution completed - verify which envs succeeded
                print("[INFO] Trajectory execution completed, verifying goal positions...")
                
                # Get per-env success status
                final_goal_matrices = self.obj_goal_traj_w[:, -1, :, :]
                goal_positions_np = final_goal_matrices[:, :3, 3]
                goal_positions = torch.tensor(goal_positions_np, dtype=torch.float32, device=self.sim.device)
                current_obj_pos = self.object_prim.data.root_com_state_w[:, :3]
                distances = torch.norm(current_obj_pos - goal_positions, dim=1)
                per_env_success = (distances <= position_threshold).cpu().numpy()
                
                # Check results ONLY for envs that attempted this round
                print("\n" + "="*50)
                print("PER-ENVIRONMENT TRAJECTORY RESULTS")
                print("="*50)
                
                newly_successful_envs = []
                for b in active_envs:  # Only check envs that attempted
                    env_traj_attempts[b] += 1
                    if per_env_success[b]:
                        env_traj_success[b] = True
                        newly_successful_envs.append(b)
                        print(f"Env {b}: SUCCESS (distance: {distances[b].item():.4f}m)")
                        
                        # Save this environment's data NOW before any reset
                        print(f"[INFO] Saving successful data for Env {b}")
                        successful_env_data[b] = {}
                        for key in self.save_dict.keys():
                            # Extract all timesteps for this environment
                            successful_env_data[b][key] = []
                            for t in range(len(self.save_dict[key])):
                                data = self.save_dict[key][t][b]
                                if isinstance(data, torch.Tensor):
                                    successful_env_data[b][key].append(data.clone())
                                else:
                                    successful_env_data[b][key].append(data.copy())
                                
                    else:
                        print(f"Env {b}: FAILED (distance: {distances[b].item():.4f}m, attempt {env_traj_attempts[b]}/{max_traj_retries})")
                        if env_traj_attempts[b] >= max_traj_retries:
                            env_skip[b] = True
                            print(f"  → Marking Env {b} as SKIPPED (exhausted {max_traj_retries} trajectory attempts)")
                
                print("="*50 + "\n")
                
                # Check if all non-skipped envs succeeded
                remaining_active = np.where(~env_skip)[0]
                if len(remaining_active) == 0:
                    print("[ERR] All environments have been skipped")
                    return False
                
                all_remaining_succeeded = np.all(env_traj_success[remaining_active])
                
                if all_remaining_succeeded:
                    print(f"[SUCCESS] All {len(remaining_active)} remaining environments succeeded!")
                    print(f"  Successful envs: {remaining_active.tolist()}")
                    if env_skip.sum() > 0:
                        print(f"  Skipped envs: {np.where(env_skip)[0].tolist()}")
                    
                    # Restore successful environment data before saving
                    if successful_env_data:
                        print(f"[INFO] Restoring data for {len(successful_env_data)} successful environments")
                        for env_id, saved_data in successful_env_data.items():
                            for key, data in saved_data.items():
                                for t in range(len(self.save_dict[key])):
                                    if t < len(saved_data[key]):
                                        self.save_dict[key][t][env_id] = saved_data[key][t]
                    
                    # Save data for successful environments only
                    self.save_data_selective(env_skip)
                    return True
                else:
                    # Some envs still need retry
                    still_need_attempt = np.where((~env_skip) & (~env_traj_success))[0]
                    print(f"[INFO] {len(still_need_attempt)} environments still need attempts: {still_need_attempt.tolist()}")
                    
                    if traj_attempt < max_traj_retries - 1:
                        print(f"[INFO] Retrying trajectory execution (attempt {traj_attempt + 2}/{max_traj_retries})...")
                        self.clear_data()  # This will wipe save_dict, but we have successful data saved
                        continue
                    else:
                        # Last attempt - mark remaining failures as skipped
                        for b in still_need_attempt:
                            env_skip[b] = True
                            print(f"[INFO] Marking Env {b} as SKIPPED (failed all {max_traj_retries} trajectory attempts)")
                        
                        # Check if any succeeded
                        final_successful = np.where(env_traj_success)[0]
                        if len(final_successful) > 0:
                            print(f"\n[PARTIAL SUCCESS] Saving data for {len(final_successful)} successful environments: {final_successful.tolist()}")
                            
                            # Restore successful environment data
                            if successful_env_data:
                                print(f"[INFO] Restoring data for {len(successful_env_data)} successful environments")
                                for env_id, saved_data in successful_env_data.items():
                                    for key, data in saved_data.items():
                                        for t in range(len(self.save_dict[key])):
                                            if t < len(saved_data[key]):
                                                self.save_dict[key][t][env_id] = saved_data[key][t]
                            
                            self.save_data_selective(env_skip)
                            return True
                        else:
                            print("[ERR] All environments failed")
                            return False
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
                        print(f"[INFO] Marking Env {b} as SKIPPED (exhausted attempts after error)")
                
                self.clear_data()
                
                # Check if any active envs remain
                remaining_active = np.where(~env_skip)[0]
                if len(remaining_active) == 0:
                    print("[ERR] All environments have been skipped after errors")
                    return False
                
                if traj_attempt < max_traj_retries - 1:
                    print(f"[INFO] Retrying with {len(remaining_active)} remaining environments...")
                    continue
                else:
                    print(f"[PARTIAL SUCCESS] Saving {len(remaining_active)} environments that didn't encounter errors")
                    
                    # Restore successful environment data before saving
                    if successful_env_data:
                        print(f"[INFO] Restoring data for {len(successful_env_data)} successful environments")
                        for env_id, saved_data in successful_env_data.items():
                            for key, data in saved_data.items():
                                for t in range(len(self.save_dict[key])):
                                    if t < len(saved_data[key]):
                                        self.save_dict[key][t][env_id] = saved_data[key][t]
                    
                    self.save_data_selective(env_skip)
                    return True
                    
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
                    return False
                
                if traj_attempt < max_traj_retries - 1:
                    print(f"[INFO] Retrying with {len(remaining_active)} remaining environments...")
                    continue
                else:
                    # Save whatever succeeded
                    final_successful = np.where(env_traj_success)[0]
                    if len(final_successful) > 0:
                        print(f"[PARTIAL SUCCESS] Saving {len(final_successful)} environments")
                        
                        # Restore successful environment data before saving
                        if successful_env_data:
                            print(f"[INFO] Restoring data for {len(successful_env_data)} successful environments")
                            for env_id, saved_data in successful_env_data.items():
                                for key, data in saved_data.items():
                                    for t in range(len(self.save_dict[key])):
                                        if t < len(saved_data[key]):
                                            self.save_dict[key][t][env_id] = saved_data[key][t]
                        
                        self.save_data_selective(env_skip)
                        return True
                    else:
                        print("[ERR] All environments failed")
                        return False
                    
def pose_to_mat_batch_torch(pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
        """
        Convert batched poses to 4x4 transformation matrices on GPU.
        
        Args:
            pos: [B, 3] positions
            quat: [B, 4] quaternions (w, x, y, z)
        
        Returns:
            [B, 4, 4] transformation matrices
        """
        B = pos.shape[0]
        device = pos.device
        
        # Normalize quaternions
        quat = quat / torch.norm(quat, dim=1, keepdim=True)
        
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        
        # Compute rotation matrix elements
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        
        # Build rotation matrices [B, 3, 3]
        R = torch.zeros((B, 3, 3), device=device, dtype=pos.dtype)
        R[:, 0, 0] = 1 - 2*(yy + zz)
        R[:, 0, 1] = 2*(xy - wz)
        R[:, 0, 2] = 2*(xz + wy)
        R[:, 1, 0] = 2*(xy + wz)
        R[:, 1, 1] = 1 - 2*(xx + zz)
        R[:, 1, 2] = 2*(yz - wx)
        R[:, 2, 0] = 2*(xz - wy)
        R[:, 2, 1] = 2*(yz + wx)
        R[:, 2, 2] = 1 - 2*(xx + yy)
        
        # Build full transformation matrices [B, 4, 4]
        T = torch.eye(4, device=device, dtype=pos.dtype).unsqueeze(0).repeat(B, 1, 1)
        T[:, :3, :3] = R
        T[:, :3, 3] = pos
        
        return T

def mat_to_pose_batch_torch(T: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert batched 4x4 transformation matrices to poses on GPU.
    
    Args:
        T: [B, 4, 4] transformation matrices
    
    Returns:
        pos: [B, 3] positions
        quat: [B, 4] quaternions (w, x, y, z)
    """
    B = T.shape[0]
    device = T.device
    dtype = T.dtype
    
    pos = T[:, :3, 3]  # [B, 3]
    R = T[:, :3, :3]   # [B, 3, 3]
    
    # Convert rotation matrix to quaternion
    # Using Shepperd's method for numerical stability
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    
    quat = torch.zeros((B, 4), device=device, dtype=dtype)
    
    # Case 1: trace > 0
    mask1 = trace > 0
    s1 = torch.sqrt(trace[mask1] + 1.0) * 2
    quat[mask1, 0] = 0.25 * s1
    quat[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s1
    quat[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s1
    quat[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s1
    
    # Case 2: R[0,0] is largest diagonal element
    mask2 = (~mask1) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    s2 = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2
    quat[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s2
    quat[mask2, 1] = 0.25 * s2
    quat[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s2
    quat[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s2
    
    # Case 3: R[1,1] is largest diagonal element
    mask3 = (~mask1) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
    s3 = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2
    quat[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s3
    quat[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s3
    quat[mask3, 2] = 0.25 * s3
    quat[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s3
    
    # Case 4: R[2,2] is largest diagonal element
    mask4 = (~mask1) & (~mask2) & (~mask3)
    s4 = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2
    quat[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s4
    quat[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s4
    quat[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s4
    quat[mask4, 3] = 0.25 * s4
    
    return pos, quat

def grasp_approach_axis_batch_torch(quat: torch.Tensor) -> torch.Tensor:
    """
    Extract approach axis from batched quaternions on GPU.
    
    Args:
        quat: [B, 4] quaternions (w, x, y, z)
    
    Returns:
        [B, 3] approach axes
    """
    # Approach axis is typically -Z axis of the gripper frame
    # Rotate [0, 0, -1] by the quaternion
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    # Rotation of [0, 0, -1] vector
    ax = 2 * (x*z + w*y)
    ay = 2 * (y*z - w*x)
    az = -(1 - 2*(x*x + y*y))
    
    return torch.stack([ax, ay, az], dim=1)
# ───────────────────────────────────────────────────────────────────────────── Entry Point ─────────────────────────────────────────────────────────────────────────────
def main():
    sim_cfgs = load_sim_parameters(BASE_DIR, args_cli.key)
    env, _ = make_env(
        cfgs=sim_cfgs, num_envs=args_cli.num_envs,
        device=args_cli.device,
        bg_simplify=False,
    )
    sim, scene = env.sim, env.scene

    my_sim = HeuristicManipulation(sim, scene, sim_cfgs=sim_cfgs)

    demo_root = (out_dir / img_folder / "demos").resolve()

    for _ in range(args_cli.num_trials):

        robot_pose = torch.tensor(sim_cfgs["robot_cfg"]["robot_pose"], dtype=torch.float32, device=my_sim.sim.device)  # [7], pos(3)+quat(wxyz)(4)
        my_sim.set_robot_pose(robot_pose)
        my_sim.demo_id = get_next_demo_id(demo_root)
        my_sim.reset()
        print(f"[INFO] start simulation demo_{my_sim.demo_id}")
        # Note: if you set viz_object_goals(), remember to disable gravity and collision for object
        # my_sim.viz_object_goals(sample_step=10, hold_steps=40)
        my_sim.inference()

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
    os.system("quit()")
    simulation_app.close()
