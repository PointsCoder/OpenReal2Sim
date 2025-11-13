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
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_trials", type=int, default=1)
parser.add_argument("--teleop_device", type=str, default="keyboard")
parser.add_argument("--sensitivity", type=float, default=1.0)
parser.add_argument("--debug", action="store_true", help="Enable debug logging")
parser.add_argument("--target_successes", type=int, default=None, help="Minimum number of successful demonstrations to collect (will rerun until target is met)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.headless = False  # headless mode for batch execution
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

# ───────────────────────────────────────────────────────────────────────────── Simulation environments ─────────────────────────────────────────────────────────────────────────────
from sim_base import BaseSimulator, get_next_demo_id
from sim_env_factory import make_env
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
        self.load_obj_goal_traj()
    
    def vprint(self, *args, **kwargs):
        """Print only if verbose flag is enabled"""
        if self.verbose:
            print(*args, **kwargs)

    def load_obj_goal_traj(self):
        """
        Load the relative trajectory Δ_w (T,4,4) and precompute the absolute
        object goal trajectory for each env using the *actual current* object pose
        in the scene as T_obj_init (not env_origin).
        T_obj_goal[t] = Δ_w[t] @ T_obj_init

        Sets:
        self.obj_rel_traj   : np.ndarray or None, shape (T,4,4)
        self.obj_goal_traj_w: np.ndarray or None, shape (B,T,4,4)
        """
        # ── 1) Load Δ_w ──
        rel = np.load(self.traj_path).astype(np.float32)
        self.obj_rel_traj = rel  # (T,4,4)

        self.reset()

        # ── 2) Read current object initial pose per env as T_obj_init (USING COM) ──
        B = self.scene.num_envs
        obj_state = self.object_prim.data.root_com_state_w[:, :7]  # ← CHANGED: Use COM
        self.show_goal(obj_state[:, :3], obj_state[:, 3:7])

        obj_state_np = obj_state.detach().cpu().numpy().astype(np.float32)
        offset_np = np.asarray(self.goal_offset, dtype=np.float32).reshape(3)
        obj_state_np[:, :3] += offset_np  # raise a bit to avoid collision

        # Add small random variations to avoid background dynamics issues
        rng = np.random.default_rng()
        
        # Position variations: ±1-2cm in x,y,z
        pos_variations = rng.uniform(-0.015, 0.015, size=(B, 3)).astype(np.float32)  # ±1.5cm
        obj_state_np[:, :3] += pos_variations
        
        # Orientation variations: small rotations around each axis (±5-10 degrees)
        for b in range(B):
            # Convert quaternion to axis-angle for easier manipulation
            quat = obj_state_np[b, 3:7]
            
            # Generate small random rotations (±5-10 degrees = ±0.087-0.174 radians)
            rot_x = rng.uniform(-0.087, 0.087)  # ±5 degrees
            rot_y = rng.uniform(-0.087, 0.087)  # ±5 degrees  
            rot_z = rng.uniform(-0.087, 0.087)  # ±5 degrees
            
            # Apply rotations sequentially
            if abs(rot_x) > 1e-6:
                quat_x = transforms3d.quaternions.axangle2quat([1, 0, 0], rot_x)
                quat = transforms3d.quaternions.qmult(quat, quat_x)
            if abs(rot_y) > 1e-6:
                quat_y = transforms3d.quaternions.axangle2quat([0, 1, 0], rot_y)
                quat = transforms3d.quaternions.qmult(quat, quat_y)
            if abs(rot_z) > 1e-6:
                quat_z = transforms3d.quaternions.axangle2quat([0, 0, 1], rot_z)
                quat = transforms3d.quaternions.qmult(quat, quat_z)
            
            # Normalize quaternion to ensure validity
            quat = quat / np.linalg.norm(quat)
            obj_state_np[b, 3:7] = quat.astype(np.float32)

        # Note: here the relative traj Δ_w is defined in world frame with origin (0,0,0),
        # Hence, we need to normalize it to each env's origin frame.
        origins = self.scene.env_origins.detach().cpu().numpy().astype(np.float32)  # (B,3)
        obj_state_np[:, :3] -= origins # normalize to env origin frame

        # ── 3) Precompute absolute object goals for all envs ──
        T = rel.shape[0]
        obj_goal = np.zeros((B, T, 4, 4), dtype=np.float32)
        for b in range(B):
            T_init = pose_to_mat(obj_state_np[b, :3], obj_state_np[b, 3:7])  # (4,4)
            for t in range(T):
                goal = rel[t] @ T_init
                goal[:3, 3] += origins[b]  # back to world frame
                obj_goal[b, t] = goal

        self.obj_goal_traj_w = obj_goal  # [B, T, 4, 4]

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

        joint_pos = start_joint_pos
        root_w = self.robot.data.root_state_w[:, 0:7]

        t_iter = list(range(0, T, sample_step))
        t_iter = t_iter + [T-1] if t_iter[-1] != T-1 else t_iter

        for t in t_iter:
            goal_pos_list, goal_quat_list = [], []
            if self.verbose:
                print(f"[INFO] follow object goal step {t}/{T}")
            for b in range(B):
                if skip_envs[b]:
                    # Dummy goal for skipped env
                    goal_pos_list.append(np.zeros(3, dtype=np.float32))
                    goal_quat_list.append(np.array([1, 0, 0, 0], dtype=np.float32))
                else:
                    T_obj_goal = obj_goal_all[b, t]
                    T_ee_goal  = T_obj_goal @ T_ee_in_obj[b]
                    pos_b, quat_b = mat_to_pose(T_ee_goal)
                    goal_pos_list.append(pos_b.astype(np.float32))
                    goal_quat_list.append(quat_b.astype(np.float32))

            goal_pos  = torch.as_tensor(np.stack(goal_pos_list),  dtype=torch.float32, device=self.sim.device)
            goal_quat = torch.as_tensor(np.stack(goal_quat_list), dtype=torch.float32, device=self.sim.device)

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
            
            # Build action data: real data for active envs, dummy data for skipped envs
            ee_pos_b_data = ee_pos_b.cpu().numpy()
            ee_quat_b_data = ee_quat_b.cpu().numpy()
            for b in np.where(skip_envs)[0]:
                ee_pos_b_data[b] = np.zeros(3, dtype=np.float32)
                ee_quat_b_data[b] = np.array([1, 0, 0, 0], dtype=np.float32)
            self.save_dict["actions"].append(np.concatenate([ee_pos_b_data, ee_quat_b_data, np.ones((B, 1))], axis=1))

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
            if self.verbose:
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
        if self.verbose:
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
        if self.verbose:
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
        lifted = lift_check & proximity_check

        # Score calculation
        score = torch.zeros_like(ee_diff)
        if torch.any(lifted):
            base_score = 1000.0
            epsilon = 0.001
            score[lifted] = base_score / (distances[lifted] + epsilon)
        
        if self.verbose:
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

                    # Build grasp proposals for ALL envs (even successful ones get dummy data)
                    grasp_pos_w_batch, grasp_quat_w_batch = [], []
                    for b in range(B):
                        if env_success[b] or env_skip[b]:
                            # Use existing successful grasp or dummy for skipped env
                            if env_success[b]:
                                p_w, q_w = env_chosen_pose_w[b]
                            else:
                                # Dummy grasp for skipped env
                                p_w, q_w = grasp_to_world(gg[0])
                            grasp_pos_w_batch.append(p_w.astype(np.float32))
                            grasp_quat_w_batch.append(q_w.astype(np.float32))
                        else:
                            # New grasp proposal for failed env
                            idx = block[b % len(block)]
                            p_w, q_w = grasp_to_world(gg[int(idx)])
                            grasp_pos_w_batch.append(p_w.astype(np.float32))
                            grasp_quat_w_batch.append(q_w.astype(np.float32))
                    
                    grasp_pos_w_batch  = np.stack(grasp_pos_w_batch,  axis=0)  # (B,3)
                    grasp_quat_w_batch = np.stack(grasp_quat_w_batch, axis=0)  # (B,4)
                    self.show_goal(grasp_pos_w_batch, grasp_quat_w_batch)
                    
                    # Random disturbance along approach axis
                    pre_dist_batch = np.full((B,), pre_dist_const, dtype=np.float32)
                    delta_batch = rng.normal(0.0, std, size=(B,)).astype(np.float32)

                    info = self.build_grasp_info(grasp_pos_w_batch, grasp_quat_w_batch,
                                                pre_dist_batch, delta_batch)

                    # Execute grasp trial for ALL envs
                    ok_batch, score_batch = self.execute_and_lift_once_batch(info)
                    
                    # Check if motion planning returned valid results
                    if ok_batch is None or score_batch is None:
                        print(f"[CRITICAL] Motion planning failed during grasp trial at block[{start}:{start+B}]")
                        raise RuntimeError("GraspTrialMotionPlanningFailure")
                    
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
                                if self.verbose:
                                    print(f"[SUCCESS] Env {b} found valid grasp (score: {score_batch[b]:.2f})")
                    
                    # Check if all active envs now succeeded
                    active_mask = ~env_skip
                    newly_successful = np.sum(env_success[active_failed_mask])
                    if self.verbose:
                        print(f"[SEARCH] block[{start}:{start+B}] -> {newly_successful}/{n_failed} previously-failed envs now successful")
                    
                    if np.all(env_success[active_mask]):
                        successful_envs = np.where(env_success & ~env_skip)[0]
                        skipped_envs = np.where(env_skip)[0]
                        if self.verbose:
                            print(f"[SUCCESS] All active environments found valid grasps on attempt {retry_attempt + 1}!")
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
    
    def save_data_selective(self, skip_envs: np.ndarray, ignore_keys: List[str] = None, 
                           demo_id: int = None, env_offset: int = 0):
        """
        Save data for successful environments.
        
        Args:
            skip_envs: Boolean array indicating which envs to skip
            ignore_keys: Keys to not save
            demo_id: Optional demo ID to use (if None, generates new ID)
            env_offset: Offset to add to environment IDs when saving (for multi-run collection)
        """
        if ignore_keys is None:
            ignore_keys = []
            
        save_root = self._demo_dir()
        if demo_id is not None:
            # Use provided demo_id instead of auto-generated one
            save_root = self.out_dir / img_folder / "demos" / f"demo_{demo_id}"
        save_root.mkdir(parents=True, exist_ok=True)

        # Stack data with proper shape handling
        stacked = {}
        for k, v in self.save_dict.items():
            try:
                arr = np.array(v)
                # Check if we got an object array (inconsistent shapes)
                if arr.dtype == object:
                    # Try to stack with explicit dtype
                    if len(v) > 0 and hasattr(v[0], 'shape'):
                        arr = np.stack(v, axis=0)
                stacked[k] = arr
            except Exception as e:
                if self.verbose:
                    print(f"[WARN] Failed to stack {k}: {e}. Saving as list.")
                stacked[k] = v
        
        active_envs = np.where(~skip_envs)[0]
        print(f"[INFO] Saving data for {len(active_envs)} active environments: {active_envs.tolist()}")

        def save_env_data(b):
            """Save data for a single environment"""
            start_time = time.time()
            # Add env_offset to the saved environment ID
            save_env_id = b + env_offset
            env_dir = self._env_dir(save_root, save_env_id)
            
            for key, arr in stacked.items():
                if key in ignore_keys:
                    continue
                    
                # Handle different array types/shapes
                if isinstance(arr, list):
                    # If stacking failed, handle as list
                    if self.verbose:
                        print(f"[WARN] Skipping {key} for env {b} (failed to stack)")
                    continue
                    
                # Check array dimensions
                if arr.ndim < 2:
                    if self.verbose:
                        print(f"[WARN] Skipping {key} for env {b} (array is {arr.ndim}D, expected at least 2D)")
                    continue
                
                if key == "rgb":
                    video_path = env_dir / "sim_video.mp4"
                    writer = imageio.get_writer(
                        video_path, fps=50, codec='libx264', quality=7, 
                        pixelformat='yuv420p', macro_block_size=None
                    )
                    for t in range(arr.shape[0]):
                        writer.append_data(arr[t, b])
                    writer.close()
                elif key == "segmask":
                    video_path = env_dir / "mask_video.mp4"
                    writer = imageio.get_writer(
                        video_path, fps=50, codec='libx264', quality=7,
                        pixelformat='yuv420p', macro_block_size=None
                    )
                    for t in range(arr.shape[0]):
                        writer.append_data((arr[t, b].astype(np.uint8) * 255))
                    writer.close()
                elif key == "depth":
                    depth_seq = arr[:, b]
                    flat = depth_seq[depth_seq > 0]
                    max_depth = np.percentile(flat, 99) if flat.size > 0 else 1.0
                    depth_norm = np.clip(depth_seq / max_depth * 255.0, 0, 255).astype(np.uint8)
                    video_path = env_dir / "depth_video.mp4"
                    writer = imageio.get_writer(
                        video_path, fps=50, codec='libx264', quality=7,
                        pixelformat='yuv420p', macro_block_size=None
                    )
                    for t in range(depth_norm.shape[0]):
                        writer.append_data(depth_norm[t])
                    writer.close()
                    np.save(env_dir / f"{key}.npy", depth_seq)
                else:
                    np.save(env_dir / f"{key}.npy", arr[:, b])
            
            json.dump(self.sim_cfgs, open(env_dir / "config.json", "w"), indent=2)
            elapsed = time.time() - start_time
            if self.verbose:
                print(f"[INFO] Env {save_env_id} saved in {elapsed:.1f}s")
            return b

        # Save environments in parallel
        if self.verbose:
            print("[INFO] Starting parallel video encoding...")
        with ThreadPoolExecutor(max_workers=min(len(active_envs), 4)) as executor:
            list(executor.map(save_env_data, active_envs))
        
        print("[INFO]: Demonstration is saved at: ", save_root)
        
        # Compose real videos (can also be parallelized)
        if self.verbose:
            print("\n[INFO] Composing real videos with background...")
        
        def compose_video(b):
            # Use the offset environment ID where data was actually saved
            save_env_id = b + env_offset
            success = self.compose_real_video(env_id=save_env_id)
            return save_env_id, success
        
        with ThreadPoolExecutor(max_workers=min(len(active_envs), 4)) as executor:
            results = list(executor.map(compose_video, active_envs))
        
        if self.verbose:
            for save_env_id, success in results:
                if success:
                    print(f"[INFO] Real video composed successfully for env {save_env_id}")
                else:
                    print(f"[WARN] Failed to compose real video for env {save_env_id}")

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
        
        # Return number of successful environments saved
        return len(active_envs)


    def inference(self, std: float = 0.0, max_grasp_retries: int = 1, 
              max_traj_retries: int = 3, position_threshold: float = 0.15,
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
                return 0
            if self.verbose:
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
            return 0
        
        if self.verbose:
            print(f"\n[GRASP SELECTION COMPLETE]")
            print(f"  Active envs: {active_envs.tolist()}")
        if env_skip.sum() > 0 and self.verbose:
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
                    if self.verbose:
                        print(f"\n[SUCCESS] All {len(successful_envs)} environments completed!")
                        print(f"  Successful envs: {successful_envs.tolist()}")
                    if env_skip.sum() > 0 and self.verbose:
                        print(f"  Skipped envs: {np.where(env_skip)[0].tolist()}")
                    
                    # Restore successful environment data
                    if successful_env_data and self.verbose:
                        print(f"[INFO] Restoring data for {len(successful_env_data)} previously successful environments")
                        for env_id, saved_data in successful_env_data.items():
                            for key, data in saved_data.items():
                                # Replace the env_id column with saved successful data
                                for t in range(len(self.save_dict[key])):
                                    if t < len(saved_data[key]):
                                        self.save_dict[key][t][env_id] = saved_data[key][t]
                    
                    num_saved = self.save_data_selective(env_skip, demo_id=demo_id, env_offset=env_offset)
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
                pre_all = env_chosen_pre.copy()
                del_all = env_chosen_delta.copy()
                
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
                if self.verbose:
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
                jp = self.follow_object_goals(jp, sample_step=5, visualize=True, skip_envs=skip_for_traj)
                
                # Trajectory execution completed - verify which envs succeeded
                if self.verbose:
                    print("[INFO] Trajectory execution completed, verifying goal positions...")
                
                # Get per-env success status
                final_goal_matrices = self.obj_goal_traj_w[:, -1, :, :]
                goal_positions_np = final_goal_matrices[:, :3, 3]
                goal_positions = torch.tensor(goal_positions_np, dtype=torch.float32, device=self.sim.device)
                current_obj_pos = self.object_prim.data.root_com_state_w[:, :3]
                distances = torch.norm(current_obj_pos - goal_positions, dim=1)
                per_env_success = (distances <= position_threshold).cpu().numpy()
                
                # Check results ONLY for envs that attempted this round
                if self.verbose:
                    print("\n" + "="*50)
                    print("PER-ENVIRONMENT TRAJECTORY RESULTS")
                    print("="*50)
                
                newly_successful_envs = []
                for b in active_envs:  # Only check envs that attempted
                    env_traj_attempts[b] += 1
                    if per_env_success[b]:
                        env_traj_success[b] = True
                        newly_successful_envs.append(b)
                        if self.verbose:
                            print(f"Env {b}: SUCCESS (distance: {distances[b].item():.4f}m)")
                        
                        # Save this environment's data NOW before any reset
                        if self.verbose:
                            print(f"[INFO] Saving successful data for Env {b}")
                        successful_env_data[b] = {}
                        for key in self.save_dict.keys():
                            # Extract all timesteps for this environment
                            successful_env_data[b][key] = []
                            for t in range(len(self.save_dict[key])):
                                successful_env_data[b][key].append(self.save_dict[key][t][b].copy())
                                
                    else:
                        if self.verbose:
                            print(f"Env {b}: FAILED (distance: {distances[b].item():.4f}m, attempt {env_traj_attempts[b]}/{max_traj_retries})")
                        if env_traj_attempts[b] >= max_traj_retries:
                            env_skip[b] = True
                            if self.verbose:
                                print(f"  → Marking Env {b} as SKIPPED (exhausted {max_traj_retries} trajectory attempts)")
                
                if self.verbose:
                    print("="*50 + "\n")
                
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
                    
                    # Restore successful environment data before saving
                    if successful_env_data and self.verbose:
                        print(f"[INFO] Restoring data for {len(successful_env_data)} successful environments")
                    if successful_env_data:
                        for env_id, saved_data in successful_env_data.items():
                            for key, data in saved_data.items():
                                for t in range(len(self.save_dict[key])):
                                    if t < len(saved_data[key]):
                                        self.save_dict[key][t][env_id] = saved_data[key][t]
                    
                    # Save data for successful environments only
                    num_saved = self.save_data_selective(env_skip, demo_id=demo_id, env_offset=env_offset)
                    return num_saved
                else:
                    # Some envs still need retry
                    still_need_attempt = np.where((~env_skip) & (~env_traj_success))[0]
                    if self.verbose:
                        print(f"[INFO] {len(still_need_attempt)} environments still need attempts: {still_need_attempt.tolist()}")
                    
                    if traj_attempt < max_traj_retries - 1:
                        if self.verbose:
                            print(f"[INFO] Retrying trajectory execution (attempt {traj_attempt + 2}/{max_traj_retries})...")
                        self.clear_data()  # This will wipe save_dict, but we have successful data saved
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
                            
                            # Restore successful environment data
                            if successful_env_data and self.verbose:
                                print(f"[INFO] Restoring data for {len(successful_env_data)} successful environments")
                            if successful_env_data:
                                for env_id, saved_data in successful_env_data.items():
                                    for key, data in saved_data.items():
                                        for t in range(len(self.save_dict[key])):
                                            if t < len(saved_data[key]):
                                                self.save_dict[key][t][env_id] = saved_data[key][t]
                            
                            num_saved = self.save_data_selective(env_skip, demo_id=demo_id, env_offset=env_offset)
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
                
                self.clear_data()
                
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
                    
                    # Restore successful environment data before saving
                    if successful_env_data and self.verbose:
                        print(f"[INFO] Restoring data for {len(successful_env_data)} successful environments")
                    if successful_env_data:
                        for env_id, saved_data in successful_env_data.items():
                            for key, data in saved_data.items():
                                for t in range(len(self.save_dict[key])):
                                    if t < len(saved_data[key]):
                                        self.save_dict[key][t][env_id] = saved_data[key][t]
                    
                    num_saved = self.save_data_selective(env_skip, demo_id=demo_id, env_offset=env_offset)
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
                        
                        # Restore successful environment data before saving
                        if successful_env_data:
                            print(f"[INFO] Restoring data for {len(successful_env_data)} successful environments")
                            for env_id, saved_data in successful_env_data.items():
                                for key, data in saved_data.items():
                                    for t in range(len(self.save_dict[key])):
                                        if t < len(saved_data[key]):
                                            self.save_dict[key][t][env_id] = saved_data[key][t]
                        
                        num_saved = self.save_data_selective(env_skip, demo_id=demo_id, env_offset=env_offset)
                        return num_saved
                    else:
                        print("[ERR] All environments failed")
                        return 0
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
            
            # Run inference with shared demo_id and cumulative env_offset
            num_saved = my_sim.inference(demo_id=demo_id, env_offset=env_offset)
            
            total_successes += num_saved
            env_offset += num_saved
            
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
    else:
        # Original behavior: single trial per num_trials
        for _ in range(args_cli.num_trials):

            robot_pose = torch.tensor(sim_cfgs["robot_cfg"]["robot_pose"], dtype=torch.float32, device=my_sim.sim.device)
            my_sim.set_robot_pose(robot_pose)
            my_sim.demo_id = get_next_demo_id(demo_root)
            my_sim.reset()
            print(f"[INFO] start simulation demo_{my_sim.demo_id}")
            my_sim.inference()

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
    os.system("quit()")
    simulation_app.close()