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
        self.load_obj_goal_traj()

    def load_obj_goal_traj(self):
        """
        Load the relative trajectory Î”_w (T,4,4) and precompute the absolute
        object goal trajectory for each env using the *actual current* object pose
        in the scene as T_obj_init (not env_origin).
          T_obj_goal[t] = Î”_w[t] @ T_obj_init

        Sets:
          self.obj_rel_traj   : np.ndarray or None, shape (T,4,4)
          self.obj_goal_traj_w: np.ndarray or None, shape (B,T,4,4)
        """
        # â€”â€” 1) Load Î”_w â€”â€”
        rel = np.load(self.traj_path).astype(np.float32)
        self.obj_rel_traj = rel  # (T,4,4)

        self.reset()

        # â€”â€” 2) Read current object initial pose per env as T_obj_init â€”â€”
        B = self.scene.num_envs
        # obj_state = self.object_prim.data.root_com_state_w[:, :7]  # [B,7], pos(3)+quat(wxyz)(4)
        obj_state = self.object_prim.data.root_state_w[:, :7]  # [B,7], pos(3)+quat(wxyz)(4)
        self.show_goal(obj_state[:, :3], obj_state[:, 3:7])

        obj_state_np = obj_state.detach().cpu().numpy().astype(np.float32)
        offset_np = np.asarray(self.goal_offset, dtype=np.float32).reshape(3)
        obj_state_np[:, :3] += offset_np  # raise a bit to avoid collision

        # Note: here the relative traj Î”_w is defined in world frame with origin (0,0,0),
        # Hence, we need to normalize it to each env's origin frame.
        origins = self.scene.env_origins.detach().cpu().numpy().astype(np.float32)  # (B,3)
        obj_state_np[:, :3] -= origins # normalize to env origin frame

        # â€”â€” 3) Precompute absolute object goals for all envs â€”â€”
        T = rel.shape[0]
        obj_goal = np.zeros((B, T, 4, 4), dtype=np.float32)
        for b in range(B):
            T_init = pose_to_mat(obj_state_np[b, :3], obj_state_np[b, 3:7])  # (4,4)
            for t in range(T):
                goal = rel[t] @ T_init
                goal[:3, 3] += origins[b]  # back to world frame
                obj_goal[b, t] = goal

        self.obj_goal_traj_w = obj_goal  # [B, T, 4, 4]

    def follow_object_goals(self, start_joint_pos, sample_step=1, visualize=True):
        """
        Follow precomputed object absolute trajectory with automatic restart on failure.
        If motion planning fails, raises an exception to trigger re-grasp from step 0.
        """
        B = self.scene.num_envs
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
            print(f"[INFO] follow object goal step {t}/{T}")
            for b in range(B):
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
            
            joint_pos, success = self.move_to(ee_pos_b, ee_quat_b, gripper_open=False)
            
            # Check for critical motion planning failure
            if joint_pos is None or success is None:
                print(f"[CRITICAL] Motion planning failed at step {t}/{T}")
                print("[CRITICAL] Object likely dropped - need to restart from grasp")
                raise RuntimeError("MotionPlanningFailure_RestartNeeded")
            
            if torch.any(success == False):
                print(f"[WARN] Some envs failed motion planning at step {t}/{T}, but continuing...")
            
            self.save_dict["actions"].append(np.concatenate([ee_pos_b.cpu().numpy(), ee_quat_b.cpu().numpy(), np.ones((B, 1))], axis=1))

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
            pose = self.object_prim.data.root_state_w[:, :7]
            # pose = self.object_prim.data.root_com_state_w[:, :7]
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

        # initial heights
        obj0 = self.object_prim.data.root_com_pos_w[:, 0:3]
        ee_w = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        ee_p0 = ee_w[:, :3]
        robot_ee_z0 = ee_p0[:, 2].clone()
        obj_z0 = obj0[:, 2].clone()
        print(f"[INFO] mean object z0={obj_z0.mean().item():.3f} m, mean EE z0={robot_ee_z0.mean().item():.3f} m")  # ADD THIS BACK

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

        # final heights and success checking
        obj1 = self.object_prim.data.root_com_pos_w[:, 0:3]
        ee_w1 = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        robot_ee_z1 = ee_w1[:, 2]
        obj_z1 = obj1[:, 2]
        print(f"[INFO] mean object z1={obj_z1.mean().item():.3f} m, mean EE z1={robot_ee_z1.mean().item():.3f} m")  # ADD THIS BACK

        # Lift check
        ee_diff  = robot_ee_z1 - robot_ee_z0
        obj_diff = obj_z1 - obj_z0
        lift_check = (torch.abs(ee_diff - obj_diff) <= 0.01) & \
            (torch.abs(ee_diff) >= 0.5 * lift_height) & \
            (torch.abs(obj_diff) >= 0.5 * lift_height)

        # Goal proximity check
        final_goal_matrices = self.obj_goal_traj_w[:, -1, :, :]
        goal_positions_np = final_goal_matrices[:, :3, 3]
        goal_positions = torch.tensor(goal_positions_np, dtype=torch.float32, device=self.sim.device)
        current_obj_pos = self.object_prim.data.root_state_w[:, :3]
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
        
        # ADD THESE DEBUG PRINTS BACK
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


    def grasp_trials(self, gg, std: float = 0.0005, max_retries: int = 3):
        """
        Async grasp trials: each env finds its own successful grasp independently.
        Environments that succeed early keep their grasp while failed envs continue searching.
        
        Returns dict with per-env success flags, chosen grasp parameters, and skip mask.
        Envs that exhaust retries are marked to skip in future processing.
        """
        B = self.scene.num_envs
        idx_all = list(range(len(gg)))
        if len(idx_all) == 0:
            print("[ERR] empty grasp list.")
            return {
                "success": np.zeros(B, dtype=bool),
                "chosen_pose_w": [None] * B,
                "chosen_pre": [None] * B,
                "chosen_delta": [None] * B,
                "skip_envs": np.zeros(B, dtype=bool),
            }

        rng = np.random.default_rng()
        pre_dist_const = 0.12  # m

        # Track per-environment success and chosen grasps
        env_success = np.zeros(B, dtype=bool)
        env_chosen_pose_w = [None] * B  # List of (pos, quat) tuples per env
        env_chosen_pre = np.zeros(B, dtype=np.float32)
        env_chosen_delta = np.zeros(B, dtype=np.float32)
        env_best_score = np.zeros(B, dtype=np.float32)
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
                    print(f"[SUCCESS] All active environments found valid grasps!")
                    if len(skipped_envs) > 0:
                        print(f"[INFO] Skipped {len(skipped_envs)} failed environments: {skipped_envs.tolist()}")
                    break
                
                # Get mask of envs that still need to find a grasp (not successful and not skipped)
                active_failed_mask = (~env_success) & (~env_skip)
                failed_env_ids = np.where(active_failed_mask)[0]
                n_failed = len(failed_env_ids)
                
                if n_failed == 0:
                    # All remaining envs are either successful or skipped
                    break
                
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
                                print(f"[SUCCESS] Env {b} found valid grasp (score: {score_batch[b]:.2f})")
                    
                    # Check if all active envs now succeeded
                    active_mask = ~env_skip
                    newly_successful = np.sum(env_success[active_failed_mask])
                    print(f"[SEARCH] block[{start}:{start+B}] -> {newly_successful}/{n_failed} previously-failed envs now successful")
                    
                    if np.all(env_success[active_mask]):
                        successful_envs = np.where(env_success & ~env_skip)[0]
                        skipped_envs = np.where(env_skip)[0]
                        print(f"[SUCCESS] All active environments found valid grasps on attempt {retry_attempt + 1}!")
                        if len(skipped_envs) > 0:
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
                    print(f"[WARN] {len(still_failed)} envs still need grasps after attempt {retry_attempt + 1}: {still_failed.tolist()}")
                    if retry_attempt < max_retries - 1:
                        print("[INFO] Retrying grasp selection for failed environments...")
                        continue
                    else:
                        # Last attempt exhausted - mark remaining failed envs as skipped
                        print(f"[WARN] Exhausted all {max_retries} grasp attempts")
                        print(f"[INFO] Marking {len(still_failed)} failed environments to skip: {still_failed.tolist()}")
                        env_skip[still_failed] = True
                        
                        successful_envs = np.where(env_success)[0]
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
                        if len(failed_envs) > 0:
                            print(f"[INFO] Marking {len(failed_envs)} failed environments to skip: {failed_envs.tolist()}")
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
                    if len(failed_envs) > 0:
                        print(f"[INFO] Marking {len(failed_envs)} failed environments to skip: {failed_envs.tolist()}")
                        env_skip[failed_envs] = True
                    return {
                        "success": env_success,
                        "chosen_pose_w": env_chosen_pose_w,
                        "chosen_pre": env_chosen_pre,
                        "chosen_delta": env_chosen_delta,
                        "skip_envs": env_skip,
                    }
        
        # Return final status
        return {
            "success": env_success,
            "chosen_pose_w": env_chosen_pose_w,
            "chosen_pre": env_chosen_pre,
            "chosen_delta": env_chosen_delta,
            "skip_envs": env_skip,
        }

    
    def is_success(self, position_threshold: float = 0.05) -> bool:
        """
        Verify if the manipulation task succeeded by comparing final object position
        with the goal position from the trajectory.
        
        Args:
            position_threshold: Distance threshold in meters (default: 0.05m = 5cm)
        
        Returns:
            bool: True if task succeeded, False otherwise
        """
        # Get the final goal position from the precomputed trajectory
        # self.obj_goal_traj_w has shape [B, T, 4, 4]
        B = self.scene.num_envs
        final_goal_matrices = self.obj_goal_traj_w[:, -1, :, :]  # [B, 4, 4] - last timestep
        
        # Extract goal positions (translation part of the transform matrix)
        goal_positions_np = final_goal_matrices[:, :3, 3]  # [B, 3] - xyz positions (numpy)
        
        # Convert to torch tensor
        goal_positions = torch.tensor(goal_positions_np, dtype=torch.float32, device=self.sim.device)
        
        # Get current object positions
        current_obj_pos = self.object_prim.data.root_state_w[:, :3]  # [B, 3]
        
        # Calculate distances
        distances = torch.norm(current_obj_pos - goal_positions, dim=1)  # [B]
        
        # Check if all environments succeeded
        success_mask = distances <= position_threshold
        
        # Print results for each environment
        print("\n" + "="*50)
        print("TASK VERIFICATION RESULTS")
        print("="*50)
        for b in range(B):
            status = "SUCCESS" if success_mask[b] else "FAILURE"
            print(f"Env {b}: {status} (distance: {distances[b].item():.4f}m, threshold: {position_threshold}m)")
            print(f"  Goal position: [{goal_positions[b, 0].item():.3f}, {goal_positions[b, 1].item():.3f}, {goal_positions[b, 2].item():.3f}]")
            print(f"  Final position: [{current_obj_pos[b, 0].item():.3f}, {current_obj_pos[b, 1].item():.3f}, {current_obj_pos[b, 2].item():.3f}]")
        
        # Overall result
        all_success = torch.all(success_mask).item()
        print("="*50)
        if all_success:
            print("TASK VERIFIER: SUCCESS - All environments completed successfully!")
        else:
            success_count = torch.sum(success_mask).item()
            print(f"TASK VERIFIER: FAILURE - {success_count}/{B} environments succeeded")
        print("="*50 + "\n")
        
        return all_success
    
    def save_data_selective(self, skip_envs: np.ndarray, ignore_keys: List[str] = []):
        """
        Save data only for non-skipped environments.
        
        Args:
            skip_envs: Boolean array [B] indicating which envs to skip
            ignore_keys: List of keys to ignore during save
        """
        save_root = self._demo_dir()
        save_root.mkdir(parents=True, exist_ok=True)

        stacked = {k: np.array(v) for k, v in self.save_dict.items()}
        
        active_envs = np.where(~skip_envs)[0]
        print(f"[INFO] Saving data for {len(active_envs)} active environments: {active_envs.tolist()}")

        for b in active_envs:
            env_dir = self._env_dir(save_root, b)
            for key, arr in stacked.items():
                if key in ignore_keys:
                    continue
                if key == "rgb":
                    video_path = env_dir / "sim_video.mp4"
                    writer = imageio.get_writer(
                        video_path, fps=50, macro_block_size=None
                    )
                    for t in range(arr.shape[0]):
                        writer.append_data(arr[t, b])
                    writer.close()
                elif key == "segmask":
                    video_path = env_dir / "mask_video.mp4"
                    writer = imageio.get_writer(
                        video_path, fps=50, macro_block_size=None
                    )
                    for t in range(arr.shape[0]):
                        writer.append_data((arr[t, b].astype(np.uint8) * 255))
                    writer.close()
                elif key == "depth":
                    depth_seq = arr[:, b]
                    flat = depth_seq[depth_seq > 0]
                    max_depth = np.percentile(flat, 99) if flat.size > 0 else 1.0
                    depth_norm = np.clip(depth_seq / max_depth * 255.0, 0, 255).astype(
                        np.uint8
                    )
                    video_path = env_dir / "depth_video.mp4"
                    writer = imageio.get_writer(
                        video_path, fps=50, macro_block_size=None
                    )
                    for t in range(depth_norm.shape[0]):
                        writer.append_data(depth_norm[t])
                    writer.close()
                    np.save(env_dir / f"{key}.npy", depth_seq)
                else:
                    np.save(env_dir / f"{key}.npy", arr[:, b])
            json.dump(self.sim_cfgs, open(env_dir / "config.json", "w"), indent=2)
        
        print("[INFO]: Demonstration is saved at: ", save_root)
        
        # Compose real videos only for active environments
        print("\n[INFO] Composing real videos with background...")
        for b in active_envs:
            print(f"[INFO] Processing environment {b}/{len(active_envs)}...")
            success = self.compose_real_video(env_id=b)
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


    def inference(self, std: float = 0.0, max_restart_attempts: int = 10, 
          max_grasp_retries: int = 1, regrasp_after_failures: int = 1) -> bool:
        """
        Main function with async per-env grasp selection and trajectory following.
        Each environment progresses independently through grasp selection, and only
        moves to trajectory following once ALL envs have found successful grasps.
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
        env_grasp_success = np.zeros(B, dtype=bool)
        env_traj_success = np.zeros(B, dtype=bool)
        env_consecutive_failures = np.zeros(B, dtype=int)
        env_skip = np.zeros(B, dtype=bool)  # Track envs to skip entirely
        
        total_grasp_attempts = 0
        max_total_grasp_attempts = 5
        
        # ========== OUTER LOOP: RE-GRASP IF NEEDED ==========
        while total_grasp_attempts < max_total_grasp_attempts:
            total_grasp_attempts += 1
            
            # ========== PHASE 1: ASYNC GRASP SELECTION ==========
            if self.grasp_idx >= 0:
                # Fixed grasp index mode - all envs use same grasp
                if self.grasp_idx >= len(gg):
                    print(f"[ERR] grasp_idx {self.grasp_idx} out of range [0,{len(gg)})")
                    return False
                print(f"[INFO] using fixed grasp index {self.grasp_idx} for all envs.")
                p_w, q_w = grasp_to_world(gg[int(self.grasp_idx)])
                
                # Prepare per-env format
                env_grasp_success = np.ones(B, dtype=bool)
                env_chosen_pose_w = [(p_w.astype(np.float32), q_w.astype(np.float32))] * B
                env_chosen_pre = np.full(B, self.grasp_pre if self.grasp_pre is not None else 0.12, dtype=np.float32)
                env_chosen_delta = np.full(B, self.grasp_delta if self.grasp_delta is not None else 0.0, dtype=np.float32)
                env_skip = np.zeros(B, dtype=bool)
            else:
                # Automatic async grasp selection
                if total_grasp_attempts > 1:
                    print(f"\n{'#'*60}")
                    print(f"RE-GRASP ATTEMPT {total_grasp_attempts}/{max_total_grasp_attempts}")
                    print(f"{'#'*60}\n")
                
                ret = self.grasp_trials(gg, std=std, max_retries=max_grasp_retries)
                
                env_grasp_success = ret["success"]
                env_chosen_pose_w = ret["chosen_pose_w"]
                env_chosen_pre = ret["chosen_pre"]
                env_chosen_delta = ret["chosen_delta"]
                env_skip = ret["skip_envs"]
            
            # Check if any active envs found grasps
            active_envs = np.where(~env_skip)[0]
            successful_envs = np.where(env_grasp_success & ~env_skip)[0]
            
            if len(active_envs) == 0:
                print("[ERR] All environments failed grasp selection and were skipped")
                return False
            
            if len(successful_envs) == 0:
                print(f"[ERR] None of the {len(active_envs)} active environments found grasps")
                return False
            
            print(f"\n[SUCCESS] {len(successful_envs)}/{len(active_envs)} active environments found valid grasps!")
            if len(successful_envs) < len(active_envs):
                skipped_envs = np.where(env_skip)[0]
                print(f"[INFO] {len(skipped_envs)} environments skipped: {skipped_envs.tolist()}")
            
            for b in successful_envs:
                print(f"  Env {b}: delta={env_chosen_delta[b]:.4f}m")
            
            # Reset trajectory failure counters for new grasps
            env_consecutive_failures[:] = 0
            env_traj_success[:] = False
            
            # ========== PHASE 2: TRAJECTORY FOLLOWING (with per-env restart) ==========
            for restart_attempt in range(max_restart_attempts):
                try:
                    print(f"\n{'='*60}")
                    print(f"TRAJECTORY EXECUTION ATTEMPT {restart_attempt + 1}/{max_restart_attempts}")
                    active_envs = np.where(~env_skip)[0]
                    successful_traj_envs = np.where(env_traj_success & ~env_skip)[0]
                    print(f"Active envs: {active_envs.tolist()}")
                    print(f"Successful envs: {len(successful_traj_envs)}/{len(active_envs)}")
                    print(f"{'='*60}\n")
                    
                    # If all active envs succeeded in trajectory, we're done
                    if np.all(env_traj_success[~env_skip]):
                        print(f"[SUCCESS] All {len(active_envs)} active environments completed trajectories!")
                        break
                    
                    # Prepare grasp info for ALL envs using their individual grasps
                    p_all = np.stack([pose[0] for pose in env_chosen_pose_w], axis=0)  # [B,3]
                    q_all = np.stack([pose[1] for pose in env_chosen_pose_w], axis=0)  # [B,4]
                    pre_all = env_chosen_pre  # [B]
                    del_all = env_chosen_delta  # [B]
                    info_all = self.build_grasp_info(p_all, q_all, pre_all, del_all)

                    # Reset and execute: open → pre → grasp → close → follow_object_goals
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
                    if jp is None or torch.any(success == False):
                        print("[WARN] Pre-grasp motion failed for some envs")
                        failed_envs = torch.where(success == False)[0].cpu().numpy() if success is not None else np.arange(B)
                        # Only increment failure counter for ACTIVE envs that failed
                        for b in failed_envs:
                            if not env_skip[b]:  # Only track failures for active envs
                                env_consecutive_failures[b] += 1
                        
                        # Check if any ACTIVE env needs re-grasp
                        active_need_regrasp = np.where((env_consecutive_failures >= regrasp_after_failures) & ~env_skip)[0]
                        if len(active_need_regrasp) > 0:
                            print(f"[INFO] Active envs {active_need_regrasp.tolist()} need re-grasping")
                            break
                        continue
                    
                    self.save_dict["actions"].append(np.concatenate([info_all["pre_p_b"].cpu().numpy(), info_all["pre_q_b"].cpu().numpy(), np.zeros((B, 1))], axis=1))
                    jp = self.wait(gripper_open=True, steps=3)

                    # Grasp
                    print("[INFO] Moving to grasp positions...")
                    jp, success = self.move_to(info_all["p_b"], info_all["q_b"], gripper_open=True)
                    if jp is None or torch.any(success == False):
                        print("[WARN] Grasp motion failed for some envs")
                        failed_envs = torch.where(success == False)[0].cpu().numpy() if success is not None else np.arange(B)
                        # Only increment failure counter for ACTIVE envs that failed
                        for b in failed_envs:
                            if not env_skip[b]:  # Only track failures for active envs
                                env_consecutive_failures[b] += 1
                        
                        # Check if any ACTIVE env needs re-grasp
                        active_need_regrasp = np.where((env_consecutive_failures >= regrasp_after_failures) & ~env_skip)[0]
                        if len(active_need_regrasp) > 0:
                            print(f"[INFO] Active envs {active_need_regrasp.tolist()} need re-grasping")
                            break
                        continue
                    
                    self.save_dict["actions"].append(np.concatenate([info_all["p_b"].cpu().numpy(), info_all["q_b"].cpu().numpy(), np.zeros((B, 1))], axis=1))

                    # Close gripper
                    print("[INFO] Closing grippers...")
                    jp = self.wait(gripper_open=False, steps=50)
                    self.save_dict["actions"].append(np.concatenate([info_all["p_b"].cpu().numpy(), info_all["q_b"].cpu().numpy(), np.ones((B, 1))], axis=1))

                    # Follow object trajectory
                    print("[INFO] Following object trajectories...")
                    jp = self.follow_object_goals(jp, sample_step=5, visualize=True)
                    
                    # If we reach here, trajectory following succeeded for all!
                    print("[SUCCESS] All trajectories completed successfully!")
                    env_traj_success[:] = True
                    
                    # Verify task completion (only for active envs)
                    task_success = self.is_success(position_threshold=0.1)
                    if task_success:
                        print(f"[SUCCESS] Task verification passed for all {len(active_envs)} active environments!")
                        self.save_data_selective(env_skip)
                        return True
                    else:
                        print("[WARN] Task verification failed for some environments")
                        self.clear_data()
                        
                except RuntimeError as e:
                    if "MotionPlanningFailure_RestartNeeded" in str(e):
                        print(f"\n[RESTART] Motion planning failed mid-trajectory on attempt {restart_attempt + 1}")
                        print("[RESTART] Object was likely dropped. Restarting from grasp phase...")
                        
                        # Increment failure counter ONLY for ACTIVE envs that haven't succeeded
                        for b in range(B):
                            if not env_traj_success[b] and not env_skip[b]:  # Only active, unsuccessful envs
                                env_consecutive_failures[b] += 1
                        
                        self.clear_data()
                        
                        # Check if any ACTIVE env needs re-grasp
                        active_need_regrasp = np.where((env_consecutive_failures >= regrasp_after_failures) & ~env_skip)[0]
                        if len(active_need_regrasp) > 0:
                            print(f"\n{'!'*60}")
                            print(f"Active envs {active_need_regrasp.tolist()} need re-grasping")
                            print(f"RE-SELECTING GRASPS (total attempt {total_grasp_attempts + 1}/{max_total_grasp_attempts})")
                            print(f"{'!'*60}\n")
                            break
                        
                        continue
                    else:
                        raise
                
                except Exception as e:
                    print(f"[ERROR] Unexpected error during execution: {type(e).__name__}: {e}")
                    
                    # Increment failure counter ONLY for ACTIVE envs that haven't succeeded
                    for b in range(B):
                        if not env_traj_success[b] and not env_skip[b]:  # Only active, unsuccessful envs
                            env_consecutive_failures[b] += 1
                    
                    self.clear_data()
                    
                    # Check if any ACTIVE env needs re-grasp
                    active_need_regrasp = np.where((env_consecutive_failures >= regrasp_after_failures) & ~env_skip)[0]
                    if len(active_need_regrasp) > 0:
                        print(f"\n{'!'*60}")
                        print(f"Active envs {active_need_regrasp.tolist()} need re-grasping")
                        print(f"RE-SELECTING GRASPS (total attempt {total_grasp_attempts + 1}/{max_total_grasp_attempts})")
                        print(f"{'!'*60}\n")
                        break
                    
                    continue
            
            # Check why we exited the trajectory loop
            active_need_regrasp = np.where((env_consecutive_failures >= regrasp_after_failures) & ~env_skip)[0]
            if len(active_need_regrasp) > 0:
                # Some ACTIVE envs need new grasps
                if total_grasp_attempts >= max_total_grasp_attempts:
                    print(f"\n[FAILURE] Active envs {active_need_regrasp.tolist()} exhausted all grasp attempts ({max_total_grasp_attempts})")
                    return False
                # Continue outer loop to select new grasps for failed envs
                continue
            else:
                # We exhausted all restart attempts
                print(f"\n[FAILURE] Failed after {max_restart_attempts} trajectory restart attempts")
                return False
        
        # If we exit the outer loop, we exhausted all grasp attempts
        print(f"\n[FAILURE] Failed after {max_total_grasp_attempts} total grasp selection attempts")
        return False
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