"""
Heuristic manipulation policy in Isaac Lab simulation.
Using grasping and motion planning to perform object manipulation tasks.
"""
from __future__ import annotations

# ─────────── AppLauncher ───────────
import argparse, os, json, random, transforms3d, typing
from typing import Optional
from pathlib import Path
import numpy as np
import torch
import yaml
from isaaclab.app import AppLauncher
import sys
file_path = Path(__file__).resolve()
sys.path.append(str(file_path.parent))
sys.path.append(str(file_path.parent.parent))
from envs.task_cfg import TaskCfg, TaskType, SuccessMetric, SuccessMetricType, TrajectoryCfg, RobotType
from envs.task_construct import construct_task_config, add_reference_trajectory, load_task_cfg
from envs.running_cfg import get_heuristic_config


# ─────────── CLI ───────────
parser = argparse.ArgumentParser("sim_policy")
parser.add_argument("--key", type=str, default="demo_video", help="scene key (outputs/<key>)")
parser.add_argument("--robot", type=str, default="franka")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments (overrides running_cfg)")
parser.add_argument("--num_trials", type=int, default=None, help="Number of trials (overrides running_cfg)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.headless = True  # headless mode for batch execution
app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

# ─────────── Runtime imports ───────────
import isaaclab.sim as sim_utils
from isaaclab.utils.math import subtract_frame_transforms

from sim_utils.grasp_group_utils import GraspGroup


# ─────────── Simulation environments ───────────
from sim_base import BaseSimulator, get_next_demo_id
from sim_env_factory import make_env

from sim_utils.transform_utils import pose_to_mat, mat_to_pose, grasp_to_world, grasp_approach_axis_batch
from sim_utils.sim_utils import load_sim_parameters




# ──────────────────────────── Heuristic Manipulation ────────────────────────────
class HeuristicManipulation(BaseSimulator):
    """
    Physical trial-and-error grasping with approach-axis perturbation:
      • Multiple grasp proposals executed in parallel;
      • Every attempt does reset → pre → grasp → close → lift → check;
      • Early stops when any env succeeds; then re-exec for logging.
    """
    def __init__(self, sim, scene, sim_cfgs: dict, args, out_dir: Path, img_folder: str, data_dir: Path):
        robot_pose = torch.tensor(
            sim_cfgs["robot_cfg"]["robot_pose"],
            dtype=torch.float32,
            device=sim.device
        )
        super().__init__(
            sim=sim, sim_cfgs=sim_cfgs, scene=scene, args=args_cli,
            robot_pose=robot_pose, cam_dict=sim_cfgs["cam_cfg"],
            out_dir=out_dir, img_folder=img_folder, data_dir = data_dir,
            enable_motion_planning=True,
            set_physics_props=True, debug_level=0,
        )
        self.final_gripper_closed = sim_cfgs["demo_cfg"]["final_gripper_closed"]
        self.selected_object_id = sim_cfgs["demo_cfg"]["manip_object_id"]
        self.traj_path = sim_cfgs["demo_cfg"]["traj_path"]
        self.goal_offset = [0, 0, sim_cfgs["demo_cfg"]["goal_offset"]]
        self.grasp_path = sim_cfgs["demo_cfg"]["grasp_path"]
        self.grasp_idx = sim_cfgs["demo_cfg"]["grasp_idx"]
        self.grasp_pre = sim_cfgs["demo_cfg"]["grasp_pre"]
        self.grasp_delta = sim_cfgs["demo_cfg"]["grasp_delta"]
        self.task_type = sim_cfgs["demo_cfg"]["task_type"]
        self.robot_type = args.robot
        self.load_obj_goal_traj()

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
        # —— 1) Load Δ_w ——
        rel = np.load(self.traj_path).astype(np.float32)
        self.obj_rel_traj = rel[1:, :, :]  # (T,4,4)

        self.reset()

        # —— 2) Read current object initial pose per env as T_obj_init ——
        B = self.scene.num_envs
        # obj_state = self.object_prim.data.root_com_state_w[:, :7]  # [B,7], pos(3)+quat(wxyz)(4)
        obj_state = self.object_prim.data.root_state_w[:, :7]  # [B,7], pos(3)+quat(wxyz)(4)
        self.show_goal(obj_state[:, :3], obj_state[:, 3:7])

        obj_state_np = obj_state.detach().cpu().numpy().astype(np.float32)
        offset_np = np.asarray(self.goal_offset, dtype=np.float32).reshape(3)
        obj_state_np[:, :3] += offset_np  # raise a bit to avoid collision

        # Note: here the relative traj Δ_w is defined in world frame with origin (0,0,0),
        # Hence, we need to normalize it to each env's origin frame.
        origins = self.scene.env_origins.detach().cpu().numpy().astype(np.float32)  # (B,3)

        obj_state_np[:, :3] -= origins # normalize to env origin frame

        # —— 3) Precompute absolute object goals for all envs ——
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
        follow precompute object absolute trajectory: self.obj_goal_traj_w:
          T_obj_goal[t] = Δ_w[t] @ T_obj_init

        EE-object transform is fixed:
          T_ee_goal[t] = T_obj_goal[t] @ (T_obj_grasp^{-1} @ T_ee_grasp)
        Here T_obj_grasp / T_ee_grasp is the transform at the moment of grasping.
        """

        B = self.scene.num_envs
        obj_goal_all = self.obj_goal_traj_w  # [B, T, 4, 4]
        T = obj_goal_all.shape[1]

        ee_w  = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]  # [B,7]
        # obj_w = self.object_prim.data.root_com_state_w[:, :7]                                 # [B,7]
        obj_w = self.object_prim.data.root_state_w[:, :7]                                 # [B,7]

        T_ee_ws = []
        T_obj_ws = []
        for b in range(B):
            T_ee_w  = pose_to_mat(ee_w[b, :3],  ee_w[b, 3:7])
            T_obj_w = pose_to_mat(obj_w[b, :3], obj_w[b, 3:7])
            T_ee_ws.append(T_ee_w)
            T_obj_ws.append(T_obj_w)

        joint_pos = start_joint_pos
        root_w = self.robot.data.root_state_w[:, 0:7]  # robot base poses per env

        t_iter = list(range(0, T, sample_step))
        t_iter = t_iter + [T-1] if t_iter[-1] != T-1 else t_iter

        for t in t_iter:
            goal_pos_list, goal_quat_list = [], []
            print(f"[INFO] follow object goal step {t}/{T}")
            for b in range(B):
                T_obj_goal = obj_goal_all[b, t]            # (4,4)
                trans_offset = T_obj_goal - T_obj_ws[b]
                T_ee_goal  = T_ee_ws[b] + trans_offset
                original_R = T_obj_ws[b][:3, :3]
                new_R = T_ee_goal[:3, :3]
                original_ee_R = T_ee_ws[b][:3, :3]
                new_ee_R = new_R @ np.linalg.inv(original_R) @ original_ee_R
                T_ee_goal[:3, :3] = new_ee_R
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
            self.save_dict["actions"].append(np.concatenate([ee_pos_b.cpu().numpy(), ee_quat_b.cpu().numpy(), np.ones((B, 1))], axis=1))

        success_ids = self.is_success()
        joint_pos = self.wait(gripper_open=not self.final_gripper_closed, steps=30)
        return joint_pos, success_ids
    
    def follow_object_relative_goals(self, start_joint_pos, sample_step=1, visualize=True):
        B = self.scene.num_envs
        obj_goal_all = self.obj_goal_traj_w  # [B, T, 4, 4]
        T = obj_goal_all.shape[1]

        ee_w  = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]  # [B,7]
        # obj_w = self.object_prim.data.root_com_state_w[:, :7]                                 # [B,7]
        current_ee_pos_w = ee_w[:, :3]
        current_ee_quat_w = ee_w[:, 3:7]

        joint_pos = start_joint_pos
        root_w = self.robot.data.root_state_w[:, 0:7]  # robot base poses per env

        t_iter = list(range(0, T, sample_step))
        t_iter = t_iter + [T-1] if t_iter[-1] != T-1 else t_iter

        for t in t_iter:
            goal_pos_list, goal_quat_list = [], []
            print(f"[INFO] follow object goal step {t}/{T}")
            for b in range(B):
                         # (4,4)
                current_T_ee = pose_to_mat(current_ee_pos_w[b], current_ee_quat_w[b])
                T_ee_goal  = self.obj_rel_traj[t] @ current_T_ee
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
            current_ee_pos_w = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:3]
            current_ee_quat_w = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 3:7]

            self.save_dict["actions"].append(np.concatenate([ee_pos_b.cpu().numpy(), ee_quat_b.cpu().numpy(), np.ones((B, 1))], axis=1))

        success_ids = self.is_success()
        joint_pos = self.wait(gripper_open=not self.final_gripper_closed, steps=30)
        return joint_pos, success_ids
    

    def follow_object_centers(self, start_joint_pos, sample_step=1, visualize=True):
        B = self.scene.num_envs
        obj_goal_all = self.obj_goal_traj_w  # [B, T, 4, 4]
        T = obj_goal_all.shape[1]

        ee_w  = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]  # [B,7]
        # obj_w = self.object_prim.data.root_com_state_w[:, :7]                                 # [B,7]
        obj_w = self.object_prim.data.root_state_w[:, :7]                                 # [B,7]

        T_ee_ws = []
        T_obj_ws = []
        for b in range(B):
            T_ee_w  = pose_to_mat(ee_w[b, :3],  ee_w[b, 3:7])
            T_obj_w = pose_to_mat(obj_w[b, :3], obj_w[b, 3:7])
            T_ee_ws.append(T_ee_w)
            T_obj_ws.append(T_obj_w)

        joint_pos = start_joint_pos
        root_w = self.robot.data.root_state_w[:, 0:7]  # robot base poses per env

        t_iter = list(range(0, T, sample_step))
        t_iter = t_iter + [T-1] if t_iter[-1] != T-1 else t_iter

        for t in t_iter:
            goal_pos_list, goal_quat_list = [], []
            print(f"[INFO] follow object goal step {t}/{T}")
            for b in range(B):
                T_obj_goal = obj_goal_all[b, t]            # (4,4)
                trans_offset = T_obj_goal - T_obj_ws[b]
                T_ee_goal  = T_ee_ws[b] + trans_offset
                pos_b, quat_b = mat_to_pose(T_ee_goal)

                goal_pos_list.append(pos_b.astype(np.float32))
                goal_quat_list.append(quat_b.astype(np.float32))

            print()

            goal_pos  = torch.as_tensor(np.stack(goal_pos_list),  dtype=torch.float32, device=self.sim.device)
            goal_quat = ee_w[:, 3:7]

            if visualize:
                self.show_goal(goal_pos, goal_quat)
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_w[:, :3], root_w[:, 3:7], goal_pos, goal_quat
            )
            joint_pos, success = self.move_to(ee_pos_b, ee_quat_b, gripper_open=False)

            print(obj_goal_all[:,t])
            print(self.object_prim.data.root_state_w[:, :7])
            self.save_dict["actions"].append(np.concatenate([ee_pos_b.cpu().numpy(), ee_quat_b.cpu().numpy(), np.ones((B, 1))], axis=1))
        is_success = self.is_success()
        joint_pos = self.wait(gripper_open=not self.final_gripper_closed, steps=30)
        return joint_pos, is_success



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
        """World → robot base frame for all envs."""
        root = self.robot.data.root_state_w[:, 0:7]  # [B,7]
        p_w, q_w = self._ensure_batch_pose(pos_w, quat_w)
        pb, qb = subtract_frame_transforms(
            root[:, 0:3], root[:, 3:7], p_w, q_w
        )
        return pb, qb  # [B,3], [B,4]

    # ---------- Batched execution & lift-check ----------
    def execute_and_lift_once_batch(self, info: dict, lift_height=0.12) -> tuple[np.ndarray, np.ndarray]:
        """
        Reset → pre → grasp → close → lift → hold; return (success[B], score[B]).
        """
        B = self.scene.num_envs
        self.reset()

        # open gripper buffer
        jp = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
        self.wait(gripper_open=True, steps=4)

        # pre-grasp
        jp, success = self.move_to(info["pre_p_b"], info["pre_q_b"], gripper_open=True)
        if torch.any(success==False): return np.zeros(B, bool), np.zeros(B, np.float32)
        jp = self.wait(gripper_open=True, steps=3)

        # grasp
        jp, success = self.move_to(info["p_b"], info["q_b"], gripper_open=True)
        if torch.any(success==False): return np.zeros(B, bool), np.zeros(B, np.float32)
        jp = self.wait(gripper_open=True, steps=2)

        # close
        jp = self.wait(gripper_open=False, steps=6)

        # initial heights
        obj0 = self.object_prim.data.root_com_pos_w[:, 0:3]     # [B,3]
        ee_w = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]  # [B,7]
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
        if torch.any(success==False): return np.zeros(B, bool), np.zeros(B, np.float32)
        jp = self.wait(gripper_open=False, steps=8)

        # final heights
        obj1 = self.object_prim.data.root_com_pos_w[:, 0:3]
        ee_w1 = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        robot_ee_z1 = ee_w1[:, 2]
        obj_z1 = obj1[:, 2]
        print(f"[INFO] mean object z1={obj_z1.mean().item():.3f} m, mean EE z1={robot_ee_z1.mean().item():.3f} m")

        # lifted if EE and object rise similarly (tight coupling)
        ee_diff  = robot_ee_z1 - robot_ee_z0
        obj_diff = obj_z1 - obj_z0
        lifted = (torch.abs(ee_diff - obj_diff) <= 0.01) & \
            (torch.abs(ee_diff) >= 0.5 * lift_height) & \
            (torch.abs(obj_diff) >= 0.5 * lift_height)  # [B] bool

        score = torch.zeros_like(ee_diff)
        score[lifted] = 1000.0
        return lifted.detach().cpu().numpy().astype(bool), score.detach().cpu().numpy().astype(np.float32)

    def lift_up(self, height=0.12, gripper_open=False, steps=8):
        """
        Lift up by a certain height (m) from current EE pose.
        """
        ee_w = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        target_p = ee_w[:, :3].clone()
        target_p[:, 2] += height

        root = self.robot.data.root_state_w[:, 0:7]
        p_lift_b, q_lift_b = subtract_frame_transforms(
            root[:, 0:3], root[:, 3:7],
            target_p, ee_w[:, 3:7]
        ) # [B,3], [B,4]  
        jp, success = self.move_to(p_lift_b, q_lift_b, gripper_open=gripper_open)
        jp = self.wait(gripper_open=gripper_open, steps=steps)
        return jp
 
    def is_success(self) -> torch.Tensor:
        ee_w  = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        obj_w = self.object_prim.data.root_com_pos_w[:, 0:3]
        dist = torch.norm(obj_w[:, :3] - ee_w[:, :3], dim=1) # [B]
        return (dist < 0.15).to(torch.bool)

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


    def grasp_trials(self, gg, std: float = 0.0005):

        B = self.scene.num_envs
        idx_all = list(range(len(gg)))
        if len(idx_all) == 0:
            print("[ERR] empty grasp list.")
            return False

        rng = np.random.default_rng()

        pre_dist_const = 0.12  # m

        success = False
        chosen_pose_w = None    # (p_w, q_w)
        chosen_pre    = None
        chosen_delta  = None

        # assign different grasp proposals to different envs
        for start in range(0, len(idx_all), B):
            block = idx_all[start : start + B]
            if len(block) < B:
                block = block + [block[-1]] * (B - len(block))

            grasp_pos_w_batch, grasp_quat_w_batch = [], []
            for idx in block:
                p_w, q_w = grasp_to_world(gg[int(idx)])
                grasp_pos_w_batch.append(p_w.astype(np.float32))
                grasp_quat_w_batch.append(q_w.astype(np.float32))
            grasp_pos_w_batch  = np.stack(grasp_pos_w_batch,  axis=0)  # (B,3)
            grasp_quat_w_batch = np.stack(grasp_quat_w_batch, axis=0)  # (B,4)
            self.show_goal(grasp_pos_w_batch, grasp_quat_w_batch)
            # random disturbance along approach axis
            pre_dist_batch = np.full((B,), pre_dist_const, dtype=np.float32)
            delta_batch    = rng.normal(-0.002, std, size=(B,)).astype(np.float32)

            info = self.build_grasp_info(grasp_pos_w_batch, grasp_quat_w_batch,
                                          pre_dist_batch, delta_batch)

            ok_batch, score_batch = self.execute_and_lift_once_batch(info)
            ok_cnt = int(ok_batch.sum())
            print(f"[SEARCH] block[{start}:{start+B}] -> success {ok_cnt}/{B}")
            if ok_cnt > 0:
                winner = int(np.argmax(score_batch))
                chosen_pose_w = (grasp_pos_w_batch[winner], grasp_quat_w_batch[winner])
                chosen_pre    = float(pre_dist_batch[winner])
                chosen_delta  = float(delta_batch[winner])
                success = True
                return {
                    "success": success,
                    "chosen_pose_w": chosen_pose_w,
                    "chosen_pre": chosen_pre,
                    "chosen_delta": chosen_delta,
                }

        if not success:
            print("[ERR] no proposal succeeded to lift after full search.")
            return {
                "success": success,
                "chosen_pose_w": None,
                "chosen_pre": None,
                "chosen_delta": None,
            }

    def replay_actions(self, actions: np.ndarray):
        """
        Replay a sequence of recorded actions: (p[B,3], q[B,4], gripper[B,1])
        """
        n_steps = actions.shape[0]

        self.reset()
        self.wait(gripper_open=True, steps=10)

        for t in range(n_steps):
            print(f"[INFO] replay step {t}/{n_steps}")
            act = actions[t:t+1]
            p_b = torch.as_tensor(act[:, 0:3], dtype=torch.float32, device=self.sim.device)
            q_b = torch.as_tensor(act[:, 3:7], dtype=torch.float32, device=self.sim.device)
            g_b = act[:, 7] < 0.5
            jp, success = self.move_to(p_b, q_b, gripper_open=g_b)
            if torch.any(success==False):
                print(f"[ERR] replay step {t} failed.")
                return False
            jp = self.wait(gripper_open=g_b, steps=3)
        return True

    def inference(self, std: float = 0.0) -> list[int]:
        """
        Main function of the heuristic manipulation policy.
        Physical trial-and-error grasping with approach-axis perturbation:
          • Multiple grasp proposals executed in parallel;
          • Every attempt does reset → pre → grasp → close → lift → check;
          • Early stops when any env succeeds; then re-exec for logging.
        """
        B = self.scene.num_envs

        self.wait(gripper_open=True, steps=10)

        # read grasp proposals
        npy_path = self.grasp_path
        if npy_path is None or (not os.path.exists(npy_path)):
            print(f"[ERR] grasps npy not found: {npy_path}")
            return []
        gg = GraspGroup().from_npy(npy_file_path=npy_path)


        if self.grasp_idx >= 0:
            if self.grasp_idx >= len(gg):
                print(f"[ERR] grasp_idx {self.grasp_idx} out of range [0,{len(gg)})")
                return []
            print(f"[INFO] using fixed grasp index {self.grasp_idx} for all envs.")
            p_w, q_w = grasp_to_world(gg[int(self.grasp_idx)])
            ret = {
                "success": True,
                "chosen_pose_w": (p_w.astype(np.float32), q_w.astype(np.float32)),
                "chosen_pre": self.grasp_pre if self.grasp_pre is not None else 0.12,
                "chosen_delta": self.grasp_delta if self.grasp_delta is not None else 0.0,
            }
            print(f"[INFO] grasp delta (m): {ret['chosen_delta']:.4f}")
        else:
            ret = self.grasp_trials(gg, std=std)

        print("[INFO] Re-exec all envs with the winning grasp, then follow object goals.")
        if ret is None or ret["success"] == False:
            print("[ERR] no proposal succeeded to lift after full search.")
            return []
        p_win, q_win = ret["chosen_pose_w"]
        p_all   = np.repeat(p_win.reshape(1, 3), B, axis=0)
        q_all   = np.repeat(q_win.reshape(1, 4), B, axis=0)
        pre_all = np.full((B,), ret["chosen_pre"],   dtype=np.float32)
        del_all = np.full((B,), ret["chosen_delta"], dtype=np.float32)

        info_all = self.build_grasp_info(p_all, q_all, pre_all, del_all)

        # reset and conduct main process: open→pre→grasp→close→follow_object_goals
        self.reset()
        #print(self.object_prim.data.root_state_w[:, :7].cpu().numpy())
        cam_p = self.camera.data.pos_w
        cam_q = self.camera.data.quat_w_ros
        gp_w  = torch.as_tensor(info_all["p_w"],     dtype=torch.float32, device=self.sim.device)
        gq_w  = torch.as_tensor(info_all["q_w"],     dtype=torch.float32, device=self.sim.device)
        pre_w = torch.as_tensor(info_all["pre_p_w"], dtype=torch.float32, device=self.sim.device)
        gp_cam,  gq_cam  = subtract_frame_transforms(cam_p, cam_q, gp_w,  gq_w)
        pre_cam, pre_qcm = subtract_frame_transforms(cam_p, cam_q, pre_w, gq_w)
       
        self.save_dict["grasp_pose_w"] = torch.cat([gp_w,  gq_w],  dim=1).cpu().numpy()
        self.save_dict["pregrasp_pose_w"] = torch.cat([pre_w, gq_w], dim=1).cpu().numpy()
        self.save_dict["grasp_pose_cam"]    = torch.cat([gp_cam,  gq_cam],  dim=1).cpu().numpy()
        self.save_dict["pregrasp_pose_cam"] = torch.cat([pre_cam, pre_qcm], dim=1).cpu().numpy()

        jp = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
        self.wait(gripper_open=True, steps=4)

        # pre → grasp
        jp, success = self.move_to(info_all["pre_p_b"], info_all["pre_q_b"], gripper_open=True)
        if torch.any(success==False): return []
        self.save_dict["actions"].append(np.concatenate([info_all["pre_p_b"].cpu().numpy(), info_all["pre_q_b"].cpu().numpy(), np.zeros((B, 1))], axis=1))
        jp = self.wait(gripper_open=True, steps=3)

        jp, success = self.move_to(info_all["p_b"], info_all["q_b"], gripper_open=True)
        if torch.any(success==False): return []
        self.save_dict["actions"].append(np.concatenate([info_all["p_b"].cpu().numpy(), info_all["q_b"].cpu().numpy(), np.zeros((B, 1))], axis=1))

        # close gripper
        jp = self.wait(gripper_open=False, steps=50)
        self.save_dict["actions"].append(np.concatenate([info_all["p_b"].cpu().numpy(), info_all["q_b"].cpu().numpy(), np.ones((B, 1))], axis=1))

        # object goal following
        print(f"[INFO] lifting up by {self.goal_offset[2]} meters")
        self.lift_up(height=self.goal_offset[2], gripper_open=False, steps=8)
        #jp = self.follow_object_goals(jp, sample_step=5, visualize=True)
        if self.task_type == "simple_pick_place" or self.task_type == "simple_pick":
            jp, success_ids = self.follow_object_centers(jp, sample_step=1, visualize=True)
        elif self.task_type == "targetted_pick_place":
            jp, success_ids = self.follow_object_goals(jp, sample_step=1, visualize=True)
        else:
            raise ValueError(f"[ERR] Invalid task type: {self.task_type}")
       
        object_prim_world_pose = self.object_prim.data.root_state_w[:, :7].cpu().numpy()
        object_prim_world_pose[:, :3] = object_prim_world_pose[:, :3] - self.scene.env_origins.cpu().numpy()
        self.save_dict["final_object_world_pose"] = object_prim_world_pose
        
        robot_world_pose = self.robot.data.root_state_w[:, :7].cpu().numpy()
        robot_world_pose[:, :3] = robot_world_pose[:, :3] - self.scene.env_origins.cpu().numpy()
        self.save_dict["final_robot_world_pose"] = robot_world_pose

        
    
        # Properly handle the case when success_ids is a numpy array
        # Convert it to a torch tensor if needed, before calling torch.where
        print(f"[INFO] success_ids: {success_ids}")
        # If success_ids is already a tensor, we keep as-is
        success_ids = torch.where(success_ids)[0].cpu().numpy()
      
    
        return success_ids
           

    def from_data_to_task_cfg(self, key:str) -> TaskCfg:
        BASE_DIR = Path.cwd()
        scene_json_path = BASE_DIR / "outputs" / key / "simulation" / "scene.json"
        task_base_folder = BASE_DIR / "tasks"
        scene_dict = json.load(open(scene_json_path))
        task_cfg, base_folder = construct_task_config(key, scene_dict, task_base_folder)
        robot_pose = self.save_dict["final_robot_world_pose"]
        robot_pose = robot_pose[0]
        robot_pose = robot_pose.tolist()

        object_trajectory = np.load(self.traj_path).astype(np.float32)
        pose_quat_traj = []
        for pose_mat in object_trajectory:
            pose, quat = mat_to_pose(pose_mat)
            pose_quat = np.concatenate([np.array(pose), np.array(quat)])
            pose_quat_traj.append(pose_quat)
        pose_quat_traj = np.array(pose_quat_traj).reshape(-1, 7).tolist()
                               
        # Convert pregrasp and grasp poses from world to env-local frame
        pregrasp_pose_world = np.array(self.save_dict["pregrasp_pose_w"])  # [B, 7]
        grasp_pose_world = np.array(self.save_dict["grasp_pose_w"])        # [B, 7]
        env_origins = self.scene.env_origins.cpu().numpy()       # [B, 3]
        
        pregrasp_pose_local = np.array(pregrasp_pose_world.copy())
        pregrasp_pose_local[:, :3] = pregrasp_pose_world[:, :3] - env_origins
        pregrasp_pose = pregrasp_pose_local[0].tolist()  # Take first env
        
        grasp_pose_local = grasp_pose_world.copy()
        grasp_pose_local[:, :3] = grasp_pose_world[:, :3] - env_origins
        grasp_pose = grasp_pose_local[0].tolist()  # Take first env
      
        final_gripper_close = self.final_gripper_closed
        
        if task_cfg.task_type == TaskType.TARGETTED_PICK_PLACE:
            success_metric = SuccessMetric(
                success_metric_type = SuccessMetricType.TARGET_POINT,
                end_pose  = self.save_dict["final_object_world_pose"][0].tolist(),
                final_gripper_close = final_gripper_close,
            )
        elif task_cfg.task_type == TaskType.SIMPLE_PICK_PLACE:
            final_object_world_pose = self.save_dict["final_object_world_pose"][0]
            ground_value = float(final_object_world_pose[2])
            success_metric = SuccessMetric(
                success_metric_type = SuccessMetricType.TARGET_PLANE,
                ground_value = ground_value,
                final_gripper_close = final_gripper_close,
                end_pose  = self.save_dict["final_object_world_pose"][0].tolist()
            )
        else:
            success_metric = SuccessMetric(
                success_metric_type = SuccessMetricType.SIMPLE_LIFT,
                lift_height = 0.05,
                final_gripper_close = final_gripper_close,
                end_pose  = self.save_dict["final_object_world_pose"][0].tolist()
            )
        object_poses = {}
        for obj in task_cfg.objects:
            object_poses[obj.object_id] = [0, 0, 0, 1, 0, 0, 0]
        
        if self.robot_type == 'franka':
            robot_type = RobotType.FRANKA
        elif self.robot_type == 'ur5':
            robot_type = RobotType.UR5
        else:
            raise ValueError(f"[ERR] Invalid robot type: {self.robot_type}")
        trajectory_cfg = TrajectoryCfg(
            robot_pose = robot_pose,
            object_poses = object_poses,
            object_trajectory = pose_quat_traj,
            final_gripper_close = final_gripper_close,
            success_metric = success_metric,
            pregrasp_pose = pregrasp_pose,
            grasp_pose = grasp_pose,
            robot_type = robot_type,
        )
        add_reference_trajectory(task_cfg, trajectory_cfg, base_folder)
        return task_cfg


# ─────robot_pose─────────────────────── Entry Point ────────────────────────────




# def main():
#     sim_cfgs = load_sim_parameters(BASE_DIR, args_cli.key)
#     env, _ = make_env(
#         cfgs=sim_cfgs, num_envs=args_cli.num_envs,
#         device=args_cli.device,
#         bg_simplify=False,
#     )
#     sim, scene = env.sim, env.scene

#     my_sim = HeuristicManipulation(
#         sim, scene, sim_cfgs=sim_cfgs, 
#         args=args_cli, out_dir=out_dir, img_folder=img_folder
#     )

#     demo_root = (out_dir / img_folder / "demos").resolve()

#     for _ in range(args_cli.num_trials):

#         robot_pose = torch.tensor(sim_cfgs["robot_cfg"]["robot_pose"], dtype=torch.float32, device=my_sim.sim.device)  # [7], pos(3)+quat(wxyz)(4)
#         my_sim.set_robot_pose(robot_pose)
#         my_sim.demo_id = get_next_demo_id(demo_root)
#         my_sim.reset()
#         print(f"[INFO] start simulation demo_{my_sim.demo_id}")
#         # Note: if you set viz_object_goals(), remember to disable gravity and collision for object
#         # my_sim.viz_object_goals(sample_step=10, hold_steps=40)
#         success_ids = my_sim.inference()
#         print(f"[INFO] success_ids: {success_ids}")
#         if success_ids is not None and len(success_ids) > 0:
#             success = True
#             my_sim.from_data_to_task_cfg(args_cli.key)
#             break
#         else:
#             success = False
        
#         # actions = np.load("outputs/lab16/demos/demo_5/env_000/actions.npy")
#         # my_sim.replay_actions(actions)

#     env.close()
#     simulation_app.close()





def sim_heuristic_manip(keys: list[str], args_cli: argparse.Namespace, config_path: Optional[str] = None, config_dict: Optional[dict] = None):
    """`
    Run heuristic manipulation simulation.
    
    Args:
        keys: List of scene keys to run (e.g., ["demo_video"])
        args_cli: Command-line arguments (argparse.Namespace)
        config_path: Path to config file (yaml/json) - alternative to args_cli
        config_dict: Config dictionary - alternative to args_cli
        
    Usage:
        # Option 1: Use command-line args (original)
        sim_heuristic_manip(["demo_video"], args_cli)
        
        # Option 2: Use config file
        sim_heuristic_manip(["demo_video"], config_path="config.yaml")
        
        # Option 3: Use config dict
        sim_heuristic_manip(["demo_video"], config_dict={"num_envs": 4, "num_trials": 10})
    """
    # Create args from config if not provided
    # if args_cli is None:    
    #     args_cli = create_args_from_config(config_path=config_path, config_dict=config_dict)
    BASE_DIR   = Path.cwd()
    

    out_dir    = BASE_DIR / "outputs"
    local_out_dir = BASE_DIR / "outputs"
    
    for key in keys:
        args_cli.key = key
        local_img_folder = key
        data_dir = BASE_DIR / "h5py" / key 
        
        # Load config from running_cfg, allow CLI args to override
        heuristic_cfg = get_heuristic_config(key)
        num_envs = args_cli.num_envs if args_cli.num_envs is not None else heuristic_cfg.num_envs
        num_trials = args_cli.num_trials if args_cli.num_trials is not None else heuristic_cfg.num_trials
        
        print(f"[INFO] Using config for key '{key}': num_envs={num_envs}, num_trials={num_trials}")
        
        sim_cfgs = load_sim_parameters(BASE_DIR, key)
        env, _ = make_env(
            cfgs=sim_cfgs, num_envs=num_envs,
            device=args_cli.device,
            bg_simplify=False,
        )
        sim, scene = env.sim, env.scene

        my_sim = HeuristicManipulation(
            sim, scene, sim_cfgs=sim_cfgs,
            args=args_cli, out_dir=local_out_dir, img_folder=local_img_folder,
            data_dir = data_dir )

        demo_root = (local_out_dir / key / "demos").resolve()
        success = False
        for i in range(num_trials):

            robot_pose = torch.tensor(sim_cfgs["robot_cfg"]["robot_pose"], dtype=torch.float32, device=my_sim.sim.device)  # [7], pos(3)+quat(wxyz)(4)
            my_sim.set_robot_pose(robot_pose)
            my_sim.demo_id = get_next_demo_id(demo_root)
            my_sim.reset()
            print(f"[INFO] start simulation demo_{my_sim.demo_id}")
            # Note: if you set viz_object_goals(), remember to disable gravity and collision for object
            # my_sim.viz_object_goals(sample_step=10, hold_steps=40)
            std = 0
            if i > 0:
                std = std + 0.001
            success_ids = my_sim.inference(std=std)
            print(f"[INFO] success_ids: {success_ids}")
            if len(success_ids) > 0:
                success = True
                task_cfg = my_sim.from_data_to_task_cfg(key)
                my_sim.task_cfg = task_cfg
                assert task_cfg.reference_trajectory is not None
                my_sim.traj_cfg_list = [task_cfg.reference_trajectory]
                my_sim.save_data(ignore_keys=["segmask", "depth"], env_ids=success_ids[:1], export_hdf5=True)
                break
            # actions = np.load("outputs/lab16/demos/demo_5/env_000/actions.npy")
            # my_sim.replay_actions(actions)
        if success == False:
            print("[ERR] no successful environments!")
        env.close()
    #simulation_app.close()
    return True 

def main():
    base_dir = Path.cwd()
    cfg = yaml.safe_load((base_dir / "config" / "config.yaml").open("r"))
    keys = cfg["keys"]
    sim_heuristic_manip(keys, args_cli)

if __name__ == "__main__":
    main()
    simulation_app.close()
