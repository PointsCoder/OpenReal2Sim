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
from envs.task_construct import construct_task_config, add_reference_trajectory, load_task_cfg, get_task_cfg
from envs.running_cfg import get_heuristic_config, get_simulation_config


# ─────────── CLI ───────────
parser = argparse.ArgumentParser("sim_policy")
parser.add_argument("--key", type=str, default="demo_video", help="scene key (outputs/<key>)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.headless = True # headless mode for batch execution
app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

# ─────────── Runtime imports ───────────
import isaaclab.sim as sim_utils
from isaaclab.utils.math import subtract_frame_transforms
import sys

from sim_utils_demo.grasp_group_utils import *


# ─────────── Simulation environments ───────────
from sim_base_demo import BaseSimulator, get_next_demo_id
from sim_env_factory_demo import make_env

from sim_utils_demo.transform_utils import pose_to_mat, mat_to_pose, grasp_approach_axis_batch
from sim_utils_demo.sim_utils import load_sim_parameters




# ──────────────────────────── Heuristic Manipulation ────────────────────────────
class HeuristicManipulation(BaseSimulator):
    """
    Physical trial-and-error grasping with approach-axis perturbation:
      • Multiple grasp proposals executed in parallel;
      • Every attempt does reset → pre → grasp → close → lift → check;
      • Early stops when any env succeeds; then re-exec for logging.
    """
    def __init__(self, sim, scene, sim_cfgs: dict, args, out_dir: Path, img_folder: str, data_dir: Path, num_trials: int, grasp_num: int, robot: str, goal_offset: float, save_interval: int, decimation: int = 1, grasps_use: int = 50, grasp_delta: float = -0.003):
        robot_pose = torch.tensor(
            sim_cfgs["robot_cfg"]["robot_pose"],
            dtype=torch.float32,
            device=sim.device
        )
        selected_object_id = sim_cfgs["demo_cfg"]["manip_object_id"]
        super().__init__(
            sim=sim, sim_cfgs=sim_cfgs, scene=scene, args=args_cli,
            robot_pose=robot_pose, cam_dict=sim_cfgs["cam_cfg"],
            out_dir=out_dir, img_folder=img_folder, data_dir = data_dir,
            enable_motion_planning=True,
            set_physics_props=True, debug_level=0,
            save_interval=save_interval,
            decimation=decimation,
            selected_object_id=selected_object_id,
         
        )
        self.final_gripper_closed = sim_cfgs["demo_cfg"]["final_gripper_closed"]
        self.robot_cfg_list = sim_cfgs["robot_cfg"]["robot_cfg_list"]
        self.traj_path = sim_cfgs["demo_cfg"]["traj_path"]
        self.goal_offset = [0, 0, goal_offset]
        self.grasp_path = sim_cfgs["demo_cfg"]["grasp_path"]
        self.grasp_idx = sim_cfgs["demo_cfg"]["grasp_idx"]
        self.grasp_pre = sim_cfgs["demo_cfg"]["grasp_pre"]
        self.grasp_delta = grasp_delta
        self.task_type = sim_cfgs["demo_cfg"]["task_type"]
        self.robot_type = robot
        self.load_obj_goal_traj()
        self.std = 0
        self.grasp_round = 0
        self.grasp_num = grasp_num
        self.num_trials = num_trials
        self.trial_num = 0
        self.grasp_poses = []
        self.pregrasp_poses = []
        self.end_object_poses = []
        self.robot_poses = []   
        self.xy_derivative_threshold = 0.02
        self.angular_derivative_threshold = 10.0
        self.test_mask = False
        self.grasps_use = grasps_use
    

    def reset(self, env_ids=None, rechoose_robot_position=False):
        super().reset(env_ids)
        if rechoose_robot_position:
            device = self.object_prim.device
            if env_ids is None:
                env_ids_t = self._all_env_ids.to(device)  # (B,)
            else:
                env_ids_t = torch.as_tensor(env_ids, device=device, dtype=torch.long).view(-1)  # (M,)
            M = int(env_ids_t.shape[0])
            # --- object pose/vel: set object at env origins with identity quat ---
            print(f"original robot pose: {self.robot.data.root_state_w[:, :7]}")
            new_rp = random.choice(self.robot_cfg_list)
            new_pos = np.array(new_rp["position"])
            new_rot = np.array(new_rp["rotation"])
            env_origins_robot = self.scene.env_origins.to(device)[env_ids_t]
            robot_pose_world = np.zeros((M, 7), dtype=np.float32)
            robot_pose_world[:, :3] = env_origins_robot.cpu().numpy() + new_pos[:3]
            robot_pose_world[:, 3:7] = new_rot[:4]
            self.robot.write_root_pose_to_sim(torch.tensor(robot_pose_world, dtype=torch.float32, device=device), env_ids=env_ids_t)
            self.robot.write_root_velocity_to_sim(
                torch.zeros((M, 6), device=device, dtype=torch.float32), env_ids=env_ids_t
            )
            joint_pos = self.robot.data.default_joint_pos.to(self.robot.device)[env_ids_t]  # (M,7)
            joint_vel = self.robot.data.default_joint_vel.to(self.robot.device)[env_ids_t]  # (M,7)
            self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_t)
            self.robot.write_data_to_sim()
            print(f"new robot pose: {self.robot.data.root_state_w[:, :7]}")
            self.step()
            self.clear_data()
            self.robot.update(self.sim_dt)

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
        #obj_state_np[:, :3] += offset_np  # raise a bit to avoid collision

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
                if t < T-1:
                    goal[:3, 3] += offset_np
                obj_goal[b, t] = goal

        self.obj_goal_traj_w = obj_goal  # [B, T, 4, 4]

    def follow_object_goals(self, start_joint_pos, sample_step=1, recalibrate_interval = 1, visualize=True):
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
        joint_pos = start_joint_pos
        root_w = self.robot.data.root_state_w[:, 0:7]  # robot base poses per env
        t_iter = list(range(0, T, sample_step))
        t_iter = t_iter + [T-1] if t_iter[-1] != T-1 else t_iter
        ee_pos_initial = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:3]
        obj_pos_initial = self.object_prim.data.root_com_pos_w[:, 0:3]
        initial_grasp_dist = torch.norm(ee_pos_initial - obj_pos_initial, dim=1) # [B]
        self.initial_grasp_dist = initial_grasp_dist

        for t in t_iter:
            if recalibrate_interval> 0 and t % recalibrate_interval == 0:
                ee_w  = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]  # [B,7]
                obj_w = self.object_prim.data.root_state_w[:, :7]                                 # [B,7]
                T_ee_in_obj = []
                for b in range(B):
                    T_ee_w  = pose_to_mat(ee_w[b, :3],  ee_w[b, 3:7])
                    T_obj_w = pose_to_mat(obj_w[b, :3], obj_w[b, 3:7])
                    T_ee_in_obj.append((np.linalg.inv(T_obj_w) @ T_ee_w).astype(np.float32))
                self.T_ee_in_obj = T_ee_in_obj
            goal_pos_list, goal_quat_list = [], []
            print(f"[INFO] follow object goal step {t}/{T}")
            for b in range(B):
                T_obj_goal = obj_goal_all[b, t]            # (4,4)
                T_ee_goal  = T_obj_goal @ T_ee_in_obj[b]   # (4,4)
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
            if self.count % self.save_interval == 0:
                self.save_dict["actions"].append(np.concatenate([ee_pos_b.cpu().numpy(), ee_quat_b.cpu().numpy(), np.ones((B, 1))], axis=1))
                action_index = np.ones((B, 1)) * self.get_current_frame_count()
                self.save_dict["action_indices"].append(action_index)

        is_success, rechoose_base_flag = self.is_success() #& self.goal_is_success()
        success_ids = torch.where(is_success)[0]
        print(f'[INFO] waiting for gripper final state: open is {not self.final_gripper_closed}')
        joint_pos = self.wait(gripper_open=not self.final_gripper_closed, steps=50)
    
        return joint_pos, success_ids, rechoose_base_flag

    ## FIXME: This should have some problem. But it works.
    def follow_object_centers(self, start_joint_pos, sample_step=1, visualize=True, recalibrate_interval=1):
            B = self.scene.num_envs
            obj_goal_all = self.obj_goal_traj_w  # [B, T, 4, 4]
            T = obj_goal_all.shape[1]
            joint_pos = start_joint_pos
            root_w = self.robot.data.root_state_w[:, 0:7]  # robot base poses per env
            ee_pos_initial = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:3]
            obj_pos_initial = self.object_prim.data.root_com_pos_w[:, 0:3]
            initial_grasp_dist = torch.norm(ee_pos_initial - obj_pos_initial, dim=1) # [B]
            self.initial_grasp_dist = initial_grasp_dist
            t_iter = list(range(0, T, sample_step))
            t_iter = t_iter + [T-1] if t_iter[-1] != T-1 else t_iter

            for t in t_iter:
                if recalibrate_interval> 0 and t % recalibrate_interval == 0:
                    ee_w  = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]  # [B,7]
                    obj_w = self.object_prim.data.root_state_w[:, :7]                                 # [B,7]
                    T_ee_ws = []
                    T_obj_ws = []
                    for b in range(B):
                        T_ee_w  = pose_to_mat(ee_w[b, :3],  ee_w[b, 3:7])
                        T_obj_w = pose_to_mat(obj_w[b, :3], obj_w[b, 3:7])
                        T_ee_ws.append(T_ee_w)
                        T_obj_ws.append(T_obj_w)
                    print(f"[INFO] recalibrated at step {t}/{T}")
                goal_pos_list, goal_quat_list = [], []
                print(f"[INFO] follow object goal step {t}/{T}")
                for b in range(B):
                    T_obj_goal = obj_goal_all[b, t]            # (4,4)
                    trans_offset = T_obj_goal - T_obj_ws[b]
                    T_ee_goal  = T_ee_ws[b] + trans_offset
                    pos_b, quat_b = mat_to_pose(T_ee_goal)

                    goal_pos_list.append(pos_b.astype(np.float32))
                    goal_quat_list.append(quat_b.astype(np.float32))


                goal_pos  = torch.as_tensor(np.stack(goal_pos_list),  dtype=torch.float32, device=self.sim.device)
                goal_quat = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 3:7]

                if visualize:
                    self.show_goal(goal_pos, goal_quat)
                ee_pos_b, ee_quat_b = subtract_frame_transforms(
                    root_w[:, :3], root_w[:, 3:7], goal_pos, goal_quat
                )
                joint_pos, success = self.move_to(ee_pos_b, ee_quat_b, gripper_open=False)
                print('[INFO] goal pose', obj_goal_all[:, t], 'current obj pose', self.object_prim.data.root_state_w[:, :3])
                print('[INFO]current ee obj trans diff', self.object_prim.data.root_state_w[:, :3] - self.robot.data.root_state_w[:, :3])
                self.save_dict["actions"].append(np.concatenate([ee_pos_b.cpu().numpy(), ee_quat_b.cpu().numpy(), np.ones((B, 1))], axis=1))
                action_index = np.ones((B, 1)) * self.get_current_frame_count()
                self.save_dict["action_indices"].append(action_index)
            
            is_success = self.holding_is_success(position_threshold = 0.10)
            print('[INFO] last obj goal', obj_goal_all[:, -1])
            print('[INFO] last obj pos', self.object_prim.data.root_state_w[:, :3])
            joint_pos = self.wait(gripper_open=not self.final_gripper_closed, steps=30)
            return joint_pos, torch.where(is_success)[0]




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
        print(f"[INFO] root: {root}")
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

        # Record initial state: position (xyz) + orientation (quat)
        obj0 = self.object_prim.data.root_com_pos_w[:, 0:3]     # [B,3]
        obj_quat0 = self.object_prim.data.root_state_w[:, 3:7]  # [B,4] wxyz
        ee_w = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]  # [B,7]
        ee_p0 = ee_w[:, :3]
        ee_q0 = ee_w[:, 3:7]
        
        print(f"[INFO] Pre-lift - Object XYZ: mean={obj0.mean(dim=0)}, EE XYZ: mean={ee_p0.mean(dim=0)}")

        # lift: keep orientation, add height
        target_p = ee_p0.clone()
        target_p[:, 2] += lift_height

        root = self.robot.data.root_state_w[:, 0:7]
        p_lift_b, q_lift_b = subtract_frame_transforms(
            root[:, 0:3], root[:, 3:7],
            target_p, ee_q0
        )
        jp, success = self.move_to(p_lift_b, q_lift_b, gripper_open=False)
        if torch.any(success==False): return np.zeros(B, bool), np.zeros(B, np.float32)
        jp = self.wait(gripper_open=False, steps=8)

        self.robot.update(dt=self.sim.get_physics_dt())
        self.object_prim.update(dt=self.sim.get_physics_dt())
        # Record final state
        obj1 = self.object_prim.data.root_com_pos_w[:, 0:3]
        obj_quat1 = self.object_prim.data.root_state_w[:, 3:7]  # [B,4] wxyz
        ee_w1 = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        ee_p1 = ee_w1[:, :3]
        ee_q1 = ee_w1[:, 3:7]
        
        print(f"[INFO] Post-lift - Object XYZ: mean={obj1.mean(dim=0)}, EE XYZ: mean={ee_p1.mean(dim=0)}")

        # --- Check 1: Z-axis tight coupling (vertical lift) ---
        ee_z_diff = ee_p1[:, 2] - ee_p0[:, 2]
        obj_z_diff = obj1[:, 2] - obj0[:, 2]
        z_coupling = torch.abs(ee_z_diff - obj_z_diff) <= 0.06  # [B]
        z_lifted = (torch.abs(ee_z_diff) >= 0.3 * lift_height) & \
                   (torch.abs(obj_z_diff) >= 0.3 * lift_height)  # [B]
        
        # --- Check 2: XY position stability (no lateral slip) ---
        # Object XY should move with gripper (within 2cm tolerance)
        ee_xy_diff = ee_p1[:, :2] - ee_p0[:, :2]  # [B, 2]
        obj_xy_diff = obj1[:, :2] - obj0[:, :2]    # [B, 2]
        xy_deviation = torch.norm(ee_xy_diff - obj_xy_diff, dim=1)  # [B]
        xy_stable = xy_deviation <= self.xy_derivative_threshold  # [B]
        
        # --- Check 3: Orientation stability (roll, pitch, yaw) ---
        # Convert quaternions to euler angles for easier checking
        def quat_to_euler(quat):
            """Convert wxyz quaternion to roll, pitch, yaw (radians)"""
            w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
            
            # Roll (x-axis rotation)
            sinr_cosp = 2 * (w * x + y * z)
            cosr_cosp = 1 - 2 * (x * x + y * y)
            roll = torch.atan2(sinr_cosp, cosr_cosp)
            
            # Pitch (y-axis rotation)
            sinp = 2 * (w * y - z * x)
            sinp = torch.clamp(sinp, -1.0, 1.0)
            pitch = torch.asin(sinp)
            
            # Yaw (z-axis rotation)
            siny_cosp = 2 * (w * z + x * y)
            cosy_cosp = 1 - 2 * (y * y + z * z)
            yaw = torch.atan2(siny_cosp, cosy_cosp)
            
            return roll, pitch, yaw
        
        roll0, pitch0, yaw0 = quat_to_euler(obj_quat0)
        roll1, pitch1, yaw1 = quat_to_euler(obj_quat1)
        
        # Check angular changes (should be minimal for good grasp)
        roll_diff = torch.abs(roll1 - roll0)
        pitch_diff = torch.abs(pitch1 - pitch0)
        yaw_diff = torch.abs(yaw1 - yaw0)
        
        # Allow up to 10 degrees rotation in any axis
        orientation_stable = (roll_diff <= torch.deg2rad(torch.tensor(self.angular_derivative_threshold, device=roll_diff.device))) & \
                            (pitch_diff <= torch.deg2rad(torch.tensor(self.angular_derivative_threshold, device=pitch_diff.device))) & \
                            (yaw_diff <= torch.deg2rad(torch.tensor(self.angular_derivative_threshold, device=yaw_diff.device)))  # [B]
        
        # --- Combined success criteria ---
        lifted = z_coupling & z_lifted & xy_stable & orientation_stable  # [B]
        
        # Detailed logging
        for b in range(min(B, 3)):  # Log first 3 envs for debugging
            print(f"  Env[{b}]: Z-coupling={z_coupling[b].item()}, "
                  f"XY-stable={xy_stable[b].item()} (dev={xy_deviation[b].item():.4f}m), "
                  f"Orient-stable={orientation_stable[b].item()} "
                  f"(roll={torch.rad2deg(roll_diff[b]).item():.1f}°, "
                  f"pitch={torch.rad2deg(pitch_diff[b]).item():.1f}°, "
                  f"yaw={torch.rad2deg(yaw_diff[b]).item():.1f}°) "
                  f"→ PASS={lifted[b].item()}")

        score = torch.zeros_like(ee_z_diff)
        score[lifted] = 1000.0
        self.save_data(formal=False)
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
        jp = self.wait(gripper_open=gripper_open, steps=50)
        return jp
 
    def goal_is_success(self, position_threshold: float = 0.10) -> torch.Tensor:
        """
        Verify if the manipulation task succeeded by checking:
        1. Object is at Goal (Distance < 10cm)
        """
        current_obj_pose = self.object_prim.data.root_state_w[:, :3]
        final_goal_pose = self.obj_goal_traj_w[:, -1, :, :]  # [B, 4, 4]
        print(f'[INFO] current_obj_pose: {current_obj_pose.mean(dim=0)}')
        print(f'[INFO] final_goal_pose: {final_goal_pose.mean(dim=0)}')
        obj_success = torch.norm(current_obj_pose - final_goal_pose, dim=1) <= position_threshold
        return obj_success
    
    def holding_is_success(self, position_threshold: float = 0.20) -> torch.Tensor:
        ee_w  = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        obj_w = self.object_prim.data.root_com_pos_w[:, 0:3]
        dist = torch.norm(obj_w[:, :3] - ee_w[:, :3], dim=1) # [B]
        print(f"[INFO] holding dist: {dist.mean().item():.3f} m")
      
        return (dist < position_threshold).to(torch.bool)
    
    def is_success(self, position_threshold: float = 0.10, gripper_threshold: float = 0.10, holding_threshold: float = 0.02) -> torch.Tensor:
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
        print(f"[INFO] root_dist: {root_dist.mean().item():.3f} m")
        print(f"[INFO] com_dist: {com_dist.mean().item():.3f} m")
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
        print(f"[INFO] grip_goal_dist: {grip_goal_dist.mean().item():.3f} m")
      
        # --- 3. Holding Check (Object in Gripper) ---
        # Check if the distance between Gripper and Object COM has remained stable.
        # We compare the final distance to the initial grasp distance.
        obj_com_final = self.object_prim.data.root_com_pos_w[:, 0:3]
        current_grip_com_dist = torch.norm(ee_pos_final - obj_com_final, dim=1)
        
        if hasattr(self, 'initial_grasp_dist') and self.initial_grasp_dist is not None:
            grasp_deviation = torch.abs(current_grip_com_dist - self.initial_grasp_dist)
            holding_success = grasp_deviation <= 0.02
            
            holding_metric = grasp_deviation
            holding_threshold = 0.02
            holding_metric_name = "Grip-COM Deviation"
        else:
            holding_success = current_grip_com_dist <= 0.18
            holding_metric = current_grip_com_dist
            holding_threshold = 0.15
            holding_metric_name = "Grip-COM Dist"
        print(f"[INFO] holding_metric: {holding_metric.mean().item():.3f} m")
    
        # Combined Success
        success_mask = obj_success & gripper_success & holding_success
        rechoose_base_flag = all(holding_success & (~obj_success) & (~gripper_success))
        print("="*50)
        print(f"TASK VERIFIER: SUCCESS - {success_mask.sum().item()}/{B} environments succeeded")
        print("="*50 + "\n")
        return success_mask, rechoose_base_flag



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

  
    def grasp_trials(self, gg):
        B = self.scene.num_envs
        print(f"[INFO] len(gg): {len(gg)}")
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

        # assign different grasp proposals to different envs\
      
        winners = []
        success = False

        current_object_state = self.object_prim.data.root_state_w[:, :7]
     
           
        while self.grasp_round < len(idx_all) and self.trial_num < self.num_trials:
            if self.trial_num == self.num_trials -1 :
                self.xy_derivative_threshold = 10000
                self.angular_derivative_threshold = 10000
            if self.grasp_round == 0 and self.trial_num > 0:
                self.reset(rechoose_robot_position=True)
                self.test_mask = False
            start = self.grasp_round
            block = idx_all[start : start + B]
            if len(block) < B:
                block = block + [block[-1]] * (B - len(block))
            grasp_pos_w_batch, grasp_quat_w_batch = [], []
            for idx in block:
                p_w, q_w = gg.retrieve_grasp_group(int(idx))
                grasp_pos_w_batch.append(p_w.astype(np.float32))
                grasp_quat_w_batch.append(q_w.astype(np.float32))
            grasp_pos_w_batch  = np.stack(grasp_pos_w_batch,  axis=0)  # (B,3)
            grasp_quat_w_batch = np.stack(grasp_quat_w_batch, axis=0)  # (B,4)
            #grasp_pos_w_batch, grasp_quat_w_batch = self.refine_grasp_pose(grasp_pos_w_batch, grasp_quat_w_batch)
            self.show_goal(grasp_pos_w_batch, grasp_quat_w_batch)
            
            # random disturbance along approach axis
            pre_dist_batch = np.full((B,), pre_dist_const, dtype=np.float32)
            delta_batch    = rng.normal(self.grasp_delta, self.std, size=(B,)).astype(np.float32)

            info = self.build_grasp_info(grasp_pos_w_batch, grasp_quat_w_batch,
                                        pre_dist_batch, delta_batch)

            ok_batch, score_batch = self.execute_and_lift_once_batch(info)
            if not self.test_mask:
                if self.judge_robot_position_with_mask():
                    print(f"[INFO] robot position is not good, resetting and trying again")
                    self.reset(rechoose_robot_position=True)
                    self.grasp_round = 0
                    self.num_trials += 1
                    continue
                else:
                    print(f"[INFO] robot position is good, continuing")
                    self.test_mask = True
            if start + B > len(idx_all):
                ok_batch = ok_batch[:(len(idx_all) - start)]
                score_batch = score_batch[:(len(idx_all) - start)]
            ok_cnt = int(ok_batch.sum())
            print(f"[SEARCH] block[{start}:{start+B}] -> success {ok_cnt}/{B}")
            self.grasp_round = start + B
            if self.grasp_round >= len(idx_all):
                self.grasp_round = 0
                self.std = self.std + 0.003
                self.trial_num +=1
                print(f"[INFO] trial_num: {self.trial_num}")
                self.xy_derivative_threshold = self.xy_derivative_threshold + 0.03
                self.angular_derivative_threshold = self.angular_derivative_threshold + 10.0
            
            
            if ok_cnt > 0:
                for candidate in range(len(score_batch)):
                    if score_batch[candidate] == np.max(score_batch):
                        winner = candidate
                        chosen_pose_w = (grasp_pos_w_batch[winner], grasp_quat_w_batch[winner])
                        chosen_pre    = float(pre_dist_batch[winner])
                        chosen_delta  = float(delta_batch[winner])
                        success = True
                        winners.append({
                            "success": success,
                            "chosen_pose_w": chosen_pose_w,
                            "chosen_pre": chosen_pre,
                            "chosen_delta": chosen_delta,
                        })
                return winners
          

        if not success:
            print("[ERR] no proposal succeeded to lift after full search.")
            return [{
                "success": success,
                "chosen_pose_w": None,
                "chosen_pre": None,
                "chosen_delta": None,
            }]


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

    def inference(self) -> list[int]:
        """
        Main function of the heuristic manipulation policy.
        Physical trial-and-error grasping with approach-axis perturbation:
          • Multiple grasp proposals executed in parallel;
          • Every attempt does reset → pre → grasp → close → lift → check;
          • Early stops when any env succeeds; then re-exec for logging.
        """
        B = self.scene.num_envs
        self.reset()
        current_object_pose = self.object_prim.data.root_state_w[:, :7].cpu().numpy()
        cur_4x4_mat = np.eye(4, dtype=np.float32)
        cur_4x4_mat[:3, :3] = transforms3d.quaternions.quat2mat(current_object_pose[0, 3:7])
        cur_4x4_mat[:3, 3] = current_object_pose[0, :3] - self.scene.env_origins.cpu().numpy()[0]
        print(f"[INFO] cur_4x4_mat: {cur_4x4_mat}")
        self.wait(gripper_open=True, steps=10)

        # read grasp proposals
        npy_path = self.grasp_path
        if npy_path is None or (not os.path.exists(npy_path)):
            print(f"[ERR] grasps npy not found: {npy_path}")
            return []
        gg = GraspGroup().from_npy(npy_file_path=npy_path)
        gg = gg.to_world_transform()
        gg = gg.transform(cur_4x4_mat.astype(np.float32))
       
        if len(gg) == 0:
            print(f"[ERR] no grasp proposals found: {npy_path}")
            return []
        success_num = 0
        rescore_flag = False
        assert self.grasp_idx < 0 or self.grasp_num == 1, "[ERR] grasp_idx and grasp_num cannot be set together"
        while success_num < self.grasp_num and self.trial_num < self.num_trials:
            if success_num > 0 and self.grasp_idx >= 0:
                return success_num
            if self.grasp_idx >= 0:
                if self.grasp_idx >= len(gg):
                    print(f"[ERR] grasp_idx {self.grasp_idx} out of range [0,{len(gg)})")
                    return []
                print(f"[INFO] using fixed grasp index {self.grasp_idx} for all envs.")
                p_w, q_w = gg.retrieve_grasp_group(int(self.grasp_idx))
                ret = {
                    "success": True,
                    "chosen_pose_w": (p_w.astype(np.float32), q_w.astype(np.float32)),
                    "chosen_pre": self.grasp_pre if self.grasp_pre is not None else 0.12,
                    "chosen_delta": self.grasp_delta if self.grasp_delta is not None else 0.0,
                }
                print(f"[INFO] grasp delta (m): {ret['chosen_delta']:.4f}")
                rets = [ret]
            
            else:
                if not rescore_flag:
                    gg = gg.rescore(direction_hint=[0, 0, -1], reorder_num=10)
                    rescore_flag = True
                rets = self.grasp_trials(gg[:self.grasps_use])

            print("[INFO] Re-exec all envs with the winning grasp, then follow object goals.")
            if (rets is None or rets[0]["success"] == False):
                print("[ERR] no proposal succeeded to lift after full search.")
                return 0


        

            B = self.scene.num_envs
            for i in range(0, len(rets), B):
                block = list(range(i, min(i + B, len(rets))))
                if len(block) < B:
                    block = block + [block[-1]] * (B - len(block))
                p_win = [rets[j]["chosen_pose_w"][0] for j in block]
                q_win = [rets[j]["chosen_pose_w"][1] for j in block]
                p_all = np.array(p_win)
                q_all = np.array(q_win)
                pre_all = np.array([rets[j]["chosen_pre"] for j in block], dtype=np.float32)
                del_all = np.array([rets[j]["chosen_delta"] for j in block], dtype=np.float32)

                info_all = self.build_grasp_info(p_all, q_all, pre_all, del_all)

                # reset and conduct main process: open→pre→grasp→close→follow_object_goals
                self.reset()

                init_manip_object_com = self.object_prim.data.root_com_pos_w[:, :3].cpu().numpy()
                init_manip_object_com -= self.scene.env_origins.cpu().numpy()
                self.init_manip_object_com = init_manip_object_com[0]
                #print(f"[INFO]init_manip_object_com: {self.init_manip_object_com}")
                robot_pose = self.robot.data.root_state_w[:, :7].cpu().numpy()
                robot_pose[:, :3] = robot_pose[:, :3] - self.scene.env_origins.cpu().numpy()
                #print("[INFO] robot pose", robot_pose)
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
                self.wait(gripper_open=True, steps=50)

                # pre → grasp
                jp, success = self.move_to(info_all["pre_p_b"], info_all["pre_q_b"], gripper_open=True)
                if torch.any(success==False): return []
                self.save_dict["actions"].append(np.concatenate([info_all["pre_p_b"].cpu().numpy(), info_all["pre_q_b"].cpu().numpy(), np.zeros((B, 1))], axis=1))
                action_index = np.ones((B, 1)) * self.get_current_frame_count()
                self.save_dict["action_indices"].append(action_index)
                jp = self.wait(gripper_open=True, steps=50)

                jp, success = self.move_to(info_all["p_b"], info_all["q_b"], gripper_open=True)
                if torch.any(success==False): return []
                self.save_dict["actions"].append(np.concatenate([info_all["p_b"].cpu().numpy(), info_all["q_b"].cpu().numpy(), np.zeros((B, 1))], axis=1))
                action_index = np.ones((B, 1)) * self.get_current_frame_count()
                self.save_dict["action_indices"].append(action_index)
                # close gripper
                jp = self.wait(gripper_open=False, steps=50)
                self.save_dict["actions"].append(np.concatenate([info_all["p_b"].cpu().numpy(), info_all["q_b"].cpu().numpy(), np.ones((B, 1))], axis=1))
                action_index = np.ones((B, 1)) * self.get_current_frame_count()
                self.save_dict["action_indices"].append(action_index)
                # object goal following
                print(f"[INFO] lifting up by {self.goal_offset[2]} meters")
                self.lift_up(height=self.goal_offset[2], gripper_open=False, steps=8)
            
                # if self.task_type == "simple_pick_place" or self.task_type == "simple_pick":
                #     jp, success_ids = self.follow_object_centers(jp, sample_step=1, visualize=True)
                # elif self.task_type == "targetted_pick_place":
                #     jp, success_ids = self.follow_object_goals(jp, sample_step=1, visualize=True)
                # else:
                #     raise ValueError(f"[ERR] Invalid task type: {self.task_type}")
                jp, success_ids, rechoose_base_flag = self.follow_object_goals(jp, sample_step=1, visualize=True)
            
                object_prim_world_pose = self.object_prim.data.root_state_w[:, :7].cpu().numpy()
                object_prim_world_pose[:, :3] = object_prim_world_pose[:, :3] - self.scene.env_origins.cpu().numpy()
            
                
                robot_world_pose = self.robot.data.root_state_w[:, :7].cpu().numpy()
                robot_world_pose[:, :3] = robot_world_pose[:, :3] - self.scene.env_origins.cpu().numpy()
            

                
            
                # Properly handle the case when success_ids is a numpy array
                # Convert it to a torch tensor if needed, before calling torch.whe\+
                print(f"[INFO] success_ids: {success_ids}")
                print(f"[INFO] rechoose_base_flag: {rechoose_base_flag}")
                self.save_data(formal=False)
                if self.judge_robot_position_with_mask() or rechoose_base_flag:
                    print(f"[INFO] robot position is not good, continuing")
                    self.reset(rechoose_robot_position=True)
                    self.num_trials += 1
                    self.grasp_round = 0
                    self.test_mask = False
                    continue
            
                # If success_ids is already a tensor, we keep as-is
                left_rets = len(rets) - i
                cleaned_success_ids = []
                for m in success_ids:
                    if m < left_rets:
                        cleaned_success_ids.append(m)
                success_num += len(cleaned_success_ids)
                if len(cleaned_success_ids) > 0:
                    self.save_data(ignore_keys=["segmask", "depth"], env_ids=cleaned_success_ids, export_hdf5=True, formal = True)
                    for k in cleaned_success_ids:
                        pregrasp_pose = self.save_dict["pregrasp_pose_w"][k]
                        pregrasp_pose[:3] = pregrasp_pose[:3] - self.scene.env_origins.cpu().numpy()[k]
                        pregrasp_mat = np.eye(4, dtype=np.float32)
                        pregrasp_mat[:3, :3] = transforms3d.quaternions.quat2mat(pregrasp_pose[3:7])
                        pregrasp_mat[:3, 3] = pregrasp_pose[:3]
                        ori_pregrasp_mat = np.linalg.inv(cur_4x4_mat) @ pregrasp_mat
                        pregrasp_pose = np.concatenate([ori_pregrasp_mat[:3, 3], transforms3d.quaternions.mat2quat(ori_pregrasp_mat[:3, :3])])
                        grasp_pose = self.save_dict["grasp_pose_w"][k]
                        grasp_pose[:3] = grasp_pose[:3] - self.scene.env_origins.cpu().numpy()[k]
                        grasp_mat = np.eye(4, dtype=np.float32)
                        grasp_mat[:3, :3] = transforms3d.quaternions.quat2mat(grasp_pose[3:7])
                        grasp_mat[:3, 3] = grasp_pose[:3]
                        ori_grasp_mat = np.linalg.inv(cur_4x4_mat) @ grasp_mat
                        grasp_pose = np.concatenate([ori_grasp_mat[:3, 3], transforms3d.quaternions.mat2quat(ori_grasp_mat[:3, :3])])
                        
                        end_object_pose = object_prim_world_pose[k]
                        robot_pose = robot_world_pose[k]
                        robot_pose[:3] = robot_pose[:3] - self.scene.env_origins.cpu().numpy()[k]
                
                        self.pregrasp_poses.append(pregrasp_pose)
                        self.grasp_poses.append(grasp_pose)
                        self.end_object_poses.append(end_object_pose)
                        self.robot_poses.append(robot_pose)
                print(f"[INFO] success_num: {success_num}")
        
        
        return success_num
           
    def judge_robot_position_with_mask(self, ratio_threshold: float = 0.65) -> bool:
        """
        Judge if robot position is good by checking if object (excluding robot) occupies too much of the image.
        Uses current camera instance segmentation to extract object-only mask.
        """
        robot_mask = self.save_dict["robot_mask"]
        object_mask = self.save_dict["object_mask"]
        seg_mask = self.save_dict["segmask"]
        seg_mask_array = np.array(seg_mask)
        if seg_mask_array.ndim == 5:
            seg_mask_array = seg_mask_array.squeeze(-1)
        elif seg_mask_array.ndim == 3:
            seg_mask_array = seg_mask_array[np.newaxis, ...]
        if seg_mask_array.ndim != 4:
            print(f"[ERROR] Unexpected seg_mask_array shape: {seg_mask_array.shape}, expected 4 dimensions after processing")
            return False
        T, B, H, W = seg_mask_array.shape
        robot_mask_array = np.array(robot_mask)
        if robot_mask_array.ndim == 5:
            robot_mask_array = robot_mask_array.squeeze(-1)
        elif robot_mask_array.ndim == 3:
            robot_mask_array = robot_mask_array[np.newaxis, ...]
        if robot_mask_array.ndim != 4:
            print(f"[ERROR] Unexpected robot_mask_array shape: {robot_mask_array.shape}, expected 4 dimensions after processing")
            return False
        T, B, H, W = robot_mask_array.shape
        object_mask_array = np.array(object_mask)
        if object_mask_array.ndim == 5:
            object_mask_array = object_mask_array.squeeze(-1)
        elif object_mask_array.ndim == 3:
            object_mask_array = object_mask_array[np.newaxis, ...]
        if object_mask_array.ndim != 4:
            print(f"[ERROR] Unexpected object_mask_array shape: {object_mask_array.shape}, expected 4 dimensions after processing")
            return False
        T, B, H, W = object_mask_array.shape
        
        seg_mask_array = np.array(seg_mask)
        if seg_mask_array.ndim == 5:
            seg_mask_array = seg_mask_array.squeeze(-1)
        elif seg_mask_array.ndim == 3:
            seg_mask_array = seg_mask_array[np.newaxis, ...]
        if seg_mask_array.ndim != 4:
            print(f"[ERROR] Unexpected seg_mask_array shape: {seg_mask_array.shape}, expected 4 dimensions after processing")
            return False
        T, B, H, W = seg_mask_array.shape

        for t in range(T):
            for b in range(B):
                mask = robot_mask_array[t, b] 
                mask_ratio = np.sum(mask) / (H * W)
                object_mask = object_mask_array[t, b]
                object_mask_ratio = np.sum(object_mask) / (H * W)
                seg_mask = seg_mask_array[t, b]
                seg_mask_ratio = np.sum(seg_mask) / (H * W)
                if object_mask_ratio < 0.005 and mask_ratio > ratio_threshold and seg_mask_ratio > 0:
                    return True
  
      
        return False


        
    def from_data_to_task_cfg(self, key:str, already_exist: bool = False) -> TaskCfg:
        try:
            BASE_DIR = Path.cwd()
            scene_json_path = BASE_DIR / "outputs" / key / "simulation" / "scene.json"
            task_base_folder = BASE_DIR / "tasks"
            base_folder = task_base_folder / key
            scene_dict = json.load(open(scene_json_path))
          
            if not base_folder.exists() and not already_exist :
                base_folder.mkdir(parents=True, exist_ok=True)
                task_cfg, _ = construct_task_config(key, scene_dict, task_base_folder)
                return task_cfg
            elif not already_exist:
                task_cfg = get_task_cfg(key, base_folder)
                return task_cfg
            else:
                task_cfg = get_task_cfg(key, base_folder)


            object_trajectory = np.load(self.traj_path).astype(np.float32)
            pose_quat_traj = []
            for pose_mat in object_trajectory:
                pose, quat = mat_to_pose(pose_mat)
                pose_quat = np.concatenate([np.array(pose), np.array(quat)])
                pose_quat_traj.append(pose_quat)
            pose_quat_traj = np.array(pose_quat_traj).reshape(-1, 7).tolist()
                                
            # Convert pregrasp and grasp poses from world to env-local frame
        
            trajectory_cfg_list = []
            final_gripper_close = self.final_gripper_closed
        
            for i in range(len(self.pregrasp_poses)):
                if task_cfg.task_type == TaskType.TARGETTED_PICK_PLACE:
                    success_metric = SuccessMetric(
                        success_metric_type = SuccessMetricType.TARGET_POINT,
                        end_pose  = self.end_object_poses[0].tolist(),
                        final_gripper_close = final_gripper_close,
                    )
                elif task_cfg.task_type == TaskType.SIMPLE_PICK_PLACE:
                    final_object_world_pose = self.end_object_poses[i]
                    ground_value = float(final_object_world_pose[2])
                    success_metric = SuccessMetric(
                        success_metric_type = SuccessMetricType.TARGET_PLANE,
                        ground_value = ground_value,
                        final_gripper_close = final_gripper_close,
                        end_pose  = self.end_object_poses[i].tolist()
                    )
                else:
                    success_metric = SuccessMetric(
                        success_metric_type = SuccessMetricType.SIMPLE_LIFT,
                        lift_height = 0.05,
                        final_gripper_close = final_gripper_close,
                        end_pose  = self.end_object_poses[i].tolist()
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
                    robot_pose = self.robot_poses[i],
                    object_poses =object_poses,
                    object_trajectory = pose_quat_traj,
                    final_gripper_close = final_gripper_close,
                    success_metric = success_metric,
                    pregrasp_pose = self.pregrasp_poses[i],
                    grasp_pose = self.grasp_poses[i],
                    robot_type = robot_type,
                )
                trajectory_cfg_list.append(trajectory_cfg)
            add_reference_trajectory(task_cfg, trajectory_cfg_list, base_folder)
        except Exception as e:
            print(f"[ERR] Error: {e}")
            import traceback
            traceback.print_exc()
            return None
        return task_cfg

    def from_self_to_sim_cfg(self, key: str) -> None:
        sim_info = {}
        sim_info["physics_freq"] = 1 / self.sim.get_physics_dt()
        sim_info["decimation"] = self.decimation
        sim_info["save_interval"] = self.save_interval
        BASE_DIR = Path.cwd()
        task_folder = BASE_DIR / "tasks" / key
        task_folder.mkdir(parents=True, exist_ok=True)
        task_file = task_folder / "sim_cfg.json"
        with open(task_file, "w") as f:
            json.dump(sim_info, f)

def sim_heuristic_manip(key: str, args_cli: argparse.Namespace, config_path: Optional[str] = None, config_dict: Optional[dict] = None):
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
    data_dir   = BASE_DIR / "h5py" / key

    args_cli.key = key
    local_img_folder = key
    # Load config from running_cfg, allow CLI args to override
    heuristic_cfg = get_heuristic_config(key)
    simulation_cfg = get_simulation_config(key)
    # Fix: argparse.Namespace does not have 'hasattr' method.
    # Use hasattr(args_cli, "num_envs") instead of args_cli.hasattr("num_envs")
    # Same for num_trials and grasp_num below
    grasps_use = args_cli.grasps_use if hasattr(args_cli, "grasps_use") and args_cli.grasps_use is not None else heuristic_cfg.grasps_use
    num_envs = args_cli.num_envs if hasattr(args_cli, "num_envs") and args_cli.num_envs is not None else heuristic_cfg.num_envs
    num_trials = args_cli.num_trials if hasattr(args_cli, "num_trials") and args_cli.num_trials is not None else heuristic_cfg.num_trials
    grasp_num = args_cli.grasp_num if hasattr(args_cli, "grasp_num") and args_cli.grasp_num is not None else heuristic_cfg.grasp_num
    robot = args_cli.robot if hasattr(args_cli, "robot") and args_cli.robot is not None else heuristic_cfg.robot
    grasp_delta = args_cli.grasp_delta if hasattr(args_cli, "grasp_delta") and args_cli.grasp_delta is not None else heuristic_cfg.grasp_delta
    goal_offset = heuristic_cfg.goal_offset
    save_interval =  simulation_cfg.save_interval
    physics_freq =  simulation_cfg.physics_freq
    decimation =  simulation_cfg.decimation
    print(f"[INFO] Using config for key '{key}': num_envs={num_envs}, num_trials={num_trials}, grasp_num={grasp_num}, robot={robot}")
    
    sim_cfgs = load_sim_parameters(BASE_DIR, key)
    env, _ = make_env(
        cfgs=sim_cfgs, num_envs=num_envs,
        device=args_cli.device,
        bg_simplify=False,
        physics_freq=physics_freq,
        decimation=decimation,
    )
    sim, scene = env.sim, env.scene

    my_sim = HeuristicManipulation(
        sim, scene, sim_cfgs=sim_cfgs,
        args=args_cli, out_dir=out_dir, img_folder=local_img_folder,
        data_dir = data_dir, num_trials = num_trials, grasp_num = grasp_num, robot = robot, goal_offset = goal_offset, save_interval = save_interval, decimation=decimation, grasps_use = grasps_use, grasp_delta = grasp_delta)

    robot_pose = torch.tensor(sim_cfgs["robot_cfg"]["robot_pose"], dtype=torch.float32, device=my_sim.sim.device)  # [7], pos(3)+quat(wxyz)(4)
    my_sim.set_robot_pose(robot_pose)
    my_sim.reset()
    task_cfg = my_sim.from_data_to_task_cfg(key, already_exist=False)
    my_sim.from_self_to_sim_cfg(key)
    my_sim.task_cfg = task_cfg
    success_num = my_sim.inference()
    print(f"[INFO] success_num: {success_num}")
    my_sim.from_data_to_task_cfg(key, already_exist=True)
    env.close()
    return True 

def main():
    base_dir = Path.cwd()
    key = args_cli.key
    sim_heuristic_manip(key, args_cli)

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
