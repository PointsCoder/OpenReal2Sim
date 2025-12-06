"""
Heuristic manipulation policy in Isaac Lab simulation.
Using grasping and motion planning to perform object manipulation tasks.
"""
from __future__ import annotations

# ─────────── AppLauncher ───────────
import argparse, os, json, random, transforms3d
from pathlib import Path
import numpy as np
import torch
import yaml
import sys
from isaaclab.app import AppLauncher
from typing import Optional, List
file_path = Path(__file__).resolve()
import imageio 

sys.path.append(str(file_path.parent))
sys.path.append(str(file_path.parent.parent))
from envs.task_cfg import TaskCfg, TaskType, SuccessMetric, SuccessMetricType, TrajectoryCfg
from envs.task_construct import construct_task_config, add_reference_trajectory, load_task_cfg, add_generated_trajectories
from envs.randomizer import Randomizer
from envs.running_cfg import get_rollout_config, get_randomizer_config
# ─────────── CLI ───────────
parser = argparse.ArgumentParser("sim_policy")
parser.add_argument("--key", type=str, default="demo_video", help="scene key (outputs/<key>)")
parser.add_argument("--robot", type=str, default="franka")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments (overrides running_cfg)")
parser.add_argument("--total_num", type=int, default=None, help="Total number of trajectories required (overrides running_cfg)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.headless = True # headless mode for batch execution
app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

# ─────────── Runtime imports ───────────
import isaaclab.sim as sim_utils
from isaaclab.utils.math import subtract_frame_transforms

# ─────────── Simulation environments ───────────
from sim_base import BaseSimulator, get_next_demo_id
from sim_env_factory import make_env
from sim_utils.transform_utils import pose_to_mat, mat_to_pose, grasp_to_world, grasp_approach_axis_batch
from sim_utils.sim_utils import load_sim_parameters

BASE_DIR   = Path.cwd()

out_dir    = BASE_DIR / "outputs"

def pose_distance(T1, T2):
    """
    Compute translation and rotation (angle) distance between two SE(3) transformation matrices in torch.
    Args:
        T1, T2: [..., 4, 4] torch.Tensor or np.ndarray, can be batched
    Returns:
        trans_dist: translation distance(s)
        angle: rotational angle(s)
    """
    if not torch.is_tensor(T1):
        T1 = torch.tensor(T1, dtype=torch.float32)
    if not torch.is_tensor(T2):
        T2 = torch.tensor(T2, dtype=torch.float32)
    
    # Translation distance
    t1 = T1[..., :3, 3]
    t2 = T2[..., :3, 3]
    trans_dist = torch.norm(t2 - t1, dim=-1)

    # Rotation difference (angle)
    R1 = T1[..., :3, :3]
    R2 = T2[..., :3, :3]
    dR = torch.matmul(R2, R1.transpose(-2, -1))
    trace = dR[..., 0, 0] + dR[..., 1, 1] + dR[..., 2, 2]
    cos_angle = (trace - 1) / 2
    
    # Handle numerical precision
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    angle = torch.acos(cos_angle)
    return trans_dist, angle

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
    final_gripper_close_list = []
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
        final_gripper_close_list.append(traj_cfg.final_gripper_close)
        end_pose_list.append(np.array(traj_cfg.success_metric.end_pose, dtype=np.float32))
  
    return robot_poses_list, object_poses_dict, object_trajectory_list, final_gripper_state_list, pregrasp_pose_list, grasp_pose_list, end_pose_list

def get_initial_com_pose(traj_cfg):
    if traj_cfg.init_manip_object_com is not None:
        return traj_cfg.init_manip_object_com
    else:
        return None


# ────────────────────────────Heuristic Manipulation ────────────────────────────
class RandomizeExecution(BaseSimulator):
    """
    Physical trial-and-error grasping with approach-axis perturbation:
      • Multiple grasp proposals executed in parallel;
      • Every attempt does reset → pre → grasp → close → lift → check;
      • Early stops when any env succeeds; then re-exec for logging.
    """
    def __init__(self, sim, scene, sim_cfgs: dict, data_dir: Path, record: bool = True, args_cli: Optional[argparse.Namespace] = None, bg_rgb: Optional[np.ndarray] = None):
        robot_pose = torch.tensor(
            sim_cfgs["robot_cfg"]["robot_pose"],
            dtype=torch.float32,
            device=sim.device,

        )
        super().__init__(
            sim=sim, sim_cfgs=sim_cfgs, scene=scene, args=args_cli,
            robot_pose=robot_pose, cam_dict=sim_cfgs["cam_cfg"],
            out_dir=out_dir, img_folder=args_cli.key, data_dir=data_dir,
            enable_motion_planning=True,
            set_physics_props=True, debug_level=0,
        )

        self.selected_object_id = sim_cfgs["demo_cfg"]["manip_object_id"]
        self._selected_object_id = str(self.selected_object_id)  # Store as string for mapping
        self._update_object_prim()  # Update object_prim based on selected_object_id
        self.record = record  # Store whether to record data
        #self.traj_cfg_list = traj_cfg_list
       
        self.task_type = sim_cfgs["demo_cfg"]["task_type"]
        self.goal_offset = [0, 0, sim_cfgs["demo_cfg"]["goal_offset"]]
        
       
    

    def reset(self, env_ids=None):
        super().reset(env_ids)
        device = self.object_prim.device
        if env_ids is None:
            env_ids_t = self._all_env_ids.to(device)  # (B,)
        else:
            env_ids_t = torch.as_tensor(env_ids, device=device, dtype=torch.long).view(-1)  # (M,)
        M = int(env_ids_t.shape[0])

        # --- object pose/vel: set object at env origins with identity quat ---
        env_origins = self.scene.env_origins.to(device)[env_ids_t]  # (M,3)
        
        # Set poses for all objects from object_poses_dict
        from sim_env_factory import get_prim_name_from_oid
        for oid in self.object_poses_dict.keys():
            # Get prim name from oid
            prim_name = get_prim_name_from_oid(str(oid))
            
            object_prim = self.scene[prim_name]
            
            # Get pose for this object (first pose in the list for now)
            # object_poses_dict[oid] is a list of (pos, quat) tuples from mat_to_pose
            if len(self.object_poses_dict[oid]) == 0:
                continue
            #import ipdb; ipdb.set_trace()
            pos, quat = np.array(self.object_poses_dict[oid], dtype = np.float32)[env_ids_t.cpu().numpy(), :3], np.array(self.object_poses_dict[oid], dtype = np.float32)[env_ids_t.cpu().numpy(), 3:7]
            object_pose = torch.zeros((M, 7), device=device, dtype=torch.float32)
            object_pose[:, :3] = env_origins + torch.tensor(pos, dtype=torch.float32, device=device)
            object_pose[:, 3:7] = torch.tensor(quat, dtype=torch.float32, device=device)  # wxyz
            
            object_prim.write_root_pose_to_sim(object_pose, env_ids=env_ids_t)
            object_prim.write_root_velocity_to_sim(
                torch.zeros((M, 6), device=device, dtype=torch.float32), env_ids=env_ids_t
            )
        
            object_prim.write_data_to_sim()
        
        rp_local = np.array(self.robot_poses_list, dtype=np.float32)
        env_origins_robot = self.scene.env_origins.to(device)[env_ids_t]
        import copy
        robot_pose_world = copy.deepcopy(rp_local)
        robot_pose_world[:, :3] = env_origins_robot.cpu().numpy() + robot_pose_world[env_ids_t.cpu().numpy(), :3]
        #robot_pose_world[:, 3:7] = [1.0, 0.0, 0.0, 0.0]
        self.robot.write_root_pose_to_sim(torch.tensor(robot_pose_world, dtype=torch.float32, device=device), env_ids=env_ids_t)
        self.robot.write_root_velocity_to_sim(
            torch.zeros((M, 6), device=device, dtype=torch.float32), env_ids=env_ids_t
        )


        joint_pos = self.robot.data.default_joint_pos.to(self.robot.device)[env_ids_t]  # (M,7)
        joint_vel = self.robot.data.default_joint_vel.to(self.robot.device)[env_ids_t]  # (M,7)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_t)
        self.robot.write_data_to_sim()

        self.clear_data()

    def compute_components(self):
        self.robot_poses_list, self.object_poses_dict, self.object_trajectory_list, self.final_gripper_state_list, self.pregrasp_pose_list, self.grasp_pose_list, self.end_pose_list = compute_poses_from_traj_cfg(self.traj_cfg_list)
       


    def compute_object_goal_traj(self):
        B = self.scene.num_envs
        # obj_state = self.object_prim.data.root_com_state_w[:, :7]  # [B,7], pos(3)+quat(wxyz)(4)
        obj_state = self.object_prim.data.root_state_w[:, :7]  # [B,7], pos(3)+quat(wxyz)(4)
        self.show_goal(obj_state[:, :3], obj_state[:, 3:7])

        obj_state_np = obj_state.detach().cpu().numpy().astype(np.float32)
        offset_np = np.asarray(self.goal_offset, dtype=np.float32)
        obj_state_np[:, :3] += offset_np  # raise a bit to avoid collision

        # Note: here the relative traj Δ_w is defined in world frame with origin (0,0,0),
        # Hence, we need to normalize it to each env's origin frame.
        origins = self.scene.env_origins.detach().cpu().numpy().astype(np.float32)  # (B,3)
        obj_state_np[:, :3] -= origins # normalize to env origin frame

        # —— 3) Precompute absolute object goals for all envs ——
        T = self.object_trajectory_list[0].shape[0]
        obj_goal = np.zeros((B, T, 4, 4), dtype=np.float32)
        print(f"[DEBUG] env_origins[0]: {origins[0]}")
        print(f"[DEBUG] offset_np: {offset_np}")
        
        for b in range(B):
            for t in range(T):
                goal_4x4 = self.object_trajectory_list[b][t].copy()  # Make a copy to avoid modifying original
                # object_trajectory_list is relative to env origin, so add origin to get world frame
                goal_4x4[:3, 3] += origins[b]  
                goal_4x4[:3, 3] += offset_np
                obj_goal[b, t] = goal_4x4
                
        self.obj_goal_traj_w = obj_goal


    def execute_and_lift_once_batch(self, lift_height=0.12) -> tuple[np.ndarray, np.ndarray]:
        """
        Test grasp quality: close → lift → check coupling in all dimensions.
        A good grasp should maintain:
        - XY position (object shouldn't slip sideways)
        - Roll, Pitch, Yaw orientation (object shouldn't rotate unexpectedly)
        - Z height (tight coupling during vertical lift)
        
        Returns (success[B], score[B]).
        """
        B = self.scene.num_envs
        
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
        jp, success = self.move_to(p_lift_b, q_lift_b, gripper_open=False, record=self.record)
        if torch.any(success==False): return np.zeros(B, bool), np.zeros(B, np.float32)
        jp = self.wait(gripper_open=False, steps=8, record=self.record)

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
        z_coupling = torch.abs(ee_z_diff - obj_z_diff) <= 0.01  # [B]
        z_lifted = (torch.abs(ee_z_diff) >= 0.5 * lift_height) & \
                   (torch.abs(obj_z_diff) >= 0.5 * lift_height)  # [B]
        
        # --- Check 2: XY position stability (no lateral slip) ---
        # Object XY should move with gripper (within 2cm tolerance)
        ee_xy_diff = ee_p1[:, :2] - ee_p0[:, :2]  # [B, 2]
        obj_xy_diff = obj1[:, :2] - obj0[:, :2]    # [B, 2]
        xy_deviation = torch.norm(ee_xy_diff - obj_xy_diff, dim=1)  # [B]
        xy_stable = xy_deviation <= 0.02  # [B]
        
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
        orientation_stable = (roll_diff <= torch.deg2rad(torch.tensor(10.0, device=roll_diff.device))) & \
                            (pitch_diff <= torch.deg2rad(torch.tensor(10.0, device=pitch_diff.device))) & \
                            (yaw_diff <= torch.deg2rad(torch.tensor(10.0, device=yaw_diff.device)))  # [B]
        
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
        jp, success = self.move_to(p_lift_b, q_lift_b, gripper_open=gripper_open, record=self.record)
        jp = self.wait(gripper_open=gripper_open, steps=steps, record=self.record)
        return jp

    def follow_object_goals(self, start_joint_pos, sample_step=1, recalibrate_interval = 3, visualize=True):
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

        t_iter = list(range(1, T, sample_step))
        t_iter = t_iter + [T-1] if t_iter[-1] != T-1 else t_iter


        ee_pos_initial = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:3]
        obj_pos_initial = self.object_prim.data.root_com_pos_w[:, 0:3]
        initial_grasp_dist = torch.norm(ee_pos_initial - obj_pos_initial, dim=1) # [B]
        self.initial_grasp_dist = initial_grasp_dist
        
        # Store T_ee_in_obj for success checking
        T_ee_in_obj = None
        
        for t in t_iter:
            if recalibrate_interval> 0 and (t-1) % recalibrate_interval == 0:
                ee_w  = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]  # [B,7]
                obj_w = self.object_prim.data.root_state_w[:, :7]                                 # [B,7]
                T_ee_in_obj = []
                for b in range(B):
                    T_ee_w  = pose_to_mat(ee_w[b, :3],  ee_w[b, 3:7])
                    T_obj_w = pose_to_mat(obj_w[b, :3], obj_w[b, 3:7])
                    T_ee_in_obj.append((np.linalg.inv(T_obj_w) @ T_ee_w).astype(np.float32))
                # Store for success checking
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
            self.save_dict["actions"].append(np.concatenate([ee_pos_b.cpu().numpy(), ee_quat_b.cpu().numpy(), np.ones((B, 1))], axis=1))

        is_grasp_success = self.is_grasp_success()
        is_success = self.is_success()

        print('[INFO] last obj goal', obj_goal_all[:, -1])
        print('[INFO] last obj pos', self.object_prim.data.root_state_w[:, :3])
        for b in range(B):
            if self.final_gripper_state_list[b]:
                self.wait(gripper_open=False, steps=10, record = self.record)
            else:
                self.wait(gripper_open=True, steps=10, record = self.record)

        return joint_pos, is_success


    def follow_object_centers(self, start_joint_pos, sample_step=1, recalibrate_interval = 3, visualize=True):
        B = self.scene.num_envs
        obj_goal_all = self.obj_goal_traj_w  # [B, T, 4, 4]
        T = obj_goal_all.shape[1]

        joint_pos = start_joint_pos
        root_w = self.robot.data.root_state_w[:, 0:7]  # robot base poses per env

        t_iter = list(range(1, T, sample_step))
        t_iter = t_iter + [T-1] if t_iter[-1] != T-1 else t_iter

        ee_pos_initial = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:3]
        obj_pos_initial = self.object_prim.data.root_com_pos_w[:, 0:3]
        initial_grasp_dist = torch.norm(ee_pos_initial - obj_pos_initial, dim=1) # [B]
        self.initial_grasp_dist = initial_grasp_dist


        for t in t_iter:
            if recalibrate_interval> 0 and (t-1) % recalibrate_interval == 0:
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
                print(f"[INFO] recalibrated at step {t}/{T}")

            goal_pos_list, goal_quat_list = [], []
            print(f"[INFO] follow object goal step {t}/{T}")
            for b in range(B):
                T_obj_goal = obj_goal_all[b, t]          
                trans_offset = T_obj_goal - T_obj_ws[b]
                T_ee_goal  = T_ee_ws[b] + trans_offset
                pos_b, quat_b = mat_to_pose(T_ee_goal)

                goal_pos_list.append(pos_b.astype(np.float32))
                goal_quat_list.append(quat_b.astype(np.float32))

            goal_pos  = torch.as_tensor(np.stack(goal_pos_list),  dtype=torch.float32, device=self.sim.device)
            goal_quat = ee_w[:, 3:7]

            if visualize:
                self.show_goal(goal_pos, goal_quat)
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_w[:, :3], root_w[:, 3:7], goal_pos, goal_quat
            )
            joint_pos, success = self.move_to(ee_pos_b, ee_quat_b, gripper_open=False, record = self.record)

            print(obj_goal_all[:,t])
            print(self.object_prim.data.root_state_w[:, :7])
            self.save_dict["actions"].append(np.concatenate([ee_pos_b.cpu().numpy(), ee_quat_b.cpu().numpy(), np.ones((B, 1))], axis=1))
        is_grasp_success = self.is_grasp_success()
        print('[INFO] last obj goal', obj_goal_all[:, -1])
        print('[INFO] last obj pos', self.object_prim.data.root_state_w[:, :3])
        for b in range(B):
            if self.final_gripper_state_list[b]:
                self.wait(gripper_open=False, steps=10, record = self.record)
            else:
                self.wait(gripper_open=True, steps=10, record = self.record)

        return joint_pos, is_grasp_success

    def refine_grasp_pose(self, init_manip_object_com, grasp_pose, pregrasp_pose):
        current_manip_object_com = self.object_prim.data.root_com_pos_w[:, :3].cpu().numpy()
        current_manip_object_com -= self.scene.env_origins.cpu().numpy()
        grasp_pose_w = grasp_pose
        pregrasp_pose_w = pregrasp_pose
        grasp_pose_w[:, :3] += current_manip_object_com - init_manip_object_com
        pregrasp_pose_w[:, :3] += current_manip_object_com - init_manip_object_com
        return grasp_pose_w, pregrasp_pose_w

    # def is_success(self):
    #     obj_w = self.object_prim.data.root_state_w[:, :7]
    #     origins = self.scene.env_origins
    #     obj_w[:, :3] -= origins
    #     trans_dist_list = []
    #     angle_list = []
    #     B = self.scene.num_envs
    #     for b in range(B):
    #         obj_pose_l = pose_to_mat(obj_w[b, :3], obj_w[b, 3:7])
    #         goal_pose_l = pose_to_mat(self.end_pose_list[b][:3], self.end_pose_list[b][3:7])
    #         trans_dist, angle = pose_distance(obj_pose_l, goal_pose_l)
    #         trans_dist_list.append(trans_dist)
    #         angle_list.append(angle)
    #     trans_dist = torch.tensor(np.stack(trans_dist_list))
    #     angle = torch.tensor(np.stack(angle_list))
    #     print(f"[INFO] trans_dist: {trans_dist}, angle: {angle}")
    #     if self.task_type == "simple_pick_place" or self.task_type == "simple_pick":
    #         is_success = trans_dist < 0.10
    #     elif self.task_type == "targetted_pick_place":
    #         is_success = (trans_dist < 0.10) & (angle < np.radians(10))
    #     else:
    #         raise ValueError(f"[ERR] Invalid task type: {self.task_type}")
    #     return is_success.cpu().numpy()

    
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
            grasp_deviation = torch.abs(current_grip_com_dist - self.initial_grasp_dist)
            holding_success = grasp_deviation <= 0.02
            
            holding_metric = grasp_deviation
            holding_threshold = 0.02
            holding_metric_name = "Grip-COM Deviation"
        else:
            holding_success = current_grip_com_dist <= 0.15
            holding_metric = current_grip_com_dist
            holding_threshold = 0.15
            holding_metric_name = "Grip-COM Dist"

        # Combined Success
        success_mask = obj_success & gripper_success & holding_success
       
        print("="*50)
        print(f"TASK VERIFIER: SUCCESS - {success_mask.sum().item()}/{B} environments succeeded")
        print("="*50 + "\n")
        
        return success_mask


    def is_grasp_success(self):
        ee_w  = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        obj_w = self.object_prim.data.root_com_pos_w[:, 0:3]
        dist = torch.norm(obj_w[:, :3] - ee_w[:, :3], dim=1) # [B]
        print(f"[INFO] dist: {dist}")
        return (dist < 0.15).cpu().numpy()


    def viz_object_goals(self, sample_step=1, hold_steps=20):
        self.reset()
        self.wait(gripper_open=True, steps=10, record = False)
        B = self.scene.num_envs
        env_ids = torch.arange(B, device=self.object_prim.device, dtype=torch.long)
        goals = self.obj_goal_traj_w
        t_iter = list(range(0, goals.shape[1], sample_step))
        t_iter = t_iter + [goals.shape[1]-1] if t_iter[-1] != goals.shape[1]-1 else t_iter
        t_iter = t_iter[-1:]
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
    def build_grasp_info(
        self,
        grasp_pos_w_batch: np.ndarray,   # (B,3)  GraspNet proposal in world frame
        grasp_quat_w_batch: np.ndarray,  # (B,4)  wxyz
        pregrasp_pos_w_batch: np.ndarray,
        pregrasp_quat_w_batch: np.ndarray,

    ) -> dict:
        B = self.scene.num_envs
        p_w   = np.asarray(grasp_pos_w_batch,  dtype=np.float32).reshape(B, 3)
        q_w   = np.asarray(grasp_quat_w_batch, dtype=np.float32).reshape(B, 4)
        pre_p_w = np.asarray(pregrasp_pos_w_batch, dtype=np.float32).reshape(B, 3)
        pre_q_w = np.asarray(pregrasp_quat_w_batch, dtype=np.float32).reshape(B, 4)
    

        origins = self.scene.env_origins.detach().cpu().numpy().astype(np.float32)  # (B,3)
        pre_p_w = pre_p_w + origins
        p_w = p_w + origins

        pre_pb, pre_qb = self._to_base(pre_p_w, pre_q_w)
        pb, qb = self._to_base(p_w, q_w)

        return {
            "pre_p_w": pre_p_w, "p_w": p_w, "q_w": q_w,
            "pre_p_b": pre_pb,  "pre_q_b": pre_qb,
            "p_b": pb,      "q_b": qb,
        }
    

    def inference(self, std: float = 0.0) -> list[int]:
        """
        Main function of the heuristic manipulation policy.
        Physical trial-and-error grasping with approach-axis perturbation:
          • Multiple grasp proposals executed in parallel;
          • Every attempt does reset → pre → grasp → close → lift → check;
          • Early stops when any env succeeds; then re-exec for logging.
        """
        B = self.scene.num_envs

        #self.wait(gripper_open=True, steps=10, record = self.record)
        # reset and conduct main process: open→pre→grasp→close→follow_object_goals
        self.reset()

        cam_p = self.camera.data.pos_w
        cam_q = self.camera.data.quat_w_ros
        gp_w  = torch.as_tensor(np.array(self.grasp_pose_list,  dtype=np.float32)[:,:3], dtype=torch.float32, device=self.sim.device)
        gq_w  = torch.as_tensor(np.array(self.grasp_pose_list, dtype=np.float32)[:,3:7], dtype=torch.float32, device=self.sim.device)
        pre_w = torch.as_tensor(np.array(self.pregrasp_pose_list, dtype=np.float32)[:,:3], dtype=torch.float32, device=self.sim.device)
        init_manip_object_com = get_initial_com_pose(self.task_cfg.reference_trajectory[-1])
        if init_manip_object_com is not None:
            gp_w, pre_w = self.refine_grasp_pose(init_manip_object_com, gp_w, pre_w)
            gp_w = torch.as_tensor(gp_w, dtype=torch.float32, device=self.sim.device)
            pre_w = torch.as_tensor(pre_w, dtype=torch.float32, device=self.sim.device)
        gp_cam,  gq_cam  = subtract_frame_transforms(cam_p, cam_q, gp_w,  gq_w)
        pre_cam, pre_qcm = subtract_frame_transforms(cam_p, cam_q, pre_w, gq_w)
        self.save_dict["grasp_pose_cam"]    = torch.cat([gp_cam,  gq_cam],  dim=1).unsqueeze(0).cpu().numpy()
        self.save_dict["pregrasp_pose_cam"] = torch.cat([pre_cam, pre_qcm], dim=1).unsqueeze(0).cpu().numpy()

        jp = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
        self.wait(gripper_open=True, steps=4, record = self.record)

        # pre → grasp
        info_all = self.build_grasp_info(gp_w.cpu().numpy(), gq_w.cpu().numpy(), pre_w.cpu().numpy(), gq_w.cpu().numpy())
        jp, success = self.move_to(info_all["pre_p_b"], info_all["pre_q_b"], gripper_open=True, record = self.record)
        if torch.any(success==False): return []
        self.save_dict["actions"].append(np.concatenate([info_all["pre_p_b"].cpu().numpy(), info_all["pre_q_b"].cpu().numpy(), np.zeros((B, 1))], axis=1))
        jp = self.wait(gripper_open=True, steps=3, record = self.record)

        jp, success = self.move_to(info_all["p_b"], info_all["q_b"], gripper_open=True, record = self.record)
        if torch.any(success==False): return []
        self.save_dict["actions"].append(np.concatenate([info_all["p_b"].cpu().numpy(), info_all["q_b"].cpu().numpy(), np.zeros((B, 1))], axis=1))

        # close gripper
        jp = self.wait(gripper_open=False, steps=50, record = self.record)
        self.save_dict["actions"].append(np.concatenate([info_all["p_b"].cpu().numpy(), info_all["q_b"].cpu().numpy(), np.ones((B, 1))], axis=1))

        # Validate grasp quality with lift test
        print("[INFO] Validating grasp quality with lift test...")
        grasp_ok, grasp_scores = self.execute_and_lift_once_batch(lift_height=0.12)
        grasp_success_count = int(grasp_ok.sum())
        print(f"[INFO] Grasp validation: {grasp_success_count}/{B} grasps passed tight coupling test")
        
        # Return indices: success=True for passed, False for failed grasps
        # Caller will regenerate poses for failed envs
        grasp_passed_mask = torch.tensor(grasp_ok, dtype=torch.bool, device=self.sim.device)
        
        if grasp_success_count == 0:
            print("[WARN] All grasps failed validation, need to regenerate all poses")
            return {"success_env_ids": [], "failed_env_ids": list(range(B))}
        
        # Only proceed with trajectory following for envs that passed grasp check
        passed_env_ids = torch.where(grasp_passed_mask)[0].cpu().numpy().tolist()
        failed_env_ids = torch.where(~grasp_passed_mask)[0].cpu().numpy().tolist()
        
        if len(failed_env_ids) > 0:
            print(f"[WARN] {len(failed_env_ids)} envs failed grasp validation: {failed_env_ids}")
            print(f"[INFO] Proceeding with trajectory collection only for successful envs: {passed_env_ids}")

        # object goal following (already lifted from grasp check, so reduce additional lift)
        # Only lift the remaining amount if needed
        remaining_lift = max(0.0, self.goal_offset[2] - 0.12)
        if remaining_lift > 0.01:
            self.lift_up(height=remaining_lift, gripper_open=False, steps=8)
        # if self.task_type == "simple_pick_place" or self.task_type == "simple_pick":
        #     jp, is_success = self.follow_object_centers(jp, sample_step=1, visualize=True)
        # elif self.task_type == "targetted_pick_place":
        #     jp, is_success = self.follow_object_goals(jp, sample_step=1, visualize=True)
        # else:
        #     raise ValueError(f"[ERR] Invalid task type: {self.task_type}")
        #jp = self.follow_object_goals(jp, sample_step=1, visualize=True)
        jp, is_success = self.follow_object_goals(jp, sample_step=1, visualize=True)

        is_success = is_success #& self.is_success()
        # Arrange the output: we want to collect only the successful env ids as a list.
        is_success = torch.tensor(is_success, dtype=torch.bool, device=self.sim.device)
        
        # Only consider envs that passed grasp validation
        is_success = is_success & grasp_passed_mask
        success_env_ids = torch.where(is_success)[0].cpu().numpy().tolist()

        print(f"[INFO] Final success_env_ids (grasp+trajectory): {success_env_ids}")
        if self.record and len(success_env_ids) > 0:
            self.save_data(ignore_keys=["segmask", "depth"], env_ids=success_env_ids, export_hdf5=True)
        
        return {"success_env_ids": success_env_ids, "failed_env_ids": failed_env_ids}

    def run_batch_trajectory(self, traj_cfg_list: List[TrajectoryCfg]):
        self.traj_cfg_list = traj_cfg_list
        self.compute_components()
        self.compute_object_goal_traj()
        
        result = self.inference()
        return result["success_env_ids"], result["failed_env_ids"]



# ──────────────────────────── Entry Point ────────────────────────────

  


def sim_randomize_rollout(keys: list[str], args_cli: argparse.Namespace):
    for key in keys:
        # Load config from running_cfg, allow CLI args to override
        rollout_cfg = get_rollout_config(key)
        randomizer_cfg = get_randomizer_config(key)
        
        total_require_traj_num = args_cli.total_num if args_cli.total_num is not None else rollout_cfg.total_num
        num_envs = args_cli.num_envs if args_cli.num_envs is not None else rollout_cfg.num_envs
        
        print(f"[INFO] Using config for key '{key}': num_envs={num_envs}, total_num={total_require_traj_num}")
        print(f"[INFO] Randomizer config: {randomizer_cfg.to_kwargs()}")
        
        success_trajectory_config_list = []
        task_json_path = BASE_DIR / "tasks" / key / "task.json"
        task_cfg = load_task_cfg(task_json_path)
        randomizer = Randomizer(task_cfg)
        
        # Use randomizer config from running_cfg
        randomizer_kwargs = randomizer_cfg.to_kwargs()
        random_task_cfg_list = randomizer.generate_randomized_scene_cfg(**randomizer_kwargs)

        args_cli.key = key
        sim_cfgs = load_sim_parameters(BASE_DIR, key)
      
        data_dir = BASE_DIR / "h5py" / key
        current_timestep = 0
        env, _ = make_env(
                cfgs=sim_cfgs, num_envs=num_envs,
                device=args_cli.device,
                bg_simplify=False,
            )
        sim, scene = env.sim, env.scene

        my_sim = RandomizeExecution(sim, scene, sim_cfgs=sim_cfgs, data_dir=data_dir, record=True, args_cli=args_cli)
        my_sim.task_cfg = task_cfg
        
        # Track pending env slots and their trajectory configs
        pending_traj_cfgs = []  # List of trajectory configs to try
        
        while len(success_trajectory_config_list) < total_require_traj_num:
            # Fill pending list if needed
            if len(pending_traj_cfgs) < num_envs:
                needed = num_envs - len(pending_traj_cfgs)
                # Generate more random configs if we've used up the initial batch
                if current_timestep >= len(random_task_cfg_list):
                    print(f"[INFO] Generating {needed} additional random trajectory configs...")
                    new_random_cfgs = randomizer.generate_randomized_scene_cfg(**randomizer_kwargs)
                    random_task_cfg_list.extend(new_random_cfgs[:needed])
                
                new_cfgs = random_task_cfg_list[current_timestep: current_timestep + needed]
                pending_traj_cfgs.extend(new_cfgs)
                current_timestep += needed
            
            # Take batch of num_envs configs to try
            traj_cfg_list = pending_traj_cfgs[:num_envs]
            
            # Run batch and get success/failure info
            success_env_ids, failed_env_ids = my_sim.run_batch_trajectory(traj_cfg_list)
            
            # Collect successful trajectories
            if len(success_env_ids) > 0:
                for env_id in success_env_ids:
                    success_trajectory_config_list.append(traj_cfg_list[env_id])
                    print(f"[INFO] Collected trajectory {len(success_trajectory_config_list)}/{total_require_traj_num}")
            
            # Remove all attempted configs from pending
            pending_traj_cfgs = pending_traj_cfgs[num_envs:]
            
            # For failed envs, generate new random poses (they'll be added to pending in next iteration)
            if len(failed_env_ids) > 0:
                print(f"[INFO] {len(failed_env_ids)} envs need new randomized poses, will retry in next batch")
            
            print(f"[INFO] Progress: {len(success_trajectory_config_list)}/{total_require_traj_num} trajectories collected")
           
        env.close()

        # for timestep in range(len(success_trajectory_config_list),10):
        #     traj_cfg_list = random_task_cfg_list[timestep: min(timestep + 10, len(random_task_cfg_list))]
        #     my_sim = RandomizeExecution(sim, scene, sim_cfgs=sim_cfgs, traj_cfg_list=traj_cfg_list, record=True)
        #     success_env_ids = my_sim.inference()
        #     del my_sim
        #     torch.cuda.empty_cache()
        add_generated_trajectories(task_cfg, success_trajectory_config_list, task_json_path.parent)
    
    return success_trajectory_config_list

def main():
    base_dir = Path.cwd()
    cfg = yaml.safe_load((base_dir / "config" / "config.yaml").open("r"))
    keys = cfg["keys"]
    sim_randomize_rollout(keys, args_cli)
  

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        simulation_app.close()
