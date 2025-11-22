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
args_cli.headless = True  # headless mode for batch execution
app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

# ─────────── Runtime imports ───────────
import isaaclab.sim as sim_utils
from isaaclab.utils.math import subtract_frame_transforms



# ─────────── Simulation environments ───────────
from sim_base import BaseSimulator, get_next_demo_id
from sim_env_factory import make_env
from isaaclab.sim_utils.transform_utils import pose_to_mat, mat_to_pose, grasp_to_world, grasp_approach_axis_batch
from isaaclab.sim_utils.sim_utils import load_sim_parameters

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



# ────────────────────────────Heuristic Manipulation ────────────────────────────
class RandomizeExecution(BaseSimulator):
    """
    Physical trial-and-error grasping with approach-axis perturbation:
      • Multiple grasp proposals executed in parallel;
      • Every attempt does reset → pre → grasp → close → lift → check;
      • Early stops when any env succeeds; then re-exec for logging.
    """
    def __init__(self, sim, scene, sim_cfgs: dict, demo_dir: Path, data_dir: Path, record: bool = True, args_cli: Optional[argparse.Namespace] = None, bg_rgb: Optional[np.ndarray] = None):
        robot_pose = torch.tensor(
            sim_cfgs["robot_cfg"]["robot_pose"],
            dtype=torch.float32,
            device=sim.device,

        )
        super().__init__(
            sim=sim, sim_cfgs=sim_cfgs, scene=scene, args=args_cli,
            robot_pose=robot_pose, cam_dict=sim_cfgs["cam_cfg"],
            out_dir=out_dir, img_folder=args_cli.key, data_dir=data_dir,
            demo_dir=demo_dir,
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
        for b in range(B):
            T_init = pose_to_mat(obj_state_np[b, :3], obj_state_np[b, 3:7])  # (4,4)
            for t in range(1, T):
                goal = self.object_trajectory_list[b][t] #@ T_init
                goal[:3, 3] += origins[b]  # back to world frame
                goal[:3, 3] += self.goal_offset
                obj_goal[b, t] = goal

        self.obj_goal_traj_w = obj_goal
    
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

        T_ee_in_obj = []
        for b in range(B):
            T_ee_w  = pose_to_mat(ee_w[b, :3],  ee_w[b, 3:7])
            T_obj_w = pose_to_mat(obj_w[b, :3], obj_w[b, 3:7])
            T_ee_in_obj.append((np.linalg.inv(T_obj_w) @ T_ee_w).astype(np.float32))

        joint_pos = start_joint_pos
        root_w = self.robot.data.root_state_w[:, 0:7]  # robot base poses per env

        t_iter = list(range(0, T, sample_step))
        t_iter = t_iter + [T-1] if t_iter[-1] != T-1 else t_iter

        for t in t_iter:
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
            joint_pos, success = self.move_to(ee_pos_b, ee_quat_b, gripper_open=False, record = self.record)
            self.save_dict["actions"].append(np.concatenate([ee_pos_b.cpu().numpy(), ee_quat_b.cpu().numpy(), np.ones((B, 1))], axis=1))

        is_grasp_success = self.is_grasp_success()
        for b in range(B):
            if self.final_gripper_state_list[b]:
                self.wait(gripper_open=False, steps=10, record = self.record)
            else:      
                self.wait(gripper_open=True, steps=10, record = self.record)

        return joint_pos, is_grasp_success
    

    def follow_object_centers(self, start_joint_pos, sample_step=1, visualize=True):
        B = self.scene.num_envs
        obj_goal_all = self.obj_goal_traj_w  # [B, T, 4, 4]
        T = obj_goal_all.shape[1]

        ee_w  = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]  # [B,7]
        # obj_w = self.object_prim.data.root_com_state_w[:, :7]                                 # [B,7]
        obj_w = self.object_prim.data.root_state_w[:, :7]                                 # [B,7]

        T_ee_in_obj = []
        for b in range(B):
            T_ee_w  = pose_to_mat(ee_w[b, :3],  ee_w[b, 3:7])
            T_obj_w = pose_to_mat(obj_w[b, :3], obj_w[b, 3:7])
            T_ee_in_obj.append((np.linalg.inv(T_obj_w) @ T_ee_w).astype(np.float32))

        joint_pos = start_joint_pos
        root_w = self.robot.data.root_state_w[:, 0:7]  # robot base poses per env

        t_iter = list(range(0, T, sample_step))
        t_iter = t_iter + [T-1] if t_iter[-1] != T-1 else t_iter

        for t in t_iter:
            goal_pos_list, goal_quat_list = [], []
            print(f"[INFO] follow object goal step {t}/{T}")
            for b in range(B):
                T_obj_goal = obj_goal_all[b, t]            # (4,4)
                T_ee_goal  = T_obj_goal @ T_ee_in_obj[b]   # (4,4)
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
            self.save_dict["actions"].append(np.concatenate([ee_pos_b.cpu().numpy(), ee_quat_b.cpu().numpy(), np.ones((B, 1))], axis=1))
        is_grasp_success = self.is_grasp_success()
        for b in range(B):
            if self.final_gripper_state_list[b]:
                self.wait(gripper_open=False, steps=10, record = self.record)
            else:
                self.wait(gripper_open=True, steps=10, record = self.record)

        return joint_pos, is_grasp_success


    def is_success(self):
        obj_w = self.object_prim.data.root_state_w[:, :7]
        origins = self.scene.env_origins
        obj_w[:, :3] -= origins
        trans_dist_list = []
        angle_list = []
        B = self.scene.num_envs
        for b in range(B):
            obj_pose_l = pose_to_mat(obj_w[b, :3], obj_w[b, 3:7])
            goal_pose_l = pose_to_mat(np.array(self.end_pose_list[b], dtype=np.float32)[:3], np.array(self.end_pose_list[b], dtype=np.float32)[3:7])
            trans_dist, angle = pose_distance(obj_pose_l, goal_pose_l)
            trans_dist_list.append(trans_dist)
            angle_list.append(angle)
        trans_dist = torch.tensor(trans_dist_list, dtype=torch.float32, device=self.sim.device)
        angle = torch.tensor(angle_list, dtype=torch.float32, device=self.sim.device)
        
        print(f"[INFO] trans_dist: {trans_dist}, angle: {angle}")
        if self.task_type == "simple_pick_place" or self.task_type == "simple_pick":
            is_success = trans_dist < 0.10
        elif self.task_type == "targetted_pick_place":
            is_success = (trans_dist < 0.10) & (angle < np.radians(10))
        else:
            raise ValueError(f"[ERR] Invalid task type: {self.task_type}")
        return is_success.cpu().numpy()

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
        """
        返回与 _build_info 相同结构，但每个 env 的抓取都可不同。
        """
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

        self.wait(gripper_open=True, steps=10, record = self.record)

        
        # reset and conduct main process: open→pre→grasp→close→follow_object_goals
        self.reset()
        #self.viz_object_goals()


        cam_p = self.camera.data.pos_w
        cam_q = self.camera.data.quat_w_ros
        gp_w  = torch.as_tensor(np.array(self.grasp_pose_list,  dtype=np.float32)[:,:3], dtype=torch.float32, device=self.sim.device)
        gq_w  = torch.as_tensor(np.array(self.grasp_pose_list, dtype=np.float32)[:,3:7], dtype=torch.float32, device=self.sim.device)
        pre_w = torch.as_tensor(np.array(self.pregrasp_pose_list, dtype=np.float32)[:,:3], dtype=torch.float32, device=self.sim.device)
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

        # object goal following
        # self.lift_up(height=0.05, gripper_open=False)
        if self.task_type == "simple_pick_place" or self.task_type == "simple_pick":
            jp, is_success = self.follow_object_centers(jp, sample_step=1, visualize=True)
        elif self.task_type == "targetted_pick_place":
            jp, is_success = self.follow_object_goals(jp, sample_step=1, visualize=True)
        else:
            raise ValueError(f"[ERR] Invalid task type: {self.task_type}")
        #jp = self.follow_object_goals(jp, sample_step=1, visualize=True)

        is_success = is_success & self.is_success()
        # Arrange the output: we want to collect only the successful env ids as a list.
        is_success = torch.tensor(is_success, dtype=torch.bool, device=self.sim.device)
        success_env_ids = torch.where(is_success)[0].cpu().numpy().tolist()

        print(f"[INFO] success_env_ids: {success_env_ids}")
        if self.record:
            self.save_data(ignore_keys=["segmask", "depth"], env_ids=success_env_ids, export_hdf5=True)
        
        return success_env_ids

    def run_batch_trajectory(self, traj_cfg_list: List[TrajectoryCfg]):
        self.traj_cfg_list = traj_cfg_list
        self.compute_components()
        self.compute_object_goal_traj()
        
        return self.inference()



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
      
        demo_dir = BASE_DIR / "tasks" / key / "demos"
        data_dir = BASE_DIR / "h5py" / key
        current_timestep = 0
        env, _ = make_env(
                cfgs=sim_cfgs, num_envs=num_envs,
                device=args_cli.device,
                bg_simplify=False,
            )
        sim, scene = env.sim, env.scene
        bg_rgb_path = task_cfg.bg_rgb_path

        bg_rgb = imageio.imread(bg_rgb_path)
        my_sim = RandomizeExecution(sim, scene, sim_cfgs=sim_cfgs, demo_dir=demo_dir, data_dir=data_dir, record=True, args_cli=args_cli, bg_rgb=bg_rgb)
        my_sim.task_cfg = task_cfg
        while len(success_trajectory_config_list) < total_require_traj_num:
            traj_cfg_list = random_task_cfg_list[current_timestep: current_timestep + num_envs]
            current_timestep += num_envs
           
            success_env_ids = my_sim.run_batch_trajectory(traj_cfg_list)
            
            env.close()
            if len(success_env_ids) > 0:
                for env_id in success_env_ids:
                    success_trajectory_config_list.append(traj_cfg_list[env_id])
            
            print(f"[INFO] success_trajectory_config_list: {len(success_trajectory_config_list)}")
            print(total_require_traj_num)

        # for timestep in range(len(success_trajectory_config_list),10):
        #     traj_cfg_list = random_task_cfg_list[timestep: min(timestep + 10, len(random_task_cfg_list))]
        #     my_sim = RandomizeExecution(sim, scene, sim_cfgs=sim_cfgs, traj_cfg_list=traj_cfg_list, demo_dir=demo_dir, record=True)
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
    main()
    simulation_app.close()