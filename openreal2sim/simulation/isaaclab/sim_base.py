# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
import random
import shutil
from pathlib import Path
from typing import Any, Optional, Dict, Sequence, Tuple
from typing import List

import numpy as np
import torch
import imageio
import cv2
import h5py
import sys
file_path = Path(__file__).resolve()
sys.path.append(str(file_path.parent))
sys.path.append(str(file_path.parent.parent))
from modules.envs.task_cfg import CameraInfo, TaskCfg, TrajectoryCfg

# Isaac Lab
import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera
from isaaclab.managers import SceneEntityCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms, transform_points, unproject_depth
from isaaclab.devices import Se3Keyboard, Se3SpaceMouse, Se3Gamepad
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


import curobo
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import join_path, load_yaml, get_robot_configs_path

def get_next_demo_id(demo_root: Path) -> int:
    if not demo_root.exists():
        return 0
    demo_ids = []
    for name in os.listdir(demo_root):
        if name.startswith("demo_"):
            try:
                demo_ids.append(int(name.split("_")[1]))
            except Exception:
                pass
    return max(demo_ids) + 1 if demo_ids else 0

class BaseSimulator:
    """
    Base class for robot simulation.

    Attributes:
      self.sim, self.scene, self.sim_dt
      self.robot, self.object_prim, self.background_prim
      self.teleop_interface, self.sim_state_machine
      self.diff_ik_cfg, self.diff_ik_controller
      self.ee_goals, self.current_goal_idx, self.ik_commands
      self.robot_entity_cfg, self.robot_gripper_cfg
      self.gripper_open_tensor, self.gripper_close_tensor
      self.ee_jacobi_idx, self.count, self.demo_id
      self.camera, self.save_dict
      self.selected_object_id, self.obj_rel_traj, self.debug_level
      self._goal_vis, self._traj_vis
    """

    def __init__(
        self,
        sim: sim_utils.SimulationContext,
        scene: Any,  # InteractiveScene
        *,
        args,
        robot_pose: torch.Tensor,
        cam_dict: Dict,
        out_dir: Path,
        img_folder: str,
        set_physics_props: bool = True,
        enable_motion_planning: bool = True,
        debug_level: int = 1,
        demo_dir: Optional[Path] = None,
        data_dir: Optional[Path] = None,
        task_cfg: Optional[TaskCfg] = None,
        traj_cfg_list: Optional[List[TrajectoryCfg]] = None,
    ) -> None:
        # basic simulation setup
        self.sim: sim_utils.SimulationContext = sim
        self.scene = scene
        self.sim_dt = sim.get_physics_dt()

        self.num_envs: int = int(scene.num_envs)
        self._all_env_ids = torch.arange(self.num_envs, device=sim.device, dtype=torch.long)

        self.cam_dict = cam_dict
        self.out_dir: Path = out_dir
        self.img_folder: str = img_folder
        self.defined_demo_dir: Optional[Path] = demo_dir
        if self.defined_demo_dir is not None:
            self.defined_demo_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_dir: Optional[Path] = data_dir
        if self.data_dir is not None:
            self.data_dir.mkdir(parents=True, exist_ok=True)
        # scene entities
        self.robot = scene["robot"]
        if robot_pose.ndim == 1:
            self.robot_pose = robot_pose.view(1, -1).repeat(self.num_envs, 1).to(self.robot.device)
        else:
            assert robot_pose.shape[0] == self.num_envs and robot_pose.shape[1] == 7, \
                f"robot_pose must be [B,7], got {robot_pose.shape}"
            self.robot_pose = robot_pose.to(self.robot.device).contiguous()
        self.task_cfg = task_cfg
        self.traj_cfg_list = traj_cfg_list
        # Get object prim based on selected_object_id
        # Default to object_00 for backward compatibility
        # Note: selected_object_id is set in subclasses after super().__init__()
        # So we use a helper method that can be called later
        self._selected_object_id = None  # Will be set by subclasses
        self.object_prim = scene["object_00"]  # Default, will be updated if needed
        self._update_object_prim()
        
        # Get all other object prims (excluding the main object)
        self.other_object_prims = [scene[key] for key in scene.keys() 
                                   if f"object_" in key and key != "object_00"]
        self.background_prim = scene["background"]
        self.camera: Camera = scene["camera"]

        # physics properties
        if set_physics_props:
            static_friction = 5.0
            dynamic_friction = 5.0
            restitution = 0.0

            # object: rigid prim -> has root_physx_view
            if hasattr(self.object_prim, "root_physx_view") and self.object_prim.root_physx_view is not None:
                obj_view = self.object_prim.root_physx_view
                obj_mats = obj_view.get_material_properties()
                vals_obj = torch.tensor([static_friction, dynamic_friction, restitution],
                                        device=obj_mats.device, dtype=obj_mats.dtype)
                obj_mats[:] = vals_obj
                obj_view.set_material_properties(obj_mats, self._all_env_ids.to(obj_mats.device))

            # background: GroundPlaneCfg -> XFormPrim (no root_physx_view); skip if unavailable
            if hasattr(self.background_prim, "root_physx_view") and self.background_prim.root_physx_view is not None:
                bg_view = self.background_prim.root_physx_view
                bg_mats = bg_view.get_material_properties()
                vals_bg = torch.tensor([static_friction, dynamic_friction, restitution],
                                       device=bg_mats.device, dtype=bg_mats.dtype)
                bg_mats[:] = vals_bg
                bg_view.set_material_properties(bg_mats, self._all_env_ids.to(bg_mats.device))

        # ik controller
        self.diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls"
        )
        self.diff_ik_controller = DifferentialIKController(
            self.diff_ik_cfg, num_envs=self.num_envs, device=sim.device
        )

        # robot: joints / gripper / jacobian indices
        self.robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
        self.robot_gripper_cfg = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"], body_names=["panda_hand"])
        self.robot_entity_cfg.resolve(scene)
        self.robot_gripper_cfg.resolve(scene)
        self.gripper_open_tensor = 0.04 * torch.ones(
            (self.num_envs, len(self.robot_gripper_cfg.joint_ids)), device=self.robot.device
        )
        self.gripper_close_tensor = torch.zeros(
            (self.num_envs, len(self.robot_gripper_cfg.joint_ids)), device=self.robot.device
        )
        if self.robot.is_fixed_base:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0] - 1
        else:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0]

        # demo count and data saving
        self.count = 0
        self.demo_id = 0
        self.save_dict = {
            "rgb": [], "depth": [], "segmask": [],
            "joint_pos": [], "joint_vel": [], "actions": [],
            "gripper_pos": [], "gripper_cmd": [],
            "composed_rgb": []  # composed rgb image with background and foreground
        }

        # visualization
        self.selected_object_id = 0
        self._selected_object_id = 0
        self.debug_level = debug_level

        self.goal_vis_list = []
        
        if self.debug_level > 0:
            for b in range(self.num_envs):
                cfg = VisualizationMarkersCfg(
                    prim_path=f"/Visuals/ee_goal/env_{b:03d}",
                    markers={
                        "frame": sim_utils.UsdFileCfg(
                            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                            scale=(0.06, 0.06, 0.06),
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                        ),
                    },
                )
                self.goal_vis_list.append(VisualizationMarkers(cfg))

        # curobo motion planning
        self.enable_motion_planning = enable_motion_planning
        if self.enable_motion_planning:
            print(f"prepare curobo motion planning: {enable_motion_planning}")
            self.prepare_curobo()
            print("curobo motion planning ready.")
        
    def _update_object_prim(self):
        """Update object_prim based on selected_object_id. Called after selected_object_id is set."""
        if self._selected_object_id is None:
            return
        try:
            from sim_env_factory import get_prim_name_from_oid
            oid_str = str(self._selected_object_id)
            prim_name = get_prim_name_from_oid(oid_str)
            if prim_name in self.scene:
                self.object_prim = self.scene[prim_name]
                # Update other_object_prims
                self.other_object_prims = [self.scene[key] for key in self.scene.keys() 
                                           if f"object_" in key and key != prim_name]
        except (ImportError, ValueError, KeyError) as e:
            # Fallback to object_00 if mapping not available
            pass

    # -------- Curobo Motion Planning ----------
    def prepare_curobo(self):
        setup_curobo_logger("error")
        # tensor_args = TensorDeviceType()
        tensor_args = TensorDeviceType(device=self.sim.device, dtype=torch.float32)
        curobo_path = curobo.__file__.split("/__init__")[0]
        robot_file = f"{curobo_path}/content/configs/robot/franka.yml"
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_cfg=robot_file, world_model=None, tensor_args=tensor_args,
            interpolation_dt=self.sim_dt,
            use_cuda_graph=True if self.num_envs == 1 else False,
        )
        self.motion_gen = MotionGen(motion_gen_config)
        if self.num_envs == 1:
            self.motion_gen.warmup(enable_graph=True)
        _ = RobotConfig.from_dict(
            load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"], tensor_args
        )

    # ---------- Helpers ----------
    def _ensure_batch_pose(self, p, q):
        """Ensure position [B,3], quaternion [B,4] on device."""
        B = self.scene.num_envs
        p = torch.as_tensor(p, dtype=torch.float32, device=self.sim.device)
        q = torch.as_tensor(q, dtype=torch.float32, device=self.sim.device)
        if p.ndim == 1: p = p.view(1, -1).repeat(B, 1)
        if q.ndim == 1: q = q.view(1, -1).repeat(B, 1)
        return p.contiguous(), q.contiguous()

    def _traj_to_BT7(self, traj):
        """Normalize various curobo traj.position shapes to [B, T, 7]."""
        B = self.scene.num_envs
        pos = traj.position  # torch or numpy
        pos = torch.as_tensor(pos, device=self.sim.device, dtype=torch.float32)

        if pos.ndim == 3:
            # candidate shapes: [B,T,7] or [T,B,7]
            if pos.shape[0] == B and pos.shape[-1] == 7:
                return pos  # [B,T,7]
            if pos.shape[1] == B and pos.shape[-1] == 7:
                return pos.permute(1, 0, 2).contiguous()  # [B,T,7]
        elif pos.ndim == 2 and pos.shape[-1] == 7:
            # [T,7] → broadcast to all envs
            return pos.unsqueeze(0).repeat(B, 1, 1)
        # Fallback: flatten and infer
        flat = pos.reshape(-1, 7)                          # [B*T,7]
        T = flat.shape[0] // B
        return flat.view(B, T, 7).contiguous()

    # ---------- Planning / Execution (Single) ----------
    def motion_planning_single(self, position, quaternion, max_attempts=1, use_graph=True):
        """
        single environment planning: prefer plan_single (supports graph / CUDA graph warmup better).
        Returns [1, T, 7], returns None on failure.
        """
        # current joint position
        joint_pos0 = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids][0:1].contiguous()  # [1,7]
        start_state = JointState.from_position(joint_pos0)

        # goal (ensure [1,3]/[1,4])
        pos_b, quat_b = self._ensure_batch_pose(position, quaternion)
        pos_b  = pos_b[0:1]
        quat_b = quat_b[0:1]
        goal_pose  = Pose(position=pos_b, quaternion=quat_b)

        plan_cfg = MotionGenPlanConfig(max_attempts=max_attempts, enable_graph=use_graph)

        result = self.motion_gen.plan_single(start_state, goal_pose, plan_cfg)

        traj = result.get_interpolated_plan()  # JointState

        if result.success[0] == True:
            T = traj.position.shape[-2]
            BT7 = traj.position.to(self.sim.device).to(torch.float32).unsqueeze(0)  # [1,T,7]
        else:
            print(f"[WARN] motion planning failed.")
            BT7 = joint_pos0.unsqueeze(1)  # [1,1,7]

        return BT7, result.success

    # ---------- Planning / Execution (Batched) ----------
    def motion_planning_batch(self, position, quaternion, max_attempts=1, allow_graph=False):
        """
        multi-environment planning: use plan_batch.
        Default require_all=True: if any env fails, return None (keep your original semantics).
        Returns [B, T, 7].
        """
        B = self.scene.num_envs
        joint_pos = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids].contiguous()  # [B,7]
        start_state = JointState.from_position(joint_pos)

        pos_b, quat_b = self._ensure_batch_pose(position, quaternion)  # [B,3], [B,4]
        goal_pose  = Pose(position=pos_b, quaternion=quat_b)

        plan_cfg = MotionGenPlanConfig(max_attempts=max_attempts, enable_graph=allow_graph)

        result = self.motion_gen.plan_batch(start_state, goal_pose, plan_cfg)

        try:
            paths = result.get_paths()  # List[JointState]
            T_max = 1
            for i, p in enumerate(paths):
                if not result.success[i]:
                    print(f"[WARN] motion planning failed for env {i}.")
                else:
                    T_max = max(T_max, int(p.position.shape[-2]))
            dof = joint_pos.shape[-1]
            BT7 = torch.zeros((B, T_max, dof), device=self.sim.device, dtype=torch.float32)
            for i, p in enumerate(paths):
                if result.success[i] == False:
                    BT7[i, :, :] = joint_pos[i:i+1, :].unsqueeze(1).repeat(1, T_max, 1)
                else:
                    Ti = p.position.shape[-2]
                    BT7[i, :Ti, :] = p.position.to(self.sim.device).to(torch.float32)
                    if Ti < T_max:
                        BT7[i, Ti:, :] = BT7[i, Ti-1:Ti, :]
        except Exception as e:
            print(f"[WARN] motion planning all failed with exception: {e}")
            success = torch.zeros(B, dtype=torch.bool, device=self.sim.device) # set to all false
            BT7 = joint_pos.unsqueeze(1)  # [B,1,7]

        # check exceptions
        if result.success is None or result.success.shape[0] != B:
            print(f"[WARN] motion planning success errors: {result.success}")
            success = torch.zeros(B, dtype=torch.bool, device=self.sim.device) # set to all false
            BT7 = joint_pos.unsqueeze(1)  # [B,1,7]
        else:
            success = result.success
        if BT7.shape[0] != B or BT7.shape[2] != joint_pos.shape[1]:
            print(f"[WARN] motion planning traj dim mismatch: {BT7.shape} vs {[B,'T',joint_pos.shape[1]]}")
            BT7 = joint_pos.unsqueeze(1)  # [B,1,7]

        return BT7, success

    def motion_planning(self, position, quaternion, max_attempts=1):
        if self.scene.num_envs == 1:
            return self.motion_planning_single(position, quaternion, max_attempts=max_attempts, use_graph=True)
        else:
            return self.motion_planning_batch(position, quaternion, max_attempts=max_attempts, allow_graph=False)

    def move_to_motion_planning(self, position: torch.Tensor, quaternion: torch.Tensor, gripper_open: bool = True, record: bool = True) -> torch.Tensor:
        """
        Cartesian space control: Move the end effector to the desired position and orientation using motion planning.
        Works with batched envs. If inputs are 1D, they will be broadcast to all envs.
        """
        traj, success = self.motion_planning(position, quaternion)
        BT7 = traj
        T = BT7.shape[1]
        last = None
        for i in range(T):
            joint_pos_des = BT7[:, i, :]  # [B,7]
            self.apply_actions(joint_pos_des, gripper_open=gripper_open)
            obs = self.get_observation(gripper_open=gripper_open)
            if record:
                self.record_data(obs)
            last = joint_pos_des
        return last, success

    # ---------- Visualization ----------
    def show_goal(self, pos, quat):
        """
        show a pose with visual marker(s).
          - if [B,3]/[B,4], update all envs;
          - if [3]/[4] or [1,3]/[1,4], default to update env 0;
          - optional env_ids specify a subset of envs to update; when a single pose is input, it will be broadcast to these envs.
        """
        if self.debug_level == 0:
            print("debug_level=0, skip visualization.")
            return

        if not isinstance(pos, torch.Tensor):
            pos_t = torch.tensor(pos, dtype=torch.float32, device=self.sim.device)
            quat_t = torch.tensor(quat, dtype=torch.float32, device=self.sim.device)
        else:
            pos_t = pos.to(self.sim.device)
            quat_t = quat.to(self.sim.device)

        if pos_t.ndim == 1:
            pos_t = pos_t.view(1, -1)
        if quat_t.ndim == 1:
            quat_t = quat_t.view(1, -1)

        B = self.num_envs

        if pos_t.shape[0] == B:
            for b in range(B):
                self.goal_vis_list[b].visualize(pos_t[b:b+1], quat_t[b:b+1])
        else:
            self.goal_vis_list[0].visualize(pos_t, quat_t)

    def set_robot_pose(self, robot_pose: torch.Tensor):
        if robot_pose.ndim == 1:
            self.robot_pose = robot_pose.view(1, -1).repeat(self.num_envs, 1).to(self.robot.device)
        else:
            assert robot_pose.shape[0] == self.num_envs and robot_pose.shape[1] == 7, \
                f"robot_pose must be [B,7], got {robot_pose.shape}"
            self.robot_pose = robot_pose.to(self.robot.device).contiguous()

    # ---------- Environment Step ----------
    def step(self):
        self.scene.write_data_to_sim()
        self.sim.step()
        self.camera.update(dt=self.sim_dt)
        self.count += 1
        self.scene.update(self.sim_dt)

    # ---------- Apply actions to robot joints ----------
    def apply_actions(self, joint_pos_des, gripper_open: bool = True):
        # joint_pos_des: [B, n_joints]
        self.robot.set_joint_position_target(joint_pos_des, joint_ids=self.robot_entity_cfg.joint_ids)
        if gripper_open:
            self.robot.set_joint_position_target(self.gripper_open_tensor, joint_ids=self.robot_gripper_cfg.joint_ids)
        else:
            self.robot.set_joint_position_target(self.gripper_close_tensor, joint_ids=self.robot_gripper_cfg.joint_ids)
        self.step()

    # ---------- EE control ----------
    def move_to(self, position: torch.Tensor, quaternion: torch.Tensor, gripper_open: bool = True, record: bool = True) -> torch.Tensor:
        if self.enable_motion_planning:
            return self.move_to_motion_planning(position, quaternion, gripper_open=gripper_open, record=record)
        else:
            return self.move_to_ik(position, quaternion, gripper_open=gripper_open, record=record)

    def move_to_ik(self, position: torch.Tensor, quaternion: torch.Tensor, steps: int = 50, gripper_open: bool = True, record: bool = True) -> torch.Tensor:
        """
        Cartesian space control: Move the end effector to the desired position and orientation using inverse kinematics.
        Works with batched envs. If inputs are 1D, they will be broadcast to all envs.

        Early-stop when both position and orientation errors are within tolerances.
        'steps' now acts as a max-iteration cap; the loop breaks earlier on convergence.
        """
        # Ensure [B,3]/[B,4] tensors on device
        position, quaternion = self._ensure_batch_pose(position, quaternion)

        # IK command (world frame goals)
        ee_goals = torch.cat([position, quaternion], dim=1).to(self.sim.device).float()
        self.diff_ik_controller.reset()
        self.diff_ik_controller.set_command(ee_goals)

        # Tolerances (you can tune if needed)
        pos_tol = 3e-3                 # meters
        ori_tol = 3.0 * np.pi / 180.0  # radians (~3 degrees)

        # Interpret 'steps' as max iterations; early-stop on convergence
        max_steps = int(steps) if steps is not None and steps > 0 else 10_000

        joint_pos_des = None
        for _ in range(max_steps):
            # Current EE pose (world) and Jacobian
            jacobian = self.robot.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids]
            ee_pose_w = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
            root_pose_w = self.robot.data.root_state_w[:, 0:7]
            joint_pos = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]

            # Current EE pose expressed in robot base
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )

            # Compute next joint command
            joint_pos_des = self.diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
           
            # Apply
            self.apply_actions(joint_pos_des, gripper_open=gripper_open)

            # Optional recording
            if record:
                obs = self.get_observation(gripper_open=gripper_open)
                self.record_data(obs)

            # --- Early-stop check ---
            # Desired EE pose in base frame (convert world goal -> base)
            des_pos_b, des_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], position, quaternion
            )
            # Position error [B]
            pos_err = torch.norm(des_pos_b - ee_pos_b, dim=1)
            # Orientation error [B]: angle between quaternions
            # Note: q and -q are equivalent -> take |dot|
            dot = torch.sum(des_quat_b * ee_quat_b, dim=1).abs().clamp(-1.0, 1.0)
            ori_err = 2.0 * torch.acos(dot)

            done = (pos_err <= pos_tol) & (ori_err <= ori_tol)
            if bool(torch.all(done)):
                break

        return joint_pos_des

    # ---------- Robot Waiting ----------
    def wait(self, gripper_open, steps: int, record: bool = True):
        joint_pos_des = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids].clone()
        for _ in range(steps):
            self.apply_actions(joint_pos_des, gripper_open=gripper_open)
            obs = self.get_observation(gripper_open=gripper_open)
            if record:
                self.record_data(obs)
        return joint_pos_des

    # ---------- Reset Envs ----------
    def reset(self, env_ids=None):
        """
        Reset all envs or only those in env_ids.
        Assumptions:
          - self.robot_pose.shape == (B, 7)        # base pose per env (wxyz in [:,3:])
          - self.robot.data.default_joint_pos == (B, 7)
          - self.robot.data.default_joint_vel == (B, 7)
        """
        device = self.object_prim.device
        if env_ids is None:
            env_ids_t = self._all_env_ids.to(device)  # (B,)
        else:
            env_ids_t = torch.as_tensor(env_ids, device=device, dtype=torch.long).view(-1)  # (M,)
        M = int(env_ids_t.shape[0])

        # --- object pose/vel: set object at env origins with identity quat ---
        env_origins = self.scene.env_origins.to(device)[env_ids_t]  # (M,3)
        object_pose = torch.zeros((M, 7), device=device, dtype=torch.float32)
        object_pose[:, :3] = env_origins
        object_pose[:, 3]  = 1.0  # wxyz = [1,0,0,0]
        self.object_prim.write_root_pose_to_sim(object_pose, env_ids=env_ids_t)
        self.object_prim.write_root_velocity_to_sim(
            torch.zeros((M, 6), device=device, dtype=torch.float32), env_ids=env_ids_t
        )
        self.object_prim.write_data_to_sim()
        for prim in self.other_object_prims:
            prim.write_root_pose_to_sim(object_pose, env_ids=env_ids_t)
            prim.write_root_velocity_to_sim(
                torch.zeros((M, 6), device=device, dtype=torch.float32), env_ids=env_ids_t
            )
            prim.write_data_to_sim()

        # --- robot base pose/vel ---
        # robot_pose is (B,7) in *local* base frame; add env origin offset per env
        rp_local = self.robot_pose.to(self.robot.device)[env_ids_t]          # (M,7)
        env_origins_robot = env_origins.to(self.robot.device)                # (M,3)
        robot_pose_world = rp_local.clone()
        robot_pose_world[:, :3] = env_origins_robot + robot_pose_world[:, :3]
        #print(f"[INFO] robot_pose_world: {robot_pose_world}")
        self.robot.write_root_pose_to_sim(robot_pose_world, env_ids=env_ids_t)
        self.robot.write_root_velocity_to_sim(
            torch.zeros((M, 6), device=self.robot.device, dtype=torch.float32), env_ids=env_ids_t
        )

        # --- joints (B,7) -> select ids (M,7) ---
        joint_pos = self.robot.data.default_joint_pos.to(self.robot.device)[env_ids_t]  # (M,7)
        joint_vel = self.robot.data.default_joint_vel.to(self.robot.device)[env_ids_t]  # (M,7)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_t)
        self.robot.write_data_to_sim()

        # housekeeping
        self.clear_data()

    # ---------- Get Observations ----------
    def get_observation(self, gripper_open) -> Dict[str, torch.Tensor]:
        # camera outputs (already batched)
        rgb = self.camera.data.output["rgb"]                           # [B,H,W,3]
        depth = self.camera.data.output["distance_to_image_plane"]     # [B,H,W]
        ins_all = self.camera.data.output["instance_id_segmentation_fast"]  # [B,H,W]

        B, H, W, _ = ins_all.shape
        fg_mask_list = []
        obj_mask_list = []
        for b in range(B):
            ins_id_seg = ins_all[b]
            id_mapping = self.camera.data.info[b]["instance_id_segmentation_fast"]["idToLabels"]
            fg_mask_b = torch.zeros_like(ins_id_seg, dtype=torch.bool, device=ins_id_seg.device)
            obj_mask_b = torch.zeros_like(ins_id_seg, dtype=torch.bool, device=ins_id_seg.device)
            for key, value in id_mapping.items():
                if "object" in value:
                    fg_mask_b |= (ins_id_seg == key)
                    obj_mask_b |= (ins_id_seg == key)
                if "Robot" in value:
                    fg_mask_b |= (ins_id_seg == key)
            fg_mask_list.append(fg_mask_b)
            obj_mask_list.append(obj_mask_b)
        fg_mask = torch.stack(fg_mask_list, dim=0)   # [B,H,W]
        obj_mask = torch.stack(obj_mask_list, dim=0) # [B,H,W]

        ee_pose_w = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        joint_pos = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
        joint_vel = self.robot.data.joint_vel[:, self.robot_entity_cfg.joint_ids]
        gripper_pos = self.robot.data.joint_pos[:, self.robot_gripper_cfg.joint_ids]
        gripper_cmd = self.gripper_open_tensor if gripper_open else self.gripper_close_tensor

        cam_pos_w = self.camera.data.pos_w
        cam_quat_w = self.camera.data.quat_w_ros
        ee_pos_cam, ee_quat_cam = subtract_frame_transforms(
            cam_pos_w, cam_quat_w, ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        ee_pose_cam = torch.cat([ee_pos_cam, ee_quat_cam], dim=1)

        points_3d_cam = unproject_depth(
            self.camera.data.output["distance_to_image_plane"], self.camera.data.intrinsic_matrices
        )
        points_3d_world = transform_points(points_3d_cam, self.camera.data.pos_w, self.camera.data.quat_w_ros)

        object_center = self.object_prim.data.root_com_pos_w[:, :3]

        return {
            "rgb": rgb,
            "depth": depth,
            "fg_mask": fg_mask,
            "joint_pos": joint_pos,
            "gripper_pos": gripper_pos,
            "gripper_cmd": gripper_cmd,
            "joint_vel": joint_vel,
            "ee_pose_cam": ee_pose_cam,
            "ee_pose_w": ee_pose_w,
            "object_mask": obj_mask,
            "points_cam": points_3d_cam,
            "points_world": points_3d_world,
            "object_center": object_center,
        }

    # ---------- Task Completion Verifier ----------
    def is_success(self) -> bool:
        raise NotImplementedError("BaseSimulator.is_success() should be implemented in subclass.")

    # ---------- Data Recording & Saving & Clearing ----------
    def record_data(self, obs: Dict[str, torch.Tensor]):
        self.save_dict["rgb"].append(obs["rgb"].cpu().numpy())          # [B,H,W,3]
        self.save_dict["depth"].append(obs["depth"].cpu().numpy())      # [B,H,W]
        self.save_dict["segmask"].append(obs["fg_mask"].cpu().numpy())  # [B,H,W]
        self.save_dict["joint_pos"].append(obs["joint_pos"].cpu().numpy())  # [B,nJ]
        self.save_dict["gripper_pos"].append(obs["gripper_pos"].cpu().numpy())  # [B,3]
        self.save_dict["gripper_cmd"].append(obs["gripper_cmd"].cpu().numpy())  # [B,1]
        self.save_dict["joint_vel"].append(obs["joint_vel"].cpu().numpy())

    def clear_data(self):
        for key in self.save_dict.keys():
            self.save_dict[key] = []

                        
  
    def _demo_dir(self) -> Path:
        if self.defined_demo_dir is not None:
            return self.defined_demo_dir
        else:
            return self.out_dir / self.img_folder / "demos" / f"demo_{self.demo_id}"

    def _env_dir(self, base: Path, b: int) -> Path:
        d = base / f"env_{b:03d}"
        d.mkdir(parents=True, exist_ok=True)
        return d
    
    def _get_next_demo_dir(self, base: Path) -> Path:
        already_existing_num = len(list(base.iterdir()))
        return base / f"demo_{already_existing_num:03d}"

    def save_data(self, ignore_keys: List[str] = [], env_ids: Optional[List[int]] = None):
        save_root = self._demo_dir()
        save_root.mkdir(parents=True, exist_ok=True)
        
        stacked = {k: np.array(v) for k, v in self.save_dict.items()}
        if env_ids is None:
            env_ids = self._all_env_ids.cpu().numpy()

        composed_rgb = []
        hdf5_names = []
        for b in env_ids:
            if self.defined_demo_dir is  None:
                env_dir = save_root / f"env_{b:03d}"
            else:
                env_dir = self._get_next_demo_dir(save_root)
                hdf5_names.append(env_dir.name)
            env_dir.mkdir(parents=True, exist_ok=True)
            for key, arr in stacked.items():
                if key in ignore_keys: # skip the keys for storage
                    continue
                if key == "rgb":
                    video_path = env_dir / "sim_video.mp4"
                    writer = imageio.get_writer(video_path, fps=50, macro_block_size=None)
                    for t in range(arr.shape[0]):
                        writer.append_data(arr[t, b])
                    writer.close()
                elif key == "segmask":
                    video_path = env_dir / "mask_video.mp4"
                    writer = imageio.get_writer(video_path, fps=50, macro_block_size=None)
                    for t in range(arr.shape[0]):
                        writer.append_data((arr[t, b].astype(np.uint8) * 255))
                    writer.close()
                elif key == "depth":
                    depth_seq = arr[:, b]
                    flat = depth_seq[depth_seq > 0]
                    max_depth = np.percentile(flat, 99) if flat.size > 0 else 1.0
                    depth_norm = np.clip(depth_seq / max_depth * 255.0, 0, 255).astype(np.uint8)
                    video_path = env_dir / "depth_video.mp4"
                    writer = imageio.get_writer(video_path, fps=50, macro_block_size=None)
                    for t in range(depth_norm.shape[0]):
                        writer.append_data(depth_norm[t])
                    writer.close()
                    np.save(env_dir / f"{key}.npy", depth_seq)
                elif key != "composed_rgb":
                    #import pdb; pdb.set_trace()
                    np.save(env_dir / f"{key}.npy", arr[:, b])
            video_path = env_dir / "real_video.mp4"
            writer = imageio.get_writer(video_path, fps=50, macro_block_size=None)
            rgb = np.array(self.save_dict["rgb"])
            mask = np.array(self.save_dict["segmask"])
            bg_rgb_path = self.task_cfg.bg_rgb_path
            self.bg_rgb = imageio.imread(bg_rgb_path)
            for t in range(rgb.shape[0]):
                #import ipdb; ipdb.set_trace()
                composed = self.convert_real(mask[t, b], self.bg_rgb, rgb[t, b])        
                writer.append_data(composed)
            writer.close()
            composed_rgb.append(composed)

   
        self.export_batch_data_to_hdf5(hdf5_names)
        

        print("[INFO]: Demonstration is saved at: ", save_root)

        demo_root = self.out_dir / "all_demos"
        demo_root.mkdir(parents=True, exist_ok=True)
        total_demo_id = get_next_demo_id(demo_root)
        demo_save_path = demo_root / f"demo_{total_demo_id}"
        demo_save_path.mkdir(parents=True, exist_ok=True)
        meta_info = {
            "path": str(save_root),
            "fps": 50,
        }
        with open(demo_save_path / "meta_info.json", "w") as f:
            json.dump(meta_info, f)
        os.system(f"cp -r {save_root}/* {demo_save_path}")
        print("[INFO]: Demonstration is saved at: ", demo_save_path)

    # def collect_data_from_folder(self, folder_path: Path):
    #     """
    #     Load `.npy` files saved by `save_data` and repopulate `self.save_dict`.

    #     The provided directory may either be:
    #       • A demo folder containing multiple `env_XXX` subdirectories, or
    #       • A single `env_XXX` directory.

    #     Args:
    #         folder_path: Path to the data directory.

    #     Returns:
    #         A tuple ``(stacked, env_dirs)`` where ``stacked`` maps data keys to
    #         stacked numpy arrays of shape `[T, B, ...]` (``T`` timesteps,
    #         ``B`` environments) and ``env_dirs`` is the ordered list of
    #         environment directories discovered in ``folder_path``.
    #     """
    #     folder_path = Path(folder_path)
    #     if not folder_path.exists():
    #         raise FileNotFoundError(f"[collect_data_from_folder] path does not exist: {folder_path}")
    #     if folder_path.is_file():
    #         raise NotADirectoryError(f"[collect_data_from_folder] expected a directory, got file: {folder_path}")

    #     if folder_path.name.startswith("env_"):
    #         env_dirs = [folder_path]
    #     else:
    #         env_dirs = sorted(
    #             [p for p in folder_path.iterdir() if p.is_dir() and p.name.startswith("env_")],
    #             key=lambda p: p.name,
    #         )

    #     if len(env_dirs) == 0:
    #         raise ValueError(f"[collect_data_from_folder] no env_XXX directories found in {folder_path}")

    #     aggregated: Dict[str, List[np.ndarray]] = {}
    #     for env_dir in env_dirs:
    #         for npy_file in sorted(env_dir.glob("*.npy")):
    #             key = npy_file.stem
    #             try:
    #                 arr = np.load(npy_file, allow_pickle=False)
    #             except Exception as exc:
    #                 print(f"[collect_data_from_folder] skip {npy_file}: {exc}")
    #                 continue
    #             aggregated.setdefault(key, []).append(arr)

    #     if len(aggregated) == 0:
    #         raise ValueError(f"[collect_data_from_folder] no npy data found in {folder_path}")

    #     stacked: Dict[str, np.ndarray] = {}
    #     for key, env_slices in aggregated.items():
    #         reference_shape = env_slices[0].shape
    #         for idx, slice_arr in enumerate(env_slices[1:], start=1):
    #             if slice_arr.shape != reference_shape:
    #                 raise ValueError(
    #                     f"[collect_data_from_folder] inconsistent shapes for '{key}': "
    #                     f"{reference_shape} vs {slice_arr.shape} (env idx {idx})"
    #                 )
    #         stacked[key] = np.stack(env_slices, axis=1)  # [T, B, ...]

    #     for key in self.save_dict.keys():
    #         if key not in stacked:
    #             self.save_dict[key] = []
    #             continue
    #         data = stacked[key]  # [T, B, ...]
    #         self.save_dict[key] = [data[t] for t in range(data.shape[0])]

    #     return stacked, env_dirs

    def _encode_rgb_sequence(self, frames: np.ndarray) -> tuple[list[bytes], int]:
        """Encode a sequence of RGB frames into JPEG bytes and return padded bytes."""
        if frames.shape[0] == 0:
            return [], 1

        encoded: List[bytes] = []
        max_len = 1
        for frame in frames:
            frame_np = np.asarray(frame)
            if frame_np.dtype != np.uint8:
                frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
            if frame_np.ndim == 3 and frame_np.shape[2] == 4:
                frame_np = frame_np[..., :3]
            if frame_np.ndim == 3 and frame_np.shape[2] == 3:
                frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            success, buffer = cv2.imencode(".jpg", frame_np)
            if not success:
                raise RuntimeError("Failed to encode RGB frame into JPEG for RoboTwin export.")
            data_bytes = buffer.tobytes()
            encoded.append(data_bytes)
            if len(data_bytes) > max_len:
                max_len = len(data_bytes)

        padded = [frame_bytes.ljust(max_len, b"\0") for frame_bytes in encoded]
        return padded, max_len

    def export_batch_data_to_hdf5(self, hdf5_names: List[str]) -> int:
        """Export buffered trajectories to RoboTwin-style HDF5 episodes."""
        if self.data_dir is not None:
            target_root = self.data_dir
        else:
            target_root = self._demo_dir() / "hdf5"
        data_dir = Path(target_root) 
        data_dir.mkdir(parents=True, exist_ok=True)

       
        num_envs = len(hdf5_names)
        stacked = {k: np.array(v) for k, v in self.save_dict.items()}
       
        episode_names = []
        for idx, name in enumerate(hdf5_names):
            name = str(name)
            episode_names.append(name.replace("demo_", "episode_"))
        handled_keys = {
            "rgb",
            "depth",
            "segmask",
            "joint_pos",
            "joint_vel",
            "gripper_pos",
            "gripper_cmd",
            "actions",
        }

        camera_params = self._get_camera_parameters()

        for env_idx, episode_name in enumerate(episode_names):
            hdf5_path = data_dir / f"{episode_name}.hdf5"
            hdf5_path.parent.mkdir(parents=True, exist_ok=True)

            with h5py.File(hdf5_path, "w") as f:
                obs_grp = f.create_group("observation")
                camera_group_name = "head_camera" if camera_params is not None else "camera"
                cam_grp = obs_grp.create_group(camera_group_name)
                if camera_params is not None:
                    intrinsics, extrinsics, resolution = camera_params
                    cam_grp.create_dataset("intrinsics", data=intrinsics)
                    cam_grp.create_dataset("extrinsics", data=extrinsics)
                    cam_grp.attrs["resolution"] = resolution

                if "rgb" in stacked:
                    rgb_frames = stacked["rgb"][:, env_idx]
                    encoded_frames, max_len = self._encode_rgb_sequence(rgb_frames)
                    dtype = f"S{max_len}" if max_len > 0 else "S1"
                    cam_grp.create_dataset("rgb", data=np.asarray(encoded_frames, dtype=dtype))
                    cam_grp.attrs["encoding"] = "jpeg"
                    cam_grp.attrs["channels"] = 3
                    cam_grp.attrs["original_shape"] = rgb_frames.shape
                    if camera_params is None:
                        cam_grp.create_dataset("intrinsics", data=np.zeros((3, 3), dtype=np.float32))
                        cam_grp.create_dataset("extrinsics", data=np.zeros((4, 4), dtype=np.float32))

                if "depth" in stacked:
                    depth_ds = cam_grp.create_dataset("depth", data=stacked["depth"][:, env_idx])
                    depth_ds.attrs["encoding"] = "float32"
                    depth_ds.attrs["unit"] = "meter"
                if "segmask" in stacked:
                    seg_ds = cam_grp.create_dataset(
                        "segmentation",
                        data=stacked["segmask"][:, env_idx].astype(np.uint8),
                    )
                    seg_ds.attrs["encoding"] = "uint8"
                    seg_ds.attrs["color_mapping"] = "instance_id"

                joint_grp = f.create_group("joint_action")
                if "joint_pos" in stacked:
                    joint_grp.create_dataset(
                        "joint_pos", data=stacked["joint_pos"][:, env_idx].astype(np.float32)
                    )
                if "joint_vel" in stacked:
                    joint_grp.create_dataset(
                        "joint_vel", data=stacked["joint_vel"][:, env_idx].astype(np.float32)
                    )
                if "gripper_cmd" in stacked:
                    joint_grp.create_dataset(
                        "gripper_cmd", data=stacked["gripper_cmd"][:, env_idx].astype(np.float32)
                    )
                if len(joint_grp.keys()) == 0:
                    del f["joint_action"]

                if "gripper_pos" in stacked:
                    endpose_grp = f.create_group("endpose")
                    endpose_grp.create_dataset(
                        "gripper_pos", data=stacked["gripper_pos"][:, env_idx].astype(np.float32)
                    )
                    if "gripper_cmd" in stacked:
                        endpose_grp.create_dataset(
                            "gripper_cmd", data=stacked["gripper_cmd"][:, env_idx].astype(np.float32)
                        )

                if "actions" in stacked:
                    f.create_dataset(
                        "actions", data=stacked["actions"][:, env_idx].astype(np.float32)
                    )

                extras = {}
                for key, value in stacked.items():
                    if key in handled_keys:
                        continue

                extras_grp = f.create_group("extras")
                if len(extras) > 0:
                    for key, value in extras.items():
                        extras_grp.create_dataset(key, data=value)

                if self.task_cfg is not None:
                   
                    extras_grp.create_dataset("task_desc", data=self.task_cfg.task_desc)
                
                if self.traj_cfg_list is not None:
                    traj_i = self.traj_cfg_list[env_idx]
                    traj_grp = extras_grp.create_group("traj")
                    traj_grp.create_dataset("robot_pose", data=traj_i.robot_pose)
                    traj_grp.create_dataset("pregrasp_pose", data=traj_i.pregrasp_pose)
                    traj_grp.create_dataset("grasp_pose", data=traj_i.grasp_pose)


                frame_count = stacked["rgb"].shape[0]
                meta_grp = f.create_group("meta")
                meta_grp.attrs["env_index"] = int(env_idx)
                meta_grp.attrs["frame_dt"] = float(self.sim_dt)
                meta_grp.attrs["frame_count"] = int(frame_count)
                meta_grp.attrs["source"] = "OpenReal2Sim"
                meta_grp.attrs["episode_name"] = episode_name
                meta_grp.create_dataset("frame_indices", data=np.arange(frame_count, dtype=np.int32))

        print(f"[INFO]: Exported {num_envs} HDF5 episodes to {data_dir}")
        return num_envs



    def convert_real(self,segmask, bg_rgb, fg_rgb):
        #import pdb; pdb.set_trace()
        segmask_2d = segmask[..., 0]
        composed = bg_rgb.copy()
        composed[segmask_2d] = fg_rgb[segmask_2d]
        return composed

    
    def delete_data(self):
        save_path = self._demo_dir()
        failure_root = self.out_dir / self.img_folder / "demos_failures"
        failure_root.mkdir(parents=True, exist_ok=True)
        fail_demo_id = get_next_demo_id(failure_root)
        failure_path = failure_root / f"demo_{fail_demo_id}"
        os.system(f"mv {save_path} {failure_path}")
        for key in self.save_dict.keys():
            self.save_dict[key] = []
        print("[INFO]: Clear up the folder: ", save_path)

    def _quat_to_rot(self, quat: Sequence[float]) -> np.ndarray:
        w, x, y, z = quat
        rot = np.array(
            [
                [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)],
            ],
            dtype=np.float32,
        )
        return rot

    def _get_camera_parameters(self) -> Optional[Tuple[np.ndarray, np.ndarray, Tuple[int, int]]]:
        if self.task_cfg is None:
            return None
        camera_info = getattr(self.task_cfg, "camera_info", None)
        if camera_info is None:
            return None

        intrinsics = np.array(
            [
                [camera_info.fx, 0.0, camera_info.cx],
                [0.0, camera_info.fy, camera_info.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        if getattr(camera_info, "camera_opencv_to_world", None) is not None:
            extrinsics = np.array(camera_info.camera_opencv_to_world, dtype=np.float32)
        else: 
            extrinsics = np.eye(4, dtype=np.float32)
            if getattr(camera_info, "camera_heading_wxyz", None) is not None:
                rot = self._quat_to_rot(camera_info.camera_heading_wxyz)
            else:
                rot = np.eye(3, dtype=np.float32)
            extrinsics[:3, :3] = rot
            if getattr(camera_info, "camera_position", None) is not None:
                extrinsics[:3, 3] = np.array(camera_info.camera_position, dtype=np.float32)
        resolution = (
            int(camera_info.width),
            int(camera_info.height),
        )
        return intrinsics, extrinsics, resolution
