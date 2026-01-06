from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path
from typing import Any, Optional, Dict, Sequence, Tuple
from typing import List
import copy
import numpy as np
import torch
import imageio
import cv2
import h5py
import sys
file_path = Path(__file__).resolve()
sys.path.append(str(file_path.parent))
sys.path.append(str(file_path.parent.parent))
from envs.task_cfg import CameraInfo, TaskCfg, TrajectoryCfg
from sim_env_factory_demo import get_prim_name_from_oid
# Isaac Lab
import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera
from isaaclab.managers import SceneEntityCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms, transform_points, unproject_depth
from isaaclab.devices import Se3Keyboard, Se3SpaceMouse, Se3Gamepad
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

 # Curobo imports are done lazily in prepare_curobo() to avoid warp/inspect issues with isaaclab namespace package
# Patch inspect.getfile to handle namespace packages (like isaaclab)
# This is needed because warp's set_module_options() uses inspect.stack() which tries to get file paths
# for all modules in the call stack, including namespace packages that don't have a __file__ attribute
import inspect
_original_getfile = inspect.getfile
def _patched_getfile(object):
    """Patched getfile that handles namespace packages."""
    try:
        return _original_getfile(object)
    except TypeError as e:
        if "is a built-in module" in str(e) or "namespace" in str(e).lower():
            # For namespace packages, return a dummy path to avoid errors
            # This allows warp's inspect.stack() to work even when isaaclab is in the call stack
            if hasattr(object, '__name__'):
                return f'<namespace:{object.__name__}>'
            return '<namespace:unknown>'
        raise
inspect.getfile = _patched_getfile

import curobo
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
)
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.devices import Se3Gamepad, Se3Keyboard, Se3SpaceMouse
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors.camera import Camera
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import (
    subtract_frame_transforms,
    transform_points,
    unproject_depth,
)

##-- for sim background to real background video composition--
from PIL import Image 
import cv2  



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
        sim_cfgs: Dict,
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
        save_interval: int = 1,
        decimation: int = 1,
        selected_object_id: Optional[int] = None,
    ) -> None:
        # basic simulation setup
        self.sim: sim_utils.SimulationContext = sim
        self.sim_cfgs = sim_cfgs
        self.scene = scene
        self.sim_dt = sim.get_physics_dt()  # Single physics substep dt
        self.decimation = decimation
        # Task step dt = physics_dt * decimation (time for one task step)
        self.task_dt = self.sim_dt * self.decimation

        self.num_envs: int = int(scene.num_envs)
        self._all_env_ids = torch.arange(
            self.num_envs, device=sim.device, dtype=torch.long
        )

        self.cam_dict = cam_dict
        self.out_dir: Path = out_dir
        self.img_folder: str = img_folder
       
        
        self.data_dir: Optional[Path] = data_dir
        if self.data_dir is not None:
            self.data_dir.mkdir(parents=True, exist_ok=True)
        # scene entities
        self.robot = scene["robot"]
        self.save_interval = save_interval
        if robot_pose.ndim == 1:
            self.robot_pose = (
                robot_pose.view(1, -1).repeat(self.num_envs, 1).to(self.robot.device)
            )
        else:
            assert robot_pose.shape[0] == self.num_envs and robot_pose.shape[1] == 7, (
                f"robot_pose must be [B,7], got {robot_pose.shape}"
            )
            self.robot_pose = robot_pose.to(self.robot.device).contiguous()
        self.task_cfg = task_cfg
        self.traj_cfg_list = traj_cfg_list
        # Get object prim based on selected_object_id
        # Default to object_00 for backward compatibility
        # Note: selected_object_id is set in subclasses after super().__init__()
        # So we use a helper method that can be called later
        self._selected_object_id = selected_object_id  # Will be set by subclasses
        self.object_prim = scene["object_00"]  # Default, will be updated if needed
        self._update_object_prim()
        
        # # Get all other object prims (excluding the main object)
        # self.other_object_prims = [scene[key] for key in scene.keys() 
        #                            if f"object_" in key and key != "object_00"]
        self.background_prim = scene["background"]
        self.camera: Camera = scene["camera"]

        # physics properties
        if set_physics_props:
            static_friction = 5.0
            dynamic_friction = 5.0
            restitution = 0.0

            # object: rigid prim -> has root_physx_view
            if (
                hasattr(self.object_prim, "root_physx_view")
                and self.object_prim.root_physx_view is not None
            ):
                obj_view = self.object_prim.root_physx_view
                obj_mats = obj_view.get_material_properties()
                vals_obj = torch.tensor(
                    [static_friction, dynamic_friction, restitution],
                    device=obj_mats.device,
                    dtype=obj_mats.dtype,
                )
                obj_mats[:] = vals_obj
                obj_view.set_material_properties(
                    obj_mats, self._all_env_ids.to(obj_mats.device)
                )

            # background: GroundPlaneCfg -> XFormPrim (no root_physx_view); skip if unavailable
            if (
                hasattr(self.background_prim, "root_physx_view")
                and self.background_prim.root_physx_view is not None
            ):
                bg_view = self.background_prim.root_physx_view
                bg_mats = bg_view.get_material_properties()
                vals_bg = torch.tensor(
                    [static_friction, dynamic_friction, restitution],
                    device=bg_mats.device,
                    dtype=bg_mats.dtype,
                )
                bg_mats[:] = vals_bg
                bg_view.set_material_properties(
                    bg_mats, self._all_env_ids.to(bg_mats.device)
                )

        # ik controller
        self.diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls"
        )
        self.diff_ik_controller = DifferentialIKController(
            self.diff_ik_cfg, num_envs=self.num_envs, device=sim.device
        )

        # robot: joints / gripper / jacobian indices
        self.robot_entity_cfg = SceneEntityCfg(
            "robot", joint_names=["panda_joint.*"], body_names=["panda_hand"]
        )
        self.robot_gripper_cfg = SceneEntityCfg(
            "robot", joint_names=["panda_finger_joint.*"], body_names=["panda_hand"]
        )
        self.robot_entity_cfg.resolve(scene)
        self.robot_gripper_cfg.resolve(scene)
        self.gripper_open_tensor = 0.04 * torch.ones(
            (self.num_envs, len(self.robot_gripper_cfg.joint_ids)),
            device=self.robot.device,
        )
        self.gripper_close_tensor = torch.zeros(
            (self.num_envs, len(self.robot_gripper_cfg.joint_ids)),
            device=self.robot.device,
        )
        if self.robot.is_fixed_base:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0] - 1
        else:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0]

        # demo count and data saving
        self.count = 0  # Physical step counter
        self.task_step_count = 0  # Task step counter (count // decimation)
        self.demo_id = 0
        self.save_dict = {
            "rgb": [], "depth": [], "segmask": [], "robot_mask": [], "object_mask": [],
            "joint_pos": [], "joint_vel": [], "actions": [], "action_indices": [],
            "gripper_pos": [], "gripper_cmd": [], "ee_pose_cam": [],
            "composed_rgb": [], "joint_pos_des": [], "ee_pose_l": [], # composed rgb image with background and foreground
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
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(1.0, 0.0, 0.0)
                            ),
                        ),
                    },
                )
                self.goal_vis_list.append(VisualizationMarkers(cfg))

        self.joint_pos_des_list = []
        self.gripper_cmd_list = []
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
        oid_str = str(self._selected_object_id)
        prim_name = get_prim_name_from_oid(oid_str)
        if prim_name in self.scene.keys():
            #import pdb; pdb.set_trace()
            self.object_prim = self.scene[prim_name]
            # Update other_object_prims
            self.other_object_prims = []
            for key in self.scene.keys():
                if f"object_" in key and key != prim_name:
                    self.other_object_prims.append(self.scene[key])
    # -------- Curobo Motion Planning ----------
    def prepare_curobo(self):

        setup_curobo_logger("error")
        # tensor_args = TensorDeviceType()
        tensor_args = TensorDeviceType(device=self.sim.device, dtype=torch.float32)
        curobo_path = curobo.__file__.split("/__init__")[0]
        robot_file = f"{curobo_path}/content/configs/robot/franka.yml"
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_cfg=robot_file,
            world_model=None,
            tensor_args=tensor_args,
            interpolation_dt=self.sim_dt,
            use_cuda_graph=True if self.num_envs == 1 else False,
        )
        self.motion_gen = MotionGen(motion_gen_config)
        if self.num_envs == 1:
            self.motion_gen.warmup(enable_graph=True)
        _ = RobotConfig.from_dict(
            load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"],
            tensor_args,
        )

    # ---------- Helpers ----------
    def _ensure_batch_pose(self, p, q):
        """Ensure position [B,3], quaternion [B,4] on device."""
        B = self.scene.num_envs
        p = torch.as_tensor(p, dtype=torch.float32, device=self.sim.device)
        q = torch.as_tensor(q, dtype=torch.float32, device=self.sim.device)
        if p.ndim == 1:
            p = p.view(1, -1).repeat(B, 1)
        if q.ndim == 1:
            q = q.view(1, -1).repeat(B, 1)
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
        flat = pos.reshape(-1, 7)  # [B*T,7]
        T = flat.shape[0] // B
        return flat.view(B, T, 7).contiguous()

    # ---------- Planning / Execution (Single) ----------
    def reinitialize_motion_gen(self):
        """
        Reinitialize the motion generation object.
        Call this after a crash to restore a clean state.
        """
        print("[INFO] Reinitializing motion planner...")
        try:
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Recreate the motion planner
            from curobo.types.base import TensorDeviceType
            from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
            from curobo.util_file import get_robot_configs_path, join_path, load_yaml
            from curobo.types.robot import RobotConfig
            import curobo
            
            tensor_args = TensorDeviceType(device=self.sim.device, dtype=torch.float32)
            curobo_path = curobo.__file__.split("/__init__")[0]
            robot_file = f"{curobo_path}/content/configs/robot/franka.yml"
            
            motion_gen_config = MotionGenConfig.load_from_robot_config(
                robot_cfg=robot_file,
                world_model=None,
                tensor_args=tensor_args,
                interpolation_dt=self.sim_dt,
                use_cuda_graph=True if self.num_envs == 1 else False,
            )
            
            self.motion_gen = MotionGen(motion_gen_config)
            
            if self.num_envs == 1:
                self.motion_gen.warmup(enable_graph=True)
            
            print("[INFO] Motion planner reinitialized successfully")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to reinitialize motion planner: {e}")
            return False

  
    def motion_planning_single(
        self, position, quaternion, max_attempts=1, use_graph=True, max_retries=1
    ):
        """
        Single environment planning with automatic recovery from crashes.
        Returns None on complete failure to signal restart needed.
        """
        joint_pos0 = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids][
            0:1
        ].contiguous()
        
        pos_b, quat_b = self._ensure_batch_pose(position, quaternion)
        pos_b = pos_b[0:1]
        quat_b = quat_b[0:1]
        
        for retry in range(max_retries):
            try:
                start_state = JointState.from_position(joint_pos0)
                goal_pose = Pose(position=pos_b, quaternion=quat_b)
                plan_cfg = MotionGenPlanConfig(
                    max_attempts=max_attempts, enable_graph=use_graph
                )
                
                result = self.motion_gen.plan_single(start_state, goal_pose, plan_cfg)
                
                # Check if result is valid
                if result is None:
                    print(f"[ERROR] Motion planning returned None result on attempt {retry+1}/{max_retries}")
                    if retry < max_retries - 1:
                        if self.reinitialize_motion_gen():
                            print(f"[INFO] Retrying motion planning (attempt {retry+2}/{max_retries})...")
                            continue
                    break
                
                traj = result.get_interpolated_plan()
                
                # Check if trajectory is valid
                if traj is None:
                    print(f"[ERROR] Motion planning returned None trajectory on attempt {retry+1}/{max_retries}")
                    if retry < max_retries - 1:
                        if self.reinitialize_motion_gen():
                            print(f"[INFO] Retrying motion planning (attempt {retry+2}/{max_retries})...")
                            continue
                    break
                
                if result.success[0] == True:
                    BT7 = (
                        traj.position.to(self.sim.device).to(torch.float32).unsqueeze(0)
                    )
                else:
                    print(f"[WARN] Motion planning failed.")
                    BT7 = joint_pos0.unsqueeze(1)
                
                return BT7, result.success
                
            except AttributeError as e:
                print(f"[ERROR] Motion planner crash on attempt {retry+1}/{max_retries}: {e}")
                
                if retry < max_retries - 1:
                    if self.reinitialize_motion_gen():
                        print(f"[INFO] Retrying motion planning (attempt {retry+2}/{max_retries})...")
                        continue
                    else:
                        break
                else:
                    print("[ERROR] Max retries reached")
                    
            except Exception as e:
                # Safe error message extraction
                try:
                    error_msg = str(e)
                    error_type = type(e).__name__
                except:
                    error_msg = "Unknown error"
                    error_type = "Exception"
                
                print(f"[ERROR] Unexpected error: {error_type}: {error_msg}")
                
                # Check for recoverable errors
                is_recoverable = False
                try:
                    is_recoverable = ("cuda graph" in error_msg.lower() or 
                                    "NoneType" in error_msg or 
                                    "has no len()" in error_msg)
                except:
                    pass
                
                if retry < max_retries - 1 and is_recoverable:
                    if self.reinitialize_motion_gen():
                        continue
                break
        
        # Complete failure - return dummy trajectory with False success
        print("[ERROR] Motion planning failed critically - returning dummy trajectory")
        joint_pos0 = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids][0:1].contiguous()
        # Return current position as 1-step trajectory with failure
        dummy_traj = joint_pos0.unsqueeze(1)  # (1, 1, dof)
        dummy_success = torch.zeros(1, dtype=torch.bool, device=self.sim.device)
        return dummy_traj, dummy_success

    # ---------- Planning / Execution (Batched) ----------
   

    def motion_planning_batch(
        self, position, quaternion, max_attempts=1, allow_graph=False, max_retries=1
    ):
        """
        Multi-environment planning with automatic recovery from crashes.
        Returns None on complete failure to signal restart needed.
        """
        B = self.scene.num_envs
        joint_pos = self.robot.data.joint_pos[
            :, self.robot_entity_cfg.joint_ids
        ].contiguous()
        
        pos_b, quat_b = self._ensure_batch_pose(position, quaternion)
        
        for retry in range(max_retries):
            try:
                # Attempt planning
                start_state = JointState.from_position(joint_pos)
                goal_pose = Pose(position=pos_b, quaternion=quat_b)
                plan_cfg = MotionGenPlanConfig(
                    max_attempts=max_attempts, enable_graph=allow_graph
                )
                
                try:
                    result = self.motion_gen.plan_batch(start_state, goal_pose, plan_cfg)
                except Exception as plan_err:
                    print(f"[ERROR] curobo.plan_batch raised exception: {plan_err}")
                    raise plan_err
                
                # Check if result is valid
                if result is None:
                    print(f"[ERROR] Motion planning returned None result on attempt {retry+1}/{max_retries}")
                    if retry < max_retries - 1:
                        if self.reinitialize_motion_gen():
                            print(f"[INFO] Retrying motion planning (attempt {retry+2}/{max_retries})...")
                            continue
                    break
                
                # Process results
                paths = result.get_paths()
                
                # Check if paths is valid - use try-except to safely check if it's iterable
                paths_valid = False
                try:
                    if paths is not None:
                        # Try to get length to verify it's iterable and not empty
                        _ = len(paths)
                        if len(paths) > 0:
                            paths_valid = True
                except:
                    pass
                
                if not paths_valid:
                    print(f"[ERROR] Motion planning returned invalid paths on attempt {retry+1}/{max_retries}")
                    if retry < max_retries - 1:
                        if self.reinitialize_motion_gen():
                            print(f"[INFO] Retrying motion planning (attempt {retry+2}/{max_retries})...")
                            continue
                        else:
                            print("[ERROR] Failed to recover motion planner")
                    # Skip to next retry iteration or exit loop
                    continue
                
                # Double-check paths is still valid (defensive programming)
                if paths is None:
                    print(f"[ERROR] paths became None after validation check")
                    continue
                
                T_max = 1
                
                # Check if result.success is valid
                if result.success is None:
                     print(f"[WARN] result.success is None. Assuming failure for all envs.")
                     # Create dummy failure tensor
                     result.success = torch.zeros(B, dtype=torch.bool, device=self.sim.device)

                try:
                    for i, p in enumerate(paths):
                        if not result.success[i]:
                            print(f"[WARN] Motion planning failed for env {i}.")
                        else:
                            T_max = max(T_max, int(p.position.shape[-2]))
                except TypeError as te:
                    print(f"[ERROR] TypeError when processing paths: {te}")
                    print(f"[DEBUG] paths type: {type(paths)}, paths value: {paths}")
                    continue
                
                dof = joint_pos.shape[-1]
                BT7 = torch.zeros(
                    (B, T_max, dof), device=self.sim.device, dtype=torch.float32
                )
                
                for i, p in enumerate(paths):
                    if result.success[i] == False:
                        BT7[i, :, :] = (
                            joint_pos[i : i + 1, :].unsqueeze(1).repeat(1, T_max, 1)
                        )
                    else:
                        Ti = p.position.shape[-2]
                        BT7[i, :Ti, :] = p.position.to(self.sim.device).to(torch.float32)
                        if Ti < T_max:
                            BT7[i, Ti:, :] = BT7[i, Ti - 1 : Ti, :]
                
                success = result.success if result.success is not None else torch.zeros(
                    B, dtype=torch.bool, device=self.sim.device
                )
                
                # Success! Return the trajectory
                return BT7, success
                
            except AttributeError as e:
                print(f"[ERROR] Motion planner crash on attempt {retry+1}/{max_retries}: {e}")
                
                if retry < max_retries - 1:
                    if self.reinitialize_motion_gen():
                        print(f"[INFO] Retrying motion planning (attempt {retry+2}/{max_retries})...")
                        continue
                    else:
                        print("[ERROR] Failed to recover motion planner")
                        break
                else:
                    print("[ERROR] Max retries reached, motion planning failed critically")
                    
            except Exception as e:
                # Safe error message extraction
                try:
                    error_msg = str(e)
                    error_type = type(e).__name__
                except:
                    error_msg = "Unknown error"
                    error_type = "Exception"
                
                print(f"[ERROR] Unexpected error in motion planning: {error_type}: {error_msg}")
                
                # Check for recoverable errors
                is_recoverable = False
                try:
                    is_recoverable = ("cuda graph" in error_msg.lower() or 
                                    "NoneType" in error_msg or 
                                    "has no len()" in error_msg)
                except:
                    pass
                
                if retry < max_retries - 1 and is_recoverable:
                    if self.reinitialize_motion_gen():
                        print(f"[INFO] Retrying after error (attempt {retry+2}/{max_retries})...")
                        continue
                break
        
        # If we get here, all retries failed - return dummy trajectory with all False success
        print("[ERROR] Motion planning failed critically - returning dummy trajectory")
        B = self.scene.num_envs
        joint_pos = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids].contiguous()
        dof = joint_pos.shape[-1]
        # Return current position as 1-step trajectory with all failures
        dummy_traj = joint_pos.unsqueeze(1)  # (B, 1, dof)
        dummy_success = torch.zeros(B, dtype=torch.bool, device=self.sim.device)
        return dummy_traj, dummy_success



    def motion_planning(self, position, quaternion, max_attempts=1):
        if self.scene.num_envs == 1:
            return self.motion_planning_single(
                position, quaternion, max_attempts=max_attempts, use_graph=True
            )
        else:
            return self.motion_planning_batch(
                position, quaternion, max_attempts=max_attempts, allow_graph=False
            )

    def move_to_motion_planning(
        self,
        position: torch.Tensor,
        quaternion: torch.Tensor,
        gripper_open: bool = True,
    ) -> torch.Tensor:
        """
        Cartesian space control: Move the end effector to the desired position and orientation using motion planning.
        Works with batched envs. If inputs are 1D, they will be broadcast to all envs.
        """
        res = self.motion_planning(position, quaternion)
        if res is None:
            print("[ERROR] motion_planning returned None in move_to_motion_planning!")
            B = self.scene.num_envs
            joint_pos = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
            return joint_pos, torch.zeros(B, dtype=torch.bool, device=self.sim.device)

        traj, success = res
        BT7 = traj

        if BT7 is None:
             print("[ERROR] BT7 is None in move_to_motion_planning!")
             B = self.scene.num_envs
             joint_pos = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
             return joint_pos, torch.zeros(B, dtype=torch.bool, device=self.sim.device)
        T = BT7.shape[1]
        print(f"[INFO] T: {T}")
        last = None
        for i in range(T):
            joint_pos_des = BT7[:, i, :]  # [B,7]
            self.apply_actions(joint_pos_des, gripper_open=gripper_open)
            obs = self.get_observation(gripper_open=gripper_open)
            self.record_data(obs)
            #print(f"[INFO]: joint_pos_des_list length: {len(self.joint_pos_des_list)}, save_dict['joint_pos'] length: {len(self.save_dict['joint_pos'])}")

            assert len(self.joint_pos_des_list) == len(self.save_dict["joint_pos"])
            last = joint_pos_des
        
        self.robot.update(self.sim_dt)
     
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
                self.goal_vis_list[b].visualize(pos_t[b : b + 1], quat_t[b : b + 1])
        else:
            self.goal_vis_list[0].visualize(pos_t, quat_t)

    def set_robot_pose(self, robot_pose: torch.Tensor):
        if robot_pose.ndim == 1:
            self.robot_pose = (
                robot_pose.view(1, -1).repeat(self.num_envs, 1).to(self.robot.device)
            )
        else:
            assert robot_pose.shape[0] == self.num_envs and robot_pose.shape[1] == 7, (
                f"robot_pose must be [B,7], got {robot_pose.shape}"
            )
            self.robot_pose = robot_pose.to(self.robot.device).contiguous()


    # ---------- Environment Step ----------
    def step(self):
        self.scene.write_data_to_sim()
        self.sim.step()
        # Camera update should use task_dt (decimation * sim_dt) since camera
        # is typically updated at task step frequency, not physics substep frequency
        # This matches the render_interval setting
        if self.count % self.decimation == 0:
            self.camera.update(dt=self.task_dt)
            self.task_step_count += 1  # Increment task step counter
        self.count += 1  # Increment physical step counter
        self.scene.update(self.sim_dt)

    # ---------- Apply actions to robot joints ----------
    def apply_actions(self, joint_pos_des, gripper_open: bool = True):
        # joint_pos_des: [B, n_joints]
        # Set joint targets (this happens at task step frequency)
        
        self.robot.set_joint_position_target(
            joint_pos_des, joint_ids=self.robot_entity_cfg.joint_ids
        )
        if gripper_open:
            self.robot.set_joint_position_target(
                self.gripper_open_tensor, joint_ids=self.robot_gripper_cfg.joint_ids
            )
        else:
            self.robot.set_joint_position_target(
                self.gripper_close_tensor, joint_ids=self.robot_gripper_cfg.joint_ids
            )
        
        joint_pos_des = joint_pos_des.cpu().numpy()
        self.joint_pos_des_list.append(joint_pos_des)
        self.gripper_cmd_list.append(gripper_open)
        # Execute decimation number of physics steps to complete one task step
        for _ in range(self.decimation):
            self.step()


    # ---------- EE control ----------
    def move_to(
        self,
        position: torch.Tensor,
        quaternion: torch.Tensor,
        gripper_open: bool = True,
        record: bool = True,
    ) -> torch.Tensor:
        if self.enable_motion_planning:
            return self.move_to_motion_planning(
                position, quaternion, gripper_open=gripper_open)
            
        else:
            return self.move_to_ik(
                position, quaternion, gripper_open=gripper_open)
            

    def move_to_ik(
        self,
        position: torch.Tensor,
        quaternion: torch.Tensor,
        steps: int = 50,
        gripper_open: bool = True,
        record: bool = True,
    ) -> torch.Tensor:
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
        pos_tol = 3e-3  # meters
        ori_tol = 3.0 * np.pi / 180.0  # radians (~3 degrees)

        # Interpret 'steps' as max iterations; early-stop on convergence
        max_steps = int(steps) if steps is not None and steps > 0 else 10_000

        joint_pos_des = None
        for _ in range(max_steps):
            # Current EE pose (world) and Jacobian
            jacobian = self.robot.root_physx_view.get_jacobians()[
                :, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids
            ]
            ee_pose_w = self.robot.data.body_state_w[
                :, self.robot_entity_cfg.body_ids[0], 0:7
            ]
            root_pose_w = self.robot.data.root_state_w[:, 0:7]
            joint_pos = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]

            # Current EE pose expressed in robot base
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3],
                root_pose_w[:, 3:7],
                ee_pose_w[:, 0:3],
                ee_pose_w[:, 3:7],
            )

            # Compute next joint command
            joint_pos_des = self.diff_ik_controller.compute(
                ee_pos_b, ee_quat_b, jacobian, joint_pos
            )

            # Apply
            self.apply_actions(joint_pos_des, gripper_open=gripper_open)
          
            # Optional recording
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
        joint_pos_des = self.robot.data.joint_pos[
            :, self.robot_entity_cfg.joint_ids
        ].clone()
        for _ in range(steps):
            self.apply_actions(joint_pos_des, gripper_open=gripper_open)
           
            obs = self.get_observation(gripper_open=gripper_open)
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
            env_ids_t = torch.as_tensor(env_ids, device=device, dtype=torch.long).view(
                -1
            )  # (M,)
        M = int(env_ids_t.shape[0])

        # --- object pose/vel: set object at env origins with identity quat ---
        env_origins = self.scene.env_origins.to(device)[env_ids_t]  # (M,3)
        object_pose = torch.zeros((M, 7), device=device, dtype=torch.float32)
        object_pose[:, :3] = env_origins
        object_pose[:, 3] = 1.0  # wxyz = [1,0,0,0]
        self.object_prim.write_root_pose_to_sim(object_pose, env_ids=env_ids_t)
        self.object_prim.write_root_velocity_to_sim(
            torch.zeros((M, 6), device=device, dtype=torch.float32), env_ids=env_ids_t
        )
        self.object_prim.write_data_to_sim()
        for prim in self.other_object_prims:
            prim.write_root_pose_to_sim(object_pose, env_ids=env_ids_t)
            prim.write_root_velocity_to_sim(
                torch.zeros((M, 6), device=device, dtype=torch.float32),
                env_ids=env_ids_t,
            )
            prim.write_data_to_sim()

        # --- robot base pose/vel ---
        # robot_pose is (B,7) in *local* base frame; add env origin offset per env
        rp_local = self.robot_pose.to(self.robot.device)[env_ids_t]  # (M,7)
        env_origins_robot = env_origins.to(self.robot.device)  # (M,3)
        robot_pose_world = rp_local.clone()
        robot_pose_world[:, :3] = env_origins_robot + robot_pose_world[:, :3]
        #print(f"[INFO] robot_pose_world: {robot_pose_world}")
        self.robot.write_root_pose_to_sim(robot_pose_world, env_ids=env_ids_t)
        self.robot.write_root_velocity_to_sim(
            torch.zeros((M, 6), device=self.robot.device, dtype=torch.float32),
            env_ids=env_ids_t,
        )

        # --- joints (B,7) -> select ids (M,7) ---
        joint_pos = self.robot.data.default_joint_pos.to(self.robot.device)[
            env_ids_t
        ]  # (M,7)
        joint_vel = self.robot.data.default_joint_vel.to(self.robot.device)[
            env_ids_t
        ]  # (M,7)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_t)
        self.robot.write_data_to_sim()

        # current_object_pose = self.object_prim.data.root_state_w[:, :7]
        # print("[INFO]Before step, current_object_pose", current_object_pose)
        # current_object_com = self.object_prim.data.root_com_pos_w[:, :3]
        # print("[INFO]Before step, current_object_com", current_object_com)
        self.clear_data()
        obs = self.get_observation(gripper_open=True)

        for _ in range(3):
            self.step()

        # current_object_pose = self.object_prim.data.root_state_w[:, :7]
        # print("[INFO]After step, current_object_pose", current_object_pose)
        # current_object_com = self.object_prim.data.root_com_pos_w[:, :3]
        # print("[INFO]After step, current_object_com", current_object_com)
        # housekeeping
        #print(f"[INFO] object prim pose: {self.object_prim.data.root_state_w[:, :7]}")
        #print(f"[INFO] object prim com: {self.object_prim.data.root_com_pos_w[:, :3]}")
        for prim in self.other_object_prims:
            prim.update(dt=self.sim.get_physics_dt())
            
            #print(f"[INFO] updated prim pose: {prim.data.root_state_w[:, :7]}")
            #print(f"[INFO] updated prim com: {prim.data.root_com_pos_w[:, :3]}")
        self.robot.update(dt=self.sim.get_physics_dt())
        self.object_prim.update(dt=self.sim.get_physics_dt())

        self.count = 0
        self.task_step_count = 0
        self.joint_pos_des_list = []
        self.gripper_cmd_list = []
        


    # ---------- Get Observations ----------
    def get_observation(self, gripper_open) -> Dict[str, torch.Tensor]:
        # camera outputs (already batched)
        rgb = self.camera.data.output["rgb"]  # [B,H,W,3]
        depth = self.camera.data.output["distance_to_image_plane"]  # [B,H,W]
        ins_all = self.camera.data.output["instance_id_segmentation_fast"]  # [B,H,W]

        B, H, W, _ = ins_all.shape
        fg_mask_list = []
        obj_mask_list = []
        robot_mask_list = []
        for b in range(B):
            ins_id_seg = ins_all[b]
            id_mapping = self.camera.data.info[b]["instance_id_segmentation_fast"][
                "idToLabels"
            ]
            fg_mask_b = torch.zeros_like(
                ins_id_seg, dtype=torch.bool, device=ins_id_seg.device
            )
            obj_mask_b = torch.zeros_like(
                ins_id_seg, dtype=torch.bool, device=ins_id_seg.device
            )
            robot_mask_b = torch.zeros_like(
                ins_id_seg, dtype=torch.bool, device=ins_id_seg.device
            )
            for key, value in id_mapping.items():
                if "object" in value:
                    fg_mask_b |= ins_id_seg == key
                    obj_mask_b |= ins_id_seg == key
                if "Robot" in value or "robot" in value.lower():
                    fg_mask_b |= ins_id_seg == key
                    robot_mask_b |= ins_id_seg == key
            fg_mask_list.append(fg_mask_b)
            obj_mask_list.append(obj_mask_b)
            robot_mask_list.append(robot_mask_b)
        fg_mask = torch.stack(fg_mask_list, dim=0)  # [B,H,W]
        obj_mask = torch.stack(obj_mask_list, dim=0)  # [B,H,W]
        robot_mask = torch.stack(robot_mask_list, dim=0)  # [B,H,W]
        ee_pose_w = self.robot.data.body_state_w[
            :, self.robot_entity_cfg.body_ids[0], 0:7
        ]
        scene_origin = self.scene.env_origins.to(self.robot.device)[0]
        ee_pose_l = ee_pose_w.clone()
        ee_pose_l[:, :3] -= scene_origin
        joint_pos = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
        joint_vel = self.robot.data.joint_vel[:, self.robot_entity_cfg.joint_ids]
        gripper_pos = self.robot.data.joint_pos[:, self.robot_gripper_cfg.joint_ids]
        gripper_cmd = (
            self.gripper_open_tensor if gripper_open else self.gripper_close_tensor
        )

        cam_pos_w = self.camera.data.pos_w
        cam_quat_w = self.camera.data.quat_w_ros
        ee_pos_cam, ee_quat_cam = subtract_frame_transforms(
            cam_pos_w, cam_quat_w, ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        ee_pose_cam = torch.cat([ee_pos_cam, ee_quat_cam], dim=1)

        points_3d_cam = unproject_depth(
            self.camera.data.output["distance_to_image_plane"],
            self.camera.data.intrinsic_matrices,
        )
        points_3d_world = transform_points(
            points_3d_cam, self.camera.data.pos_w, self.camera.data.quat_w_ros
        )

        object_center = self.object_prim.data.root_com_pos_w[:, :3]

        return {
            "rgb": rgb,
            "depth": depth,
            "fg_mask": fg_mask,
            "robot_mask": robot_mask,
            "joint_pos": joint_pos,
            "gripper_pos": gripper_pos,
            "gripper_cmd": gripper_cmd,
            "joint_vel": joint_vel,
            "ee_pose_cam": ee_pose_cam,
            "ee_pose_w": ee_pose_w,
            "ee_pose_l": ee_pose_l,
            "object_mask": obj_mask,
            "points_cam": points_3d_cam,
            "points_world": points_3d_world,
            "object_center": object_center,
        }

    # ---------- Task Completion Verifier ----------
    def is_success(self) -> bool:
        raise NotImplementedError(
            "BaseSimulator.is_success() should be implemented in subclass."
        )

    # ---------- Data Recording & Saving & Clearing ----------
    def record_data(self, obs: Dict[str, torch.Tensor]):
        # task_step_count increments at task step frequency (every decimation physics steps)
        if self.task_step_count % self.save_interval == 0:
            self.save_dict["rgb"].append(obs["rgb"].cpu().numpy())  # [B,H,W,3]
            self.save_dict["depth"].append(obs["depth"].cpu().numpy())  # [B,H,W]
            self.save_dict["segmask"].append(obs["fg_mask"].cpu().numpy())  # [B,H,W]
            self.save_dict["robot_mask"].append(obs["robot_mask"].cpu().numpy())  # [B,H,W]
            self.save_dict["object_mask"].append(obs["object_mask"].cpu().numpy())  # [B,H,W]
            self.save_dict["joint_pos"].append(obs["joint_pos"].cpu().numpy())  # [B,nJ]
            self.save_dict["gripper_pos"].append(obs["gripper_pos"].cpu().numpy())  # [B,3]
            self.save_dict["gripper_cmd"].append(obs["gripper_cmd"].cpu().numpy())  # [B,1]
            self.save_dict["joint_vel"].append(obs["joint_vel"].cpu().numpy())
            self.save_dict["ee_pose_cam"].append(obs["ee_pose_cam"].cpu().numpy())
            self.save_dict["ee_pose_l"].append(obs["ee_pose_l"].cpu().numpy())

    def clear_data(self):
        for key in self.save_dict.keys():
            self.save_dict[key] = []
 
    def _env_dir(self, base: Path, b: int) -> Path:
        d = base / f"env_{b:03d}"
        d.mkdir(parents=True, exist_ok=True)
        return d
    
    def _get_next_demo_dir(self, base: Path) -> Path:
        already_existing_num = len(list(base.iterdir()))
        return base / f"demo_{already_existing_num:03d}.mp4"

    def save_data(self, ignore_keys: List[str] = [], env_ids: Optional[List[int]] = None, export_hdf5: bool = False, save_other_things: bool = True, formal = True):
        self.save_dict["joint_pos_des"] = np.array(self.joint_pos_des_list) # Size: T, B, 7
        stacked = {k: np.array(v) for k, v in self.save_dict.items()}
        if env_ids is None:
            env_ids = self._all_env_ids.cpu().numpy()
        if formal:
            video_dir = self.out_dir / self.img_folder / "videos"
        else:
            video_dir = self.out_dir / self.img_folder / "videos_debug"
        video_dir.mkdir(parents=True, exist_ok=True)
        
        # Pre-allocate composed_rgb array (no deep copy needed)
        self.save_dict["composed_rgb"] = np.empty_like(stacked["rgb"])
        
        # Load background image once
        bg_rgb_path = self.task_cfg.background_cfg.background_rgb_path
        bg_rgb = imageio.imread(bg_rgb_path)
        
        hdf5_names = []
        video_paths = []
        fps = 1 / (self.sim_dt * self.decimation * self.save_interval)
        
        for b in env_ids:
            demo_path = self._get_next_demo_dir(video_dir)
            hdf5_names.append(demo_path.stem)
            if save_other_things:
                if formal:
                    demo_dir = self.out_dir / self.img_folder/ "demos"
                else:
                    demo_dir = self.out_dir / self.img_folder/ "demos_debug"
                demo_dir.mkdir(parents=True, exist_ok=True)
                
                demo_dir = self._get_next_demo_dir(demo_dir).with_suffix('')
                env_dir = self._env_dir(demo_dir, b)

                env_dir.mkdir(parents=True, exist_ok=True)

                for key, arr in stacked.items():
                    if key in ignore_keys:  # skip the keys for storage
                        continue
                    if key == "rgb":
                        video_path = env_dir / "sim_video.mp4"
                        # Use ffmpeg with faster codec
                        writer = imageio.get_writer(
                            video_path, fps=fps, codec='libx264', quality=8, macro_block_size=None
                        )
                        for frame in arr[:, b]:
                            writer.append_data(frame)
                        writer.close()                     
                    elif key == "segmask":
                        video_path = env_dir / "mask_video.mp4"
                        writer = imageio.get_writer(
                            video_path, fps=fps, codec='libx264', quality=8, macro_block_size=None
                        )
                        # Vectorize mask conversion
                        mask_frames = (arr[:, b].astype(np.uint8) * 255)
                        for frame in mask_frames:
                            writer.append_data(frame)
                        writer.close()
                    elif key == "depth":
                        depth_seq = arr[:, b]
                        flat = depth_seq[depth_seq > 0]
                        max_depth = np.percentile(flat, 99) if flat.size > 0 else 1.0
                        # Vectorized depth normalization
                        depth_norm = np.clip(depth_seq / max_depth * 255.0, 0, 255).astype(np.uint8)
                        video_path = env_dir / "depth_video.mp4"
                        writer = imageio.get_writer(
                            video_path, fps=fps, codec='libx264', quality=8, macro_block_size=None
                        )
                        for frame in depth_norm:
                            writer.append_data(frame)
                        writer.close()
                        np.save(env_dir / f"{key}.npy", depth_seq)
                    elif key != "composed_rgb" and len(arr.shape) >= 2:
                        np.save(env_dir / f"{key}.npy", arr[:, b])
            # Batch compose all frames for this environment
            rgb_env = stacked["rgb"][:, b]  # [T, H, W, 3]
            mask_env = stacked["segmask"][:, b]  # [T, H, W]
            
            # Compose frames in batch
            composed_frames = []
            for t in range(rgb_env.shape[0]):
                composed = self.convert_real(mask_env[t], bg_rgb, rgb_env[t])
                self.save_dict["composed_rgb"][t, b] = composed
                composed_frames.append(composed)
            
            # Write video with faster codec
            writer = imageio.get_writer(demo_path, fps=fps, codec='libx264', quality=8, macro_block_size=None)
            for frame in composed_frames:
                writer.append_data(frame)
            writer.close()
            video_paths.append(str(demo_path))
            print(f"[INFO]: Demonstration is saved at: {demo_path}")
        
        
        if export_hdf5:
            self.export_batch_data_to_hdf5(hdf5_names, video_paths)
           
       
    def get_current_frame_count(self) -> int:
        return len(self.save_dict["rgb"])

    def export_batch_data_to_hdf5(self, hdf5_names: List[str], video_paths: List[str]) -> int:
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
      
        camera_params = self._get_camera_parameters()

        for env_idx, (episode_name, video_path) in enumerate(zip(episode_names, video_paths)):
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

                # Use frames from memory instead of re-reading video (much faster!)
                rgb_frames = stacked["composed_rgb"][:, env_idx]  # (T, H, W, 3)
                
                # Encode frames using JPEG compression with parallel processing
                encode_data = []
                max_len = 0
                
                # Batch encode for better performance
                for i in range(len(rgb_frames)):
                    # Use quality=85 for good balance of size/quality
                    success, encoded_image = cv2.imencode(".jpg", rgb_frames[i], [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if not success:
                        raise RuntimeError(f"Failed to encode frame {i}")
                    jpeg_data = encoded_image.tobytes()
                    encode_data.append(jpeg_data)
                    max_len = max(max_len, len(jpeg_data))
                
                # Pad all encoded frames to the same length
                padded_data = [jpeg_data.ljust(max_len, b"\0") for jpeg_data in encode_data]
                
                # Store encoded data with compression
                cam_grp.create_dataset(
                    "rgb", 
                    data=np.array(padded_data, dtype=f"S{max_len}"),
                    compression="lzf"
                )
                cam_grp.attrs["encoding"] = "jpeg"
                cam_grp.attrs["channels"] = 3
                cam_grp.attrs["original_shape"] = rgb_frames.shape

                # Save robot mask
                if "robot_mask" in stacked:
                    robot_mask_frames = stacked["robot_mask"][:, env_idx]  # (T, H, W)
                    cam_grp.create_dataset(
                        "robot_mask",
                        data=robot_mask_frames.astype(np.uint8),
                        compression="lzf"
                    )

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
                if "joint_pos_des" in stacked and len(stacked["joint_pos_des"]) > 0:
                    joint_grp.create_dataset(
                        "joint_pos_des", data=stacked["joint_pos_des"][:, env_idx].astype(np.float32)
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
                    action_grp = f.create_group("action")
                    action_grp.create_dataset(
                        "actions", data=stacked["actions"][:, env_idx].astype(np.float32)
                    )
                    action_grp.create_dataset(
                        "action_indices", data=stacked["action_indices"][:, env_idx].astype(np.int32)
                    )
                    
                ee_grp = f.create_group("ee_pose")
                if "ee_pose_cam" in stacked:
                    ee_grp.create_dataset(
                        "ee_pose_cam", data=stacked["ee_pose_cam"][:, env_idx].astype(np.float32)
                    )
                if "ee_pose_l" in stacked:
                    ee_grp.create_dataset(
                        "ee_pose_l", data=stacked["ee_pose_l"][:, env_idx].astype(np.float32)
                    )
                extras_grp = f.create_group("extras")
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
                meta_grp.attrs["frame_dt"] = float(self.sim_dt * self.decimation * self.save_interval)  # Use task_dt since data is recorded at task step frequency
                meta_grp.attrs["frame_count"] = int(frame_count)
                meta_grp.attrs["source"] = "OpenReal2Sim"
                meta_grp.attrs["episode_name"] = episode_name
                meta_grp.create_dataset("frame_indices", data=np.arange(frame_count, dtype=np.int32))

        print(f"[INFO]: Exported {num_envs} HDF5 episodes to {data_dir}")
        return num_envs



    def add_text_to_image(self, image, env_idx=0, time_step=0):
        """
        Add text overlay to image showing count, joint positions, and gripper command.
        
        Args:
            image: numpy array [H, W, 3] in uint8 format (0-255)
            env_idx: environment index to use for joint_pos_des_list (default: 0)
        
        Returns:
            image with text overlay
        """
        image = image.copy()
        
        # Draw count
        cv2.putText(image, f"Count: {time_step}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0), 2)
        
        # Draw joint positions
        if hasattr(self, 'joint_pos_des_list') and len(self.joint_pos_des_list) > 0:
            last_joint_pos = self.joint_pos_des_list[time_step]  # [B, 7] or [7]
            
            # Convert to numpy if torch tensor
            if isinstance(last_joint_pos, torch.Tensor):
                if last_joint_pos.ndim == 2:
                    # Batch format: take env_idx-th row
                    joint_vals = last_joint_pos[env_idx].cpu().numpy()
                else:
                    # Single row
                    joint_vals = last_joint_pos.cpu().numpy()
            else:
                # Already numpy
                if isinstance(last_joint_pos, np.ndarray) and last_joint_pos.ndim == 2:
                    joint_vals = last_joint_pos[env_idx]
                else:
                    joint_vals = np.asarray(last_joint_pos)
            
            # Format joint positions (split into two lines)
            joint_pos_str = "Joint: "
            for j in range(min(7, len(joint_vals))):
                if j == 4:
                    joint_pos_str += "\n"
                joint_pos_str += f"{joint_vals[j]:.3f}, "
            joint_pos_str = joint_pos_str.rstrip(", ")  # Remove trailing comma
            
            # Draw joint positions (cv2.putText doesn't support \n, so draw on two lines)
            cv2.putText(image, joint_pos_str.split("\n")[0], (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            if "\n" in joint_pos_str:
                cv2.putText(image, joint_pos_str.split("\n")[1], (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 0), 2)
                gripper_y = 120
            else:
                gripper_y = 90
        else:
            gripper_y = 60
        
        # Draw gripper command
        if hasattr(self, 'gripper_cmd_list') and len(self.gripper_cmd_list) > 0:
            gripper_cmd = self.gripper_cmd_list[time_step]
            
            # Convert to scalar value
            if isinstance(gripper_cmd, torch.Tensor):
                if gripper_cmd.ndim == 0:
                    gripper_val = gripper_cmd.item()
                elif gripper_cmd.ndim == 1:
                    gripper_val = gripper_cmd[env_idx].item() if len(gripper_cmd) > env_idx else gripper_cmd[0].item()
                else:
                    gripper_val = gripper_cmd[env_idx, 0].item() if gripper_cmd.shape[0] > env_idx else gripper_cmd[0, 0].item()
            elif isinstance(gripper_cmd, (list, np.ndarray)):
                gripper_cmd_np = np.asarray(gripper_cmd)
                if gripper_cmd_np.ndim == 0:
                    gripper_val = float(gripper_cmd_np)
                elif gripper_cmd_np.ndim == 1:
                    gripper_val = float(gripper_cmd_np[env_idx]) if len(gripper_cmd_np) > env_idx else float(gripper_cmd_np[0])
                else:
                    gripper_val = float(gripper_cmd_np[env_idx, 0]) if gripper_cmd_np.shape[0] > env_idx else float(gripper_cmd_np[0, 0])
            else:
                gripper_val = float(gripper_cmd) if not isinstance(gripper_cmd, bool) else (1.0 if gripper_cmd else 0.0)
            
            gripper_cmd_str = f"Gripper: {gripper_val:.3f}"
            cv2.putText(image, gripper_cmd_str, (10, gripper_y), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 255, 0), 2)
        
        return image
        
    def convert_real(self,segmask, bg_rgb, fg_rgb):
        segmask_2d = segmask[..., 0]
        composed = bg_rgb.copy()
        composed[segmask_2d] = fg_rgb[segmask_2d]
        return composed

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

