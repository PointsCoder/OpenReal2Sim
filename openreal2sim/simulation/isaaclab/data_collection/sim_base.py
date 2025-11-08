from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional


import curobo
import imageio

# Isaac Lab
import isaaclab.sim as sim_utils
import numpy as np
import torch
import time
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

os.environ['CUROBO_TORCH_CUDA_GRAPH_RESET'] = '1'


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
    ) -> None:
        # basic simulation setup
        self.sim: sim_utils.SimulationContext = sim
        self.sim_cfgs = sim_cfgs
        self.scene = scene
        self.sim_dt = sim.get_physics_dt()

        self.num_envs: int = int(scene.num_envs)
        self._all_env_ids = torch.arange(
            self.num_envs, device=sim.device, dtype=torch.long
        )

        self.cam_dict = cam_dict
        self.out_dir: Path = out_dir
        self.img_folder: str = img_folder

        # scene entities
        self.robot = scene["robot"]
        if robot_pose.ndim == 1:
            self.robot_pose = (
                robot_pose.view(1, -1).repeat(self.num_envs, 1).to(self.robot.device)
            )
        else:
            assert robot_pose.shape[0] == self.num_envs and robot_pose.shape[1] == 7, (
                f"robot_pose must be [B,7], got {robot_pose.shape}"
            )
            self.robot_pose = robot_pose.to(self.robot.device).contiguous()

        self.object_prim = scene["object_00"]
        self.other_object_prims = [
            scene[key]
            for key in scene.keys()
            if f"object_" in key and key != "object_00"
        ]
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
        self.count = 0
        self.demo_id = 0
        self.save_dict = {
            "rgb": [],
            "depth": [],
            "segmask": [],
            "joint_pos": [],
            "joint_vel": [],
            "actions": [],
            "gripper_pos": [],
            "gripper_cmd": [],
        }

        # visualization
        self.selected_object_id = 0
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

        # curobo motion planning
        self.enable_motion_planning = enable_motion_planning
        if self.enable_motion_planning:
            print(f"prepare curobo motion planning: {enable_motion_planning}")
            self.prepare_curobo()
            print("curobo motion planning ready.")

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
                
                result = self.motion_gen.plan_batch(start_state, goal_pose, plan_cfg)
                
                # Process results
                paths = result.get_paths()
                T_max = 1
                
                for i, p in enumerate(paths):
                    if not result.success[i]:
                        print(f"[WARN] Motion planning failed for env {i}.")
                    else:
                        T_max = max(T_max, int(p.position.shape[-2]))
                
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
                print(f"[ERROR] Unexpected error in motion planning: {type(e).__name__}: {e}")
                
                if retry < max_retries - 1 and ("cuda graph" in str(e).lower() or "NoneType" in str(e)):
                    if self.reinitialize_motion_gen():
                        print(f"[INFO] Retrying after CUDA error (attempt {retry+2}/{max_retries})...")
                        continue
                break
        
        # If we get here, all retries failed - return None to signal complete failure
        print("[ERROR] Motion planning failed critically - returning None to trigger restart")
        return None, None


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
                traj = result.get_interpolated_plan()
                
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
                print(f"[ERROR] Unexpected error: {type(e).__name__}: {e}")
                
                if retry < max_retries - 1 and ("cuda graph" in str(e).lower() or "NoneType" in str(e)):
                    if self.reinitialize_motion_gen():
                        continue
                break
        
        # Complete failure - return None to signal restart
        print("[ERROR] Motion planning failed critically - returning None to trigger restart")
        return None, None

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
        record: bool = True,
    ) -> torch.Tensor:
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

    def compose_real_video(self, env_id: int = 0):
        """
        Composite simulated video onto real background using mask-based rendering.
        
        Args:
            env_id: Environment ID to process
        """
    
        def pad_to_even(frame):
            """Pad frame to even dimensions for video encoding."""
            H, W = frame.shape[:2]
            pad_h = H % 2
            pad_w = W % 2
            if pad_h or pad_w:
                pad = ((0, pad_h), (0, pad_w)) if frame.ndim == 2 else ((0, pad_h), (0, pad_w), (0, 0))
                frame = np.pad(frame, pad, mode='edge')
            return frame
        
        # Construct paths
        base_path = self.out_dir / self.img_folder
        demo_path = self._demo_dir() / f"env_{env_id:03d}"
        
        SIM_VIDEO_PATH = demo_path / "sim_video.mp4"
        MASK_VIDEO_PATH = demo_path / "mask_video.mp4"
        REAL_BACKGROUND_PATH = base_path / "reconstruction" / "background.jpg"
        OUTPUT_PATH = demo_path / "real_video.mp4"
        
        # Check if required files exist
        if not SIM_VIDEO_PATH.exists():
            print(f"[WARN] Simulated video not found: {SIM_VIDEO_PATH}")
            return False
        if not MASK_VIDEO_PATH.exists():
            print(f"[WARN] Mask video not found: {MASK_VIDEO_PATH}")
            return False
        if not REAL_BACKGROUND_PATH.exists():
            print(f"[WARN] Real background not found: {REAL_BACKGROUND_PATH}")
            return False
        
        # Load real background image
        print(f"[INFO] Loading real background: {REAL_BACKGROUND_PATH}")
        real_img = np.array(Image.open(REAL_BACKGROUND_PATH))
        
        H, W, _ = real_img.shape
        print(f"[INFO] Background size: {W}x{H}")
        
        # Open videos
        print(f"[INFO] Loading simulated video: {SIM_VIDEO_PATH}")
        rgb_reader = imageio.get_reader(SIM_VIDEO_PATH)
        print(f"[INFO] Loading mask video: {MASK_VIDEO_PATH}")
        mask_reader = imageio.get_reader(MASK_VIDEO_PATH)
        
        # Get metadata from original video
        N = rgb_reader.count_frames()
        fps = rgb_reader.get_meta_data()['fps']
        print(f"[INFO] Processing {N} frames at {fps} FPS...")
        
        composed_images = []
        
        for i in range(N):
            # Read frames
            sim_rgb = rgb_reader.get_data(i)
            sim_mask = mask_reader.get_data(i)
            
            # Convert mask to binary (grayscale > 127 = foreground)
            if sim_mask.ndim == 3:
                sim_mask = cv2.cvtColor(sim_mask, cv2.COLOR_RGB2GRAY)
            sim_mask = sim_mask > 127
            
            # Resize sim frames to match real background if needed
            if sim_rgb.shape[:2] != (H, W):
                sim_rgb = cv2.resize(sim_rgb, (W, H))
                sim_mask = cv2.resize(sim_mask.astype(np.uint8), (W, H)) > 0
            
            # Compose: real background + simulated foreground (where mask is True)
            composed = real_img.copy()
            composed = pad_to_even(composed)
            composed[sim_mask] = sim_rgb[sim_mask]
            
            composed_images.append(composed)
            
            if (i + 1) % 10 == 0:
                print(f"[INFO] Processed {i + 1}/{N} frames")
        
        # Save composed video
        print(f"[INFO] Saving composed video to: {OUTPUT_PATH}")
        writer = imageio.get_writer(OUTPUT_PATH, fps=fps, macro_block_size=None)
        for frame in composed_images:
            writer.append_data(frame)
        writer.close()
        
        print(f"[INFO] Done! Saved {len(composed_images)} frames to {OUTPUT_PATH} at {fps} FPS")
        return True
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
        self.camera.update(dt=self.sim_dt)
        self.count += 1
        self.scene.update(self.sim_dt)

    # ---------- Apply actions to robot joints ----------
    def apply_actions(self, joint_pos_des, gripper_open: bool = True):
        # joint_pos_des: [B, n_joints]
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
                position, quaternion, gripper_open=gripper_open, record=record
            )
        else:
            return self.move_to_ik(
                position, quaternion, gripper_open=gripper_open, record=record
            )

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
        joint_pos_des = self.robot.data.joint_pos[
            :, self.robot_entity_cfg.joint_ids
        ].clone()
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

        # housekeeping
        self.clear_data()

    # ---------- Get Observations ----------
    def get_observation(self, gripper_open) -> Dict[str, torch.Tensor]:
        # camera outputs (already batched)
        rgb = self.camera.data.output["rgb"]  # [B,H,W,3]
        depth = self.camera.data.output["distance_to_image_plane"]  # [B,H,W]
        ins_all = self.camera.data.output["instance_id_segmentation_fast"]  # [B,H,W]

        B, H, W, _ = ins_all.shape
        fg_mask_list = []
        obj_mask_list = []
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
            for key, value in id_mapping.items():
                if "object" in value:
                    fg_mask_b |= ins_id_seg == key
                    obj_mask_b |= ins_id_seg == key
                if "Robot" in value:
                    fg_mask_b |= ins_id_seg == key
            fg_mask_list.append(fg_mask_b)
            obj_mask_list.append(obj_mask_b)
        fg_mask = torch.stack(fg_mask_list, dim=0)  # [B,H,W]
        obj_mask = torch.stack(obj_mask_list, dim=0)  # [B,H,W]

        ee_pose_w = self.robot.data.body_state_w[
            :, self.robot_entity_cfg.body_ids[0], 0:7
        ]
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
        raise NotImplementedError(
            "BaseSimulator.is_success() should be implemented in subclass."
        )

    # ---------- Data Recording & Saving & Clearing ----------
    def record_data(self, obs: Dict[str, torch.Tensor]):
        self.save_dict["rgb"].append(obs["rgb"].cpu().numpy())  # [B,H,W,3]
        self.save_dict["depth"].append(obs["depth"].cpu().numpy())  # [B,H,W]
        self.save_dict["segmask"].append(obs["fg_mask"].cpu().numpy())  # [B,H,W]
        self.save_dict["joint_pos"].append(obs["joint_pos"].cpu().numpy())  # [B,nJ]
        self.save_dict["gripper_pos"].append(obs["gripper_pos"].cpu().numpy())  # [B,3]
        self.save_dict["gripper_cmd"].append(obs["gripper_cmd"].cpu().numpy())  # [B,1]
        self.save_dict["joint_vel"].append(obs["joint_vel"].cpu().numpy())

    def clear_data(self):
        for key in self.save_dict.keys():
            self.save_dict[key] = []

    def _demo_dir(self) -> Path:
        return self.out_dir / self.img_folder / "demos" / f"demo_{self.demo_id}"

    def _env_dir(self, base: Path, b: int) -> Path:
        d = base / f"env_{b:03d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_data(self, ignore_keys: List[str] = []):
        save_root = self._demo_dir()
        save_root.mkdir(parents=True, exist_ok=True)

        stacked = {k: np.array(v) for k, v in self.save_dict.items()}

        for b in range(self.num_envs):
            env_dir = self._env_dir(save_root, b)
            for key, arr in stacked.items():
                if key in ignore_keys:  # skip the keys for storage
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
        
        # Compose real videos for all environments
        print("\n[INFO] Composing real videos with background...")
        for b in range(self.num_envs):
            print(f"[INFO] Processing environment {b}/{self.num_envs}...")
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
        }
        with open(demo_save_path / "meta_info.json", "w") as f:
            json.dump(meta_info, f)
        os.system(f"cp -r {save_root}/* {demo_save_path}")
        print("[INFO]: Demonstration is saved at: ", demo_save_path)

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
