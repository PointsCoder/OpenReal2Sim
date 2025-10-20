# -*- coding: utf-8 -*-
"""
OpenReal2Sim ManiSkill Environment.

Loads reconstructed scenes from the OpenReal2Sim pipeline and creates
an interactive robotic manipulation environment.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from .utils.scene_loader import SceneConfig, load_scene_config, resolve_path
from .utils.transform_utils import (
    convert_camera_pose,
    intrinsic_to_fov,
    opencv_to_sapien_pose,
    pose2qt,
    qvec2rotmat,
    rotmat2qvec,
    rotz_np,
)


@register_env("OpenReal2Sim-v0", max_episode_steps=200)
class OpenReal2SimEnv(BaseEnv):
    """
    OpenReal2Sim environment for ManiSkill.

    This environment loads a reconstructed scene from the OpenReal2Sim pipeline
    and provides a robotic manipulation environment with:
    - Reconstructed background geometry
    - Reconstructed object meshes
    - Camera configuration matching the original scene
    - Franka Panda robot for manipulation

    Args:
        scene_json_path: Path to scene.json file from OpenReal2Sim reconstruction
        robot_uids: Robot model to use (default: "panda")
        robot_init_qpos_noise: Noise level for initial robot configuration
        **kwargs: Additional arguments passed to BaseEnv
    """

    ROBOT_BASE_POS = [-0.6, 0.0, 0.3]
    ROBOT_BASE_QUAT = [1, 0, 0, 0]
    ROBOT_BASE_POSE = sapien.Pose(p=ROBOT_BASE_POS, q=ROBOT_BASE_QUAT)
    OBJECT_INIT_QUAT = [0, 0, 0, 1]
    OBJECT_INIT_POS = [0, 0, 0.35]
    BACKGROUND_INIT_POS = [0, 0, 0.2]
    BACKGROUND_INIT_QUAT = [0, 0, 0, 1]
    ROBOT_INIT_QPOS = [
        0.0,
        np.pi / 8,
        0,
        -np.pi * 5 / 8,
        0,
        np.pi * 3 / 4,
        np.pi / 4,
    ]
    ROBOT_INIT_QPOS_NOISE = 0.02
    ROBOT_INIT_QPOS_NOISE_2 = 0.04
    SUPPORTED_ROBOTS = ["panda"]
    CAMERA_WORLD_YAW_RAD = 0

    def __init__(
        self,
        *args,
        scene_json_path: str = None,
        robot_uids: str = "panda",
        robot_init_qpos_noise: float = 0.02,
        **kwargs,
    ):
        if scene_json_path is None:
            raise ValueError("scene_json_path must be provided")

        self.scene_json_path = Path(scene_json_path)
        self.scene_config: SceneConfig = load_scene_config(self.scene_json_path)
        self.robot_init_qpos_noise = robot_init_qpos_noise

        # Scene objects (populated in _load_scene)
        self.ground = None
        self.background_actor = None
        self.object_actors: Dict[str, sapien.Entity] = {}

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: dict):
        """Load the robot agent with proper initial pose."""
        super()._load_agent(options, sapien.Pose(p=self.ROBOT_BASE_POS))
        self.agent.robot.set_pose(self.ROBOT_BASE_POSE)  # set the robot base pose

    def _load_scene(self, options: dict):
        """Load all scene assets."""
        # Build ground plane
        ground_z = self.scene_config.ground_plane_point[2]
        self.ground = build_ground(
            self.scene,
            floor_width=100,
            altitude=ground_z - 0.01,
        )

        # Load background mesh
        self._load_background()

        # Load object meshes
        self._load_objects()

        # Setup lighting
        self._setup_lighting()

    def _load_background(self):
        """Load the reconstructed background mesh."""
        bg_path = resolve_path(self.scene_config.background_mesh_path)

        if not bg_path.exists():
            raise FileNotFoundError(f"Background mesh not found: {bg_path}")

        builder = self.scene.create_actor_builder()
        builder.add_visual_from_file(str(bg_path))
        builder.add_multiple_convex_collisions_from_file(
            str(bg_path), decomposition="coacd"
        )

        # rotate along y axis by 180 degrees
        builder.set_initial_pose(
            sapien.Pose(p=self.BACKGROUND_INIT_POS, q=self.BACKGROUND_INIT_QUAT)
        )
        self.background_actor = builder.build_static(name="background")

    def _load_objects(self):
        """Load reconstructed object meshes as dynamic actors."""
        for idx, (obj_id, obj_config) in enumerate(self.scene_config.objects.items()):
            obj_path = resolve_path(obj_config.mesh_path)

            if not obj_path.exists():
                print(
                    f"[WARN] Skipping {obj_config.name}: mesh not found at {obj_path}"
                )
                continue

            builder = self.scene.create_actor_builder()
            builder.add_visual_from_file(str(obj_path))
            builder.add_multiple_convex_collisions_from_file(
                str(obj_path), decomposition="none"
            )

            builder.set_initial_pose(
                sapien.Pose(p=self.OBJECT_INIT_POS, q=self.OBJECT_INIT_QUAT)
            )

            actor = builder.build(name=f"object_{obj_config.name}")
            self.object_actors[obj_id] = actor

    def _setup_lighting(self):
        """Setup scene lighting."""
        self.scene.add_directional_light(
            direction=[0, 0, -1], color=[1, 1, 1], shadow=True
        )
        self.scene.add_point_light(position=[2, 2, 2], color=[1, 1, 1], shadow=False)
        self.scene.add_point_light(
            position=[-2, -2, 2], color=[0.8, 0.8, 0.8], shadow=False
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Initialize episode poses and robot configuration."""
        with torch.device(self.device):
            b = len(env_idx)

            # Reset background to origin
            if self.background_actor is not None:
                self.background_actor.set_pose(
                    Pose.create_from_pq(
                        p=self.BACKGROUND_INIT_POS, q=self.BACKGROUND_INIT_QUAT
                    )
                )

            # Set object poses from reconstruction data

            for obj_id, actor in self.object_actors.items():
                pos = torch.tensor(
                    self.OBJECT_INIT_POS, dtype=torch.float32, device=self.device
                )
                quat = torch.tensor(
                    self.OBJECT_INIT_QUAT, dtype=torch.float32, device=self.device
                )
                actor.set_pose(Pose.create_from_pq(p=pos, q=quat))

            # Initialize robot
            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                ]
            )
            if self.robot_init_qpos_noise > 0:
                qpos = (
                    self._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
                qpos[:, -2:] = 0.04
            self.agent.reset(qpos)
            self.agent.robot.set_pose(self.ROBOT_BASE_POSE)

    def evaluate(self) -> dict:
        """Evaluate success conditions (placeholder)."""
        # TODO: Implement task-specific success conditions
        return {
            "success": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        }

    def _get_obs_extra(self, info: Dict) -> Dict:
        """Get additional task-specific observations."""
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)

        if "state" in self.obs_mode:
            for obj_id, actor in self.object_actors.items():
                obj_name = self.scene_config.objects[obj_id].name
                obs[f"obj_{obj_name}_pose"] = actor.pose.raw_pose

        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """Compute dense reward (placeholder)."""
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        """Compute normalized dense reward."""
        return self.compute_dense_reward(obs, action, info)

    @property
    def _default_sensor_configs(self) -> List[CameraConfig]:
        """
        Configure cameras to match the reconstructed scene.

        Note: Unlike typical ManiSkill tasks, we use world-frame camera positioning
        (similar to Isaac Lab) to match the original reconstruction camera pose.
        """
        cam_config = self.scene_config.camera

        # Convert camera from OpenCV world frame to SAPIEN world frame
        # get the background object z position
        height = np.array(self.BACKGROUND_INIT_POS)[2]
        extrinsic_matrix = np.array(cam_config.extrinsic_matrix, dtype=np.float32)
        extrinsic_matrix[2, 3] += height
        # y = -y:
        # extrinsic_matrix[1, 3] = -extrinsic_matrix[1, 3]
        camera_pose_world = opencv_to_sapien_pose(extrinsic_matrix)

        # # Mirror the camera pose across the world's XZ plane
        # # Convert pose tensor to numpy
        # q_numpy = camera_pose_world.q.cpu().numpy().squeeze()
        # p_numpy = camera_pose_world.p.cpu().numpy().squeeze()

        # # 1. Mirror the position across the XZ plane by negating the Y coordinate
        # new_camera_p = p_numpy * np.array([1, -1, 1])

        # # 2. Mirror the orientation
        # camera_rot_mat = qvec2rotmat(q_numpy)

        # # Get original look-at (X) and up (Z) vectors for SAPIEN camera in world frame
        # look_at_orig = camera_rot_mat[:, 0]
        # up_orig = camera_rot_mat[:, 2]

        # # Reflect these vectors across the world's XZ plane
        # look_at_new = look_at_orig * np.array([1, -1, 1])
        # up_new = up_orig * np.array([1, -1, 1])

        # # Re-create a valid right-handed coordinate system (rotation matrix)
        # x_axis = look_at_new / np.linalg.norm(look_at_new)
        # y_axis = np.cross(up_new, x_axis)
        # y_axis = y_axis / np.linalg.norm(y_axis)
        # z_axis = np.cross(x_axis, y_axis)
        # new_camera_rot_mat = np.stack([x_axis, y_axis, z_axis], axis=1)

        # new_camera_q = rotmat2qvec(new_camera_rot_mat)

        # reflected_camera_pose = sapien.Pose(p=new_camera_p, q=new_camera_q)

        # Compute FOV from intrinsics
        fov = intrinsic_to_fov(
            cam_config.fx, cam_config.width, cam_config.fy, cam_config.height
        )

        return [
            CameraConfig(
                uid="base_camera",
                pose=camera_pose_world,  # reflected_camera_pose,  # World-frame pose (matches Isaac Lab)
                width=cam_config.width,
                height=cam_config.height,
                fov=fov,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self) -> CameraConfig:
        """Configure camera for human viewing/recording."""
        pose = sapien_utils.look_at(eye=[0.8, 0.8, 0.6], target=[0, 0, 0.2])
        return CameraConfig(
            uid="render_camera",
            pose=pose,
            width=512,
            height=512,
            fov=np.pi / 3,
            near=0.01,
            far=100,
        )
