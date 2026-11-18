# -*- coding: utf-8 -*-
"""
Isaac Lab-based simulation environment factory.
"""

from __future__ import annotations
import copy
import json
import random
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import numpy as np
import transforms3d

# Isaac Lab core
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.schemas import schemas_cfg
import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import CameraCfg
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

# Manager-based API (terms/configs)
from isaaclab.managers import (
    TerminationTermCfg as DoneTerm,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    CurriculumTermCfg as CurrTerm,
    SceneEntityCfg,
)

# Task-specific MDP helpers (adjust path if needed)
import isaaclab_tasks.manager_based.manipulation.lift.mdp as mdp

from dataclasses import dataclass, MISSING

from dataclasses import dataclass
from typing import Optional, List, Dict


@dataclass
class SceneCtx:
    cam_dict: Dict
    obj_paths: List[str]
    background_path: str
    robot_pos: List[float]
    robot_rot: List[float]
    bg_physics: Optional[Dict] = None
    obj_physics: List[Dict] = None
    use_ground_plane: bool = False
    ground_z: Optional[float] = None
    oid_to_index: Optional[Dict[str, int]] = None  # Maps oid (str) to prim index
    index_to_oid: Optional[Dict[int, str]] = None  # Maps prim index to oid (str)
    robot_cfg_list: List[Dict] = None   

_SCENE_CTX: Optional[SceneCtx] = None

# ---- default physx presets ----
DEFAULT_BG_PHYSICS = {
    "mass_props": {"mass": 100.0},
    "rigid_props": {"disable_gravity": True, "kinematic_enabled": True},
    "collision_props": {
        "collision_enabled": True,
        "contact_offset": 0.0015,
        "rest_offset": 0.0003,
        "torsional_patch_radius": 0.02,
        "min_torsional_patch_radius": 0.005,
    },
}
DEFAULT_OBJ_PHYSICS = {
    "mass_props": {"mass": 0.5},
    "rigid_props": {"disable_gravity": False, "kinematic_enabled": False},
    "collision_props": {
        "collision_enabled": True,
        "contact_offset": 0.0015,
        "rest_offset": 0.0003,
        "torsional_patch_radius": 0.02,
        "min_torsional_patch_radius": 0.005,
    },
}

# ---- extra visualization cameras (overview + wrist) ----
EXTRA_CAM_WIDTH = 640
EXTRA_CAM_HEIGHT = 480
# shorter than the 24 mm default, i.e. a wider field of view
EXTRA_CAM_FOCAL_LENGTH = 14.0
# overview placement relative to the real camera, see overview_camera_pose()
OVERVIEW_PULL_BACK = 1.6
OVERVIEW_LIFT = 1.5
OVERVIEW_MIN_HEIGHT = 0.6
# gripper link the wrist camera is parented to
HAND_LINK_NAME = "panda_hand"

DEFAULT_NO_COLLISION_BG_PHYSICS = {
    "mass_props": {"mass": 100.0},
    "rigid_props": {"disable_gravity": True, "kinematic_enabled": True},
    "collision_props": {
        "collision_enabled": False,
    },
}

DEFAULT_NO_COLLISION_OBJ_PHYSICS = {
    "mass_props": {"mass": 0.5},
    "rigid_props": {"disable_gravity": True, "kinematic_enabled": False},
    "collision_props": {
        "collision_enabled": False,
    },
}

def _deep_update(dst: dict, src: dict | None) -> dict:
    """Recursive dict update without touching the original."""
    out = copy.deepcopy(dst)
    if not src:
        return out
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


# --------------------------------------------------------------------------------------
# Camera / Robot builders (use _SCENE_CTX)
# --------------------------------------------------------------------------------------
def create_camera():
    """Return a CameraCfg using the global SceneCtx."""
    assert _SCENE_CTX is not None, (
        "init_scene_configs/init_scene_from_scene_dict must be called first."
    )
    C = _SCENE_CTX.cam_dict
    width = int(C["width"])
    height = int(C["height"])
    fx, fy, cx, cy = C["fx"], C["fy"], C["cx"], C["cy"]
    cam_orientation = tuple(C["cam_orientation"])
    cam_pos = tuple(C["scene_info"]["move_to"])
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        offset=CameraCfg.OffsetCfg(pos=cam_pos, rot=cam_orientation, convention="ros"),
        data_types=[
            "rgb",
            "distance_to_image_plane",
            "distance_to_camera",
            "instance_id_segmentation_fast",
        ],
        colorize_instance_id_segmentation=False,
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=[fx, 0, cx, 0, fy, cy, 0, 0, 1],
            width=width,
            height=height,
        ),
        width=width,
        height=height,
    )


def _look_at_quat_ros(cam_pos, target) -> Tuple[float, float, float, float]:
    """Return the wxyz quaternion that aims a ros-convention camera at target.

    In the ros convention the camera looks along +Z, with +X right and +Y down.
    """
    cam_pos = np.asarray(cam_pos, dtype=float)
    target = np.asarray(target, dtype=float)
    forward = target - cam_pos
    norm = float(np.linalg.norm(forward))
    if norm < 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    forward = forward / norm
    world_up = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(forward, world_up))) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0], dtype=float)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right) + 1e-12
    down = np.cross(forward, right)
    down /= np.linalg.norm(down) + 1e-12
    rot = np.column_stack([right, down, forward])
    return tuple(float(q) for q in transforms3d.quaternions.mat2quat(rot))


def overview_camera_pose():
    """Derive a third person pose from the real camera pose and the scene bounds.

    The background mesh is reconstructed from the real camera only, so it has
    holes when seen from the other side. The overview camera therefore keeps the
    real camera azimuth (same hemisphere) and only moves back and up, aiming at
    the manipulated object center.
    """
    C = _SCENE_CTX.cam_dict
    info = C["scene_info"]
    target = np.asarray(info["object_center"], dtype=float)
    cam_pos = np.asarray(info["move_to"], dtype=float)
    scene_min = np.asarray(info["scene_min"], dtype=float)
    scene_max = np.asarray(info["scene_max"], dtype=float)
    scene_diag = float(np.linalg.norm(scene_max - scene_min))

    delta = cam_pos - target
    horizontal = delta[:2]
    horizontal_dist = float(np.linalg.norm(horizontal))
    if horizontal_dist < 1e-6:
        horizontal = np.array([1.0, 0.0], dtype=float)
        horizontal_dist = 1.0
    direction = horizontal / horizontal_dist

    back_dist = max(OVERVIEW_PULL_BACK * horizontal_dist, horizontal_dist + 0.5 * scene_diag)
    height = target[2] + max(
        OVERVIEW_LIFT * float(delta[2]), 0.5 * scene_diag, OVERVIEW_MIN_HEIGHT
    )
    pos = np.array(
        [
            target[0] + direction[0] * back_dist,
            target[1] + direction[1] * back_dist,
            height,
        ],
        dtype=float,
    )
    return tuple(float(v) for v in pos), _look_at_quat_ros(pos, target)


def create_overview_camera():
    """Return a fixed third person CameraCfg derived from the real camera pose."""
    assert _SCENE_CTX is not None, (
        "init_scene_configs/init_scene_from_scene_dict must be called first."
    )
    pos, rot = overview_camera_pose()
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/OverviewCamera",
        offset=CameraCfg.OffsetCfg(pos=pos, rot=rot, convention="ros"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=EXTRA_CAM_FOCAL_LENGTH,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 1.0e5),
        ),
        width=EXTRA_CAM_WIDTH,
        height=EXTRA_CAM_HEIGHT,
    )


def create_wrist_camera():
    """Return a CameraCfg parented to the gripper, looking along the approach axis."""
    assert _SCENE_CTX is not None, (
        "init_scene_configs/init_scene_from_scene_dict must be called first."
    )
    return CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{HAND_LINK_NAME}/wrist_camera",
        # the hand frame +Z is the approach axis; sit beside and behind the
        # fingertips so the fingers stay in view
        offset=CameraCfg.OffsetCfg(
            pos=(0.05, 0.0, 0.0), rot=(0.707, 0.0, 0.0, 0.707), convention="ros"
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=EXTRA_CAM_FOCAL_LENGTH,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 1.0e5),
        ),
        width=EXTRA_CAM_WIDTH,
        height=EXTRA_CAM_HEIGHT,
    )


def create_robot():
    """Return a configured Franka Panda config using the global SceneCtx."""
    assert _SCENE_CTX is not None, (
        "init_scene_configs/init_scene_from_scene_dict must be called first."
    )
    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = tuple(_SCENE_CTX.robot_pos)
    robot.init_state.rot = tuple(_SCENE_CTX.robot_rot)
    return robot


# --------------------------------------------------------------------------------------
# Dynamic InteractiveSceneCfg builder
# --------------------------------------------------------------------------------------
def build_tabletop_scene_cfg(has_collision: bool = True):
    """
    Auto-generate a multi-object InteractiveSceneCfg subclass:
      - background, camera, robot
      - object_00, object_01, ... based on _SCENE_CTX.obj_paths
    """
    assert _SCENE_CTX is not None, (
        "init_scene_configs/init_scene_from_scene_dict must be called first."
    )
    C = _SCENE_CTX

    base_attrs = {}

    # Light
    base_attrs["light"] = AssetBaseCfg(
        prim_path="/World/lightDome",
        spawn=sim_utils.DomeLightCfg(intensity=4000.0, color=(1.0, 1.0, 1.0)),
    )

    if has_collision:
        BG_PHYSICS = DEFAULT_BG_PHYSICS
        OBJ_PHYSICS = DEFAULT_OBJ_PHYSICS
    else:
        BG_PHYSICS = DEFAULT_NO_COLLISION_BG_PHYSICS
        OBJ_PHYSICS = DEFAULT_NO_COLLISION_OBJ_PHYSICS

    _bg = _deep_update(BG_PHYSICS, C.bg_physics)
    _objs = [
        _deep_update(OBJ_PHYSICS, obj_physics) for obj_physics in C.obj_physics
    ]

    _bg["collision_props"]["collision_enabled"] = BG_PHYSICS["collision_props"]["collision_enabled"]
    for _obj in _objs:
        _obj["collision_props"]["collision_enabled"] = OBJ_PHYSICS["collision_props"]["collision_enabled"]
    
    # Force disable_gravity to match default (prevent override from external configs)
    _bg["rigid_props"]["disable_gravity"] = BG_PHYSICS["rigid_props"]["disable_gravity"]
    for _obj in _objs:
        _obj["rigid_props"]["disable_gravity"] = OBJ_PHYSICS["rigid_props"]["disable_gravity"]

    bg_mass_cfg = schemas_cfg.MassPropertiesCfg(**_bg["mass_props"])
    bg_rigid_cfg = schemas_cfg.RigidBodyPropertiesCfg(**_bg["rigid_props"])
    bg_colli_cfg = schemas_cfg.CollisionPropertiesCfg(**_bg["collision_props"])

    # add another ground plane (mainly for better visualization)
    base_attrs["backgroundn"] = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    z = float(C.ground_z if C.ground_z is not None else 0.0)
    base_attrs["backgroundn"].init_state.pos = (0.0, 0.0, z - 0.2)

    # ---------- Background ----------
    if C.use_ground_plane:
        # Simple horizontal ground; only z is customized.
        # Note: GroundPlaneCfg doesn't take mass/rigid/collision configs (it's a helper),
        #       so we only set pose here.
        base_attrs["background"] = AssetBaseCfg(
            prim_path="/World/defaultGroundPlane",
            spawn=sim_utils.GroundPlaneCfg(),
        )
        z = float(C.ground_z if C.ground_z is not None else 0.0)
        base_attrs["background"].init_state.pos = (0.0, 0.0, z)
        # no usd_path assignment in __post_init__ when using ground
    else:
        # original USD background
        base_attrs["background"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Background",
            spawn=sim_utils.UsdFileCfg(
                usd_path="",
                mass_props=bg_mass_cfg,
                rigid_props=bg_rigid_cfg,
                collision_props=bg_colli_cfg,
            ),
        )

    # Placeholder entries to be replaced in __post_init__.
    # Order matters: InteractiveScene._add_entities_from_cfg iterates the cfg
    # fields in declaration order, so the robot articulation must be declared
    # before any sensor parented under one of its links (the wrist camera),
    # otherwise the camera spawn cannot find its parent prim.
    base_attrs["robot"] = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )
    base_attrs["camera"] = CameraCfg(prim_path="{ENV_REGEX_NS}/Camera")
    base_attrs["overview_camera"] = CameraCfg(prim_path="{ENV_REGEX_NS}/OverviewCamera")
    base_attrs["wrist_camera"] = CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{HAND_LINK_NAME}/wrist_camera"
    )

    # Instantiate objects
    for i, usd_path in enumerate(C.obj_paths):
        obj_mass_cfg = schemas_cfg.MassPropertiesCfg(**_objs[i]["mass_props"])
        obj_rigid_cfg = schemas_cfg.RigidBodyPropertiesCfg(**_objs[i]["rigid_props"])
        obj_colli_cfg = schemas_cfg.CollisionPropertiesCfg(
            **_objs[i]["collision_props"]
        )

        obj_template = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=sim_utils.UsdFileCfg(
                usd_path="",
                mass_props=obj_mass_cfg,
                rigid_props=obj_rigid_cfg,
                collision_props=obj_colli_cfg,
            ),
        )

        name = f"object_{i:02d}"
        cfg_i = copy.deepcopy(obj_template)
        cfg_i.prim_path = f"{{ENV_REGEX_NS}}/{name}"
        cfg_i.spawn.usd_path = usd_path
        base_attrs[name] = cfg_i

    # Inject background path + replace camera/robot on finalize
    def __post_init__(self):
        if not C.use_ground_plane:
            self.background.spawn.usd_path = C.background_path
        # keep the robot first, see the ordering note above
        self.robot = create_robot()
        self.camera = create_camera()
        self.overview_camera = create_overview_camera()
        self.wrist_camera = create_wrist_camera()

    attrs = dict(base_attrs)
    attrs["__doc__"] = "Auto-generated multi-object TableTop scene cfg."
    attrs["__post_init__"] = __post_init__

    DynamicSceneCfg = configclass(
        type("TableTopSceneCfgAuto", (InteractiveSceneCfg,), attrs)
    )
    return DynamicSceneCfg


# --------------------------------------------------------------------------------------
# Build cam_dict & init directly from a raw scene dict
# --------------------------------------------------------------------------------------
def init_scene_from_scene_dict(
    scene: dict,
    cfgs: dict,
    *,
    use_ground_plane: bool = False,
):
    """
    Initialize SceneCtx directly from a raw scene dict.
    If robot pose not provided, sample one with robot_placement_candidates_v2().
    """
    cam_dict = cfgs["cam_cfg"]
    
    # Sort objects by oid to ensure consistent ordering
    # scene["objects"] keys are oid strings (e.g., "1", "2", ...)
    sorted_objects = sorted(scene["objects"].items(), key=lambda x: int(x[0]))
    
    # Build oid <-> index mapping
    oid_to_index = {}
    index_to_oid = {}
    obj_paths = []
    obj_physics_list = []
    
    for index, (oid_str, obj_data) in enumerate(sorted_objects):
        oid_to_index[oid_str] = index
        index_to_oid[index] = oid_str
        obj_paths.append(obj_data["usd"])
        obj_physics_list.append(obj_data.get("physics", None))
    print("[INFO] oid_to_index", oid_to_index)
    background_path = scene["background"]["usd"]

    # overwrite physics
    # Priority: args > scene > default
    obj_physics = cfgs["physics_cfg"]["obj_physics"]
    bg_physics = cfgs["physics_cfg"]["bg_physics"]
    if obj_physics is None:
        obj_physics = obj_physics_list  # Use physics from sorted objects
    elif isinstance(obj_physics, dict):
        obj_physics = [obj_physics for _ in range(len(obj_paths))]
    elif isinstance(obj_physics, list):
        assert len(obj_physics) == len(scene["objects"]), (
            "obj_physics must be a list of the same length as scene['objects'] if provided."
        )
        pass
    else:
        raise TypeError("obj_physics must be None, a dict, or a list of dicts.")
    bg_physics = (
        scene["background"].get("physics", None) if bg_physics is None else bg_physics
    )

    robot_pos = cfgs["robot_cfg"]["robot_pose"][:3]
    robot_rot = cfgs["robot_cfg"]["robot_pose"][3:]
    robot_cfg_list = cfgs["robot_cfg"]["robot_cfg_list"]
    ground_z = None
    if use_ground_plane:
        try:
            ground_z = float(scene["plane"]["simulation"]["point"][2])
        except Exception as e:
            raise ValueError(
                f"use_ground_plane=True but scene['plane']['simulation'] missing/invalid: {e}"
            )

    # write global ctx (keep old fields the same)
    global _SCENE_CTX
    _SCENE_CTX = SceneCtx(
        cam_dict=cam_dict,
        obj_paths=obj_paths,
        background_path=background_path,
        robot_pos=list(robot_pos),
        robot_rot=list(robot_rot),
        bg_physics=bg_physics,
        obj_physics=list(obj_physics),
        use_ground_plane=use_ground_plane,
        ground_z=ground_z,
        oid_to_index=oid_to_index,
        index_to_oid=index_to_oid,
    )

    return {
        "cam_dict": cam_dict,
        "obj_usd_paths": obj_paths,
        "background_usd": background_path,
        "robot_pos": robot_pos,
        "robot_rot": robot_rot,
        "use_ground_plane": use_ground_plane,
        "ground_z": ground_z,
        "robot_cfg_list": robot_cfg_list,
    }


# --------------------------------------------------------------------------------------
# Env factory
# --------------------------------------------------------------------------------------
def _build_manip_env_cfg(scene_cfg_cls, *, num_envs: int, env_spacing: float = 2.5):
    """Return a ManagerBasedRLEnvCfg subclass stitched together from sub-Cfgs."""
    from isaaclab.envs import ManagerBasedRLEnvCfg
    from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
    from isaaclab.envs.mdp.actions.actions_cfg import (
        DifferentialInverseKinematicsActionCfg,
    )

    @configclass
    class ManipEnvCfg(ManagerBasedRLEnvCfg):
        scene = scene_cfg_cls(num_envs=num_envs, env_spacing=env_spacing)
        commands = CommandsCfg()
        actions = ActionsCfg()
        observations = ObservationsCfg()
        events = EventCfg()
        rewards = RewardsCfg()
        terminations = TerminationsCfg()
        curriculum = CurriculumCfg()

        def __post_init__(self):
            # ---- Sim & PhysX ----
            self.decimation = 2  # 4
            self.episode_length_s = 5.0
            self.sim.dt = 0.01
            self.sim.render_interval = self.decimation

            physx = self.sim.physx
            physx.enable_ccd = True
            physx.solver_type = 1  # TGS
            physx.num_position_iterations = 16
            physx.num_velocity_iterations = 2
            physx.contact_offset = 0.003
            physx.rest_offset = 0.0
            physx.max_depenetration_velocity = 0.5
            physx.enable_stabilization = True
            physx.enable_sleeping = True

            # ---- IK arm & binary gripper ----
            self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
                asset_name="robot",
                joint_names=["panda_joint.*"],
                body_name="panda_hand",
                controller=DifferentialIKControllerCfg(
                    command_type="pose", use_relative_mode=True, ik_method="dls"
                ),
                scale=0.5,
                body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                    pos=[0.0, 0.0, 0.107]
                ),
            )
            self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
                asset_name="robot",
                joint_names=["panda_finger.*"],
                open_command_expr={"panda_finger_.*": 0.04},
                close_command_expr={"panda_finger_.*": 0.0},
            )
            self.commands.object_pose.body_name = "panda_hand"

    return ManipEnvCfg


def load_scene_json(key: str) -> dict:
    """Return the raw scene dict from outputs/<key>/simulation/scene.json."""
    scene_path = Path.cwd() / "outputs" / key / "simulation" / "scene.json"
    if not scene_path.exists():
        raise FileNotFoundError(scene_path)
    return json.load(open(scene_path))


def get_prim_name_from_oid(oid: str | int) -> str:
    """
    Get the prim name (e.g., "object_00") from object id.
    
    Args:
        oid: Object ID as string (e.g., "1") or integer (e.g., 1)
        
    Returns:
        Prim name string (e.g., "object_00")
    """
    assert _SCENE_CTX is not None, "init_scene_from_scene_dict must be called first."
    oid_str = str(oid)
    if _SCENE_CTX.oid_to_index is None:
        raise ValueError("oid_to_index mapping not available. Scene may not have been initialized from scene dict.")
    index = int(_SCENE_CTX.oid_to_index.get(oid_str))
    if index is None:
        raise ValueError(f"OID '{oid_str}' not found in scene. Available OIDs: {list(_SCENE_CTX.oid_to_index.keys())}")
    return f"object_{index:02d}"


def get_oid_from_prim_name(prim_name: str) -> str:
    """
    Get the object id from prim name (e.g., "object_00" -> "1").
    
    Args:
        prim_name: Prim name string (e.g., "object_00")
        
    Returns:
        OID as string (e.g., "1")
    """
    assert _SCENE_CTX is not None, "init_scene_from_scene_dict must be called first."
    if not prim_name.startswith("object_"):
        raise ValueError(f"Invalid prim name format: {prim_name}")
    try:
        index = int(prim_name.split("_")[1])
    except (IndexError, ValueError):
        raise ValueError(f"Invalid prim name format: {prim_name}")
    if _SCENE_CTX.index_to_oid is None:
        raise ValueError("index_to_oid mapping not available. Scene may not have been initialized from scene dict.")
    oid = _SCENE_CTX.index_to_oid.get(index)
    if oid is None:
        raise ValueError(f"Index {index} not found in scene. Available indices: {list(_SCENE_CTX.index_to_oid.keys())}")
    return oid


def make_env(
    cfgs: dict,
    num_envs: int = 1,
    device: str = "cuda:0",
    bg_simplify: bool = False,
    physics_freq: int = 100,
    decimation: int = 1,
    has_collision: bool = True,
) -> Tuple["ManagerBasedRLEnv", "ManagerBasedRLEnvCfg"]:
    """
    Public entry to construct a ManagerBasedRLEnv from outputs/<key>/simulation/scene.json.
    Returns: (env, env_cfg)
    """
    from isaaclab.envs import ManagerBasedRLEnv

    # Load scene json and initialize global SceneCtx
    scene = cfgs["scene_cfg"]
    init_scene_from_scene_dict(
        scene,
        cfgs=cfgs,
        use_ground_plane=bg_simplify)

    # Build scene & env cfg
    SceneCfg = build_tabletop_scene_cfg(has_collision=has_collision)
    ManipEnvCfg = _build_manip_env_cfg(SceneCfg, num_envs=num_envs, env_spacing=2.5)
    env_cfg = ManipEnvCfg()
    env_cfg.sim.device = device
    env_cfg.scene.num_envs = num_envs  # double safety
    env_cfg.sim.dt = 1 / physics_freq
    env_cfg.sim.render_interval = decimation
    env_cfg.decimation = decimation
    env = ManagerBasedRLEnv(cfg=env_cfg)
    return env, env_cfg


# --------------------------------------------------------------------------------------
# Observation/Action/Reward/Termination/Curriculum config classes
# --------------------------------------------------------------------------------------
@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6),
            pos_y=(-0.25, 0.25),
            pos_z=(0.25, 0.5),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: (
        mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg
    ) = MISSING  # type: ignore
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING  # type: ignore


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        target_object_position = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "object_pose"}
        )
        actions = ObsTerm(func=mdp.last_action)
        rgb = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg(name="camera"),
                "data_type": "rgb",
                "convert_perspective_to_orthogonal": False,
                "normalize": False,
            },
        )
        depth = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg(name="camera"),
                "data_type": "distance_to_image_plane",
                "convert_perspective_to_orthogonal": False,
                "normalize": False,
            },
        )
        segmask = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg(name="camera"),
                "data_type": "instance_id_segmentation_fast",
                "convert_perspective_to_orthogonal": False,
                "normalize": False,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class RewardsCfg:
    """Reward terms for the MDP (kept simple)."""

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000},
    )


# Public symbols
__all__ = [
    # Scene init
    "init_scene_configs",
    "init_scene_from_scene_dict",
    # Scene builders
    "create_camera",
    "create_overview_camera",
    "create_wrist_camera",
    "create_robot",
    "build_tabletop_scene_cfg",
    # Placement samplers
    "robot_placement_candidates",
    "robot_placement_candidates_v2",
    # Manager config groups
    "CommandsCfg",
    "ActionsCfg",
    "ObservationsCfg",
    "EventCfg",
    "RewardsCfg",
    "TerminationsCfg",
    "CurriculumCfg",
    # Env factory
    "make_env",
]
