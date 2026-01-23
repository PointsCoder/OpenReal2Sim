# -*- coding: utf-8 -*-
"""
Isaac Lab-based simulation environment factory.
"""

from __future__ import annotations
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Tuple, Optional, List, Dict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg

# Isaac Lab core
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.schemas import schemas_cfg
import isaaclab.sim as sim_utils

# Manager-based API (terms/configs)
from isaaclab.managers import (
    TerminationTermCfg as DoneTerm,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    CurriculumTermCfg as CurrTerm,
)

# Task-specific MDP helpers (adjust path if needed)
import isaaclab_tasks.manager_based.manipulation.lift.mdp as mdp


@dataclass
class SceneCtx:
    """Context for scene configuration."""
    obj_paths: List[str]
    background_path: str
    bg_physics: Optional[Dict] = None
    obj_physics: Optional[List[Dict]] = None
    use_ground_plane: bool = False
    ground_z: Optional[float] = None


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


def _deep_update(dst: dict, src: Optional[dict]) -> dict:
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
# Dynamic InteractiveSceneCfg builder
# --------------------------------------------------------------------------------------
def build_tabletop_scene_cfg():
    """
    Auto-generate a multi-object InteractiveSceneCfg subclass:
      - background
      - object_00, object_01, ... based on _SCENE_CTX.obj_paths
    """
    assert _SCENE_CTX is not None, (
        "init_scene_from_scene_dict must be called first."
    )
    C = _SCENE_CTX

    base_attrs = {}

    # Light
    base_attrs["light"] = AssetBaseCfg(
        prim_path="/World/lightDome",
        spawn=sim_utils.DomeLightCfg(intensity=4000.0, color=(1.0, 1.0, 1.0)),
    )

    _bg = _deep_update(DEFAULT_BG_PHYSICS, C.bg_physics)
    if C.obj_physics is None:
        raise ValueError("obj_physics must be initialized before building scene config.")
    _objs = [
        _deep_update(DEFAULT_OBJ_PHYSICS, obj_physics) for obj_physics in C.obj_physics
    ]

    bg_mass_cfg = schemas_cfg.MassPropertiesCfg(**_bg["mass_props"])
    bg_rigid_cfg = schemas_cfg.RigidBodyPropertiesCfg(**_bg["rigid_props"])
    bg_colli_cfg = schemas_cfg.CollisionPropertiesCfg(**_bg["collision_props"])

    # Add another ground plane (mainly for better visualization)
    base_attrs["background_n"] = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    z = float(C.ground_z if C.ground_z is not None else 0.0)
    base_attrs["background_n"].init_state.pos = (0.0, 0.0, z - 0.2)

    # ---------- Background ----------
    if C.use_ground_plane:
        base_attrs["background"] = AssetBaseCfg(
            prim_path="/World/defaultGroundPlane",
            spawn=sim_utils.GroundPlaneCfg(),
        )
        z = float(C.ground_z if C.ground_z is not None else 0.0)
        base_attrs["background"].init_state.pos = (0.0, 0.0, z)
    else:
        base_attrs["background"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Background",
            spawn=sim_utils.UsdFileCfg(
                usd_path="",
                mass_props=bg_mass_cfg,
                rigid_props=bg_rigid_cfg,
                collision_props=bg_colli_cfg,
            ),
        )

    # Instantiate objects
    for i, usd_path in enumerate(C.obj_paths):
        obj_physics_i = _objs[i]
        obj_mass_cfg = schemas_cfg.MassPropertiesCfg(**obj_physics_i["mass_props"])
        obj_rigid_cfg = schemas_cfg.RigidBodyPropertiesCfg(**obj_physics_i["rigid_props"])
        obj_colli_cfg = schemas_cfg.CollisionPropertiesCfg(
            **obj_physics_i["collision_props"]
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

    # Inject background path on finalize
    def __post_init__(self):
        """Post-initialization hook to set background USD path."""
        if not C.use_ground_plane:
            self.background.spawn.usd_path = C.background_path

    attrs = dict(base_attrs)
    attrs["__doc__"] = "Auto-generated multi-object TableTop scene cfg."
    attrs["__post_init__"] = __post_init__

    DynamicSceneCfg = configclass(
        type("TableTopSceneCfgAuto", (InteractiveSceneCfg,), attrs)
    )
    return DynamicSceneCfg


# --------------------------------------------------------------------------------------
# Build & init directly from a raw scene dict
# --------------------------------------------------------------------------------------
def init_scene_from_scene_dict(
    scene: dict,
    cfgs: dict,
    *,
    use_ground_plane: bool = False,
):
    """
    Initialize SceneCtx directly from a raw scene dict.
    """
    obj_paths = [o["usd"] for o in scene["objects"].values()]
    background_path = scene["background"]["usd"]

    # Overwrite physics
    # Priority: args > scene > default
    obj_physics = cfgs["physics_cfg"]["obj_physics"]
    bg_physics = cfgs["physics_cfg"]["bg_physics"]
    if obj_physics is None:
        obj_physics = [o.get("physics", None) for o in scene["objects"].values()]
    elif isinstance(obj_physics, dict):
        obj_physics = [obj_physics for _ in scene["objects"].values()]
    elif isinstance(obj_physics, list):
        assert len(obj_physics) == len(scene["objects"]), (
            "obj_physics must be a list of the same length as scene['objects'] if provided."
        )
    else:
        raise TypeError("obj_physics must be None, a dict, or a list of dicts.")
    bg_physics = (
        scene["background"].get("physics", None) if bg_physics is None else bg_physics
    )

    ground_z = None
    if use_ground_plane:
        try:
            ground_z = float(scene["plane"]["simulation"]["point"][2])
        except (KeyError, TypeError, IndexError, ValueError) as e:
            raise ValueError(
                f"use_ground_plane=True but scene['plane']['simulation'] missing/invalid: {e}"
            ) from e

    # Write global ctx
    global _SCENE_CTX
    _SCENE_CTX = SceneCtx(
        obj_paths=obj_paths,
        background_path=background_path,
        bg_physics=bg_physics,
        obj_physics=list(obj_physics),
        use_ground_plane=use_ground_plane,
        ground_z=ground_z,
    )

    return {
        "obj_usd_paths": obj_paths,
        "background_usd": background_path,
        "use_ground_plane": use_ground_plane,
        "ground_z": ground_z,
    }


# --------------------------------------------------------------------------------------
# Env factory
# --------------------------------------------------------------------------------------
def _build_manip_env_cfg(scene_cfg_cls, *, num_envs: int, env_spacing: float = 2.5):
    """Return a ManagerBasedRLEnvCfg subclass for object manipulation scenes."""
    from isaaclab.envs import ManagerBasedRLEnvCfg

    @configclass
    class ManipEnvCfg(ManagerBasedRLEnvCfg):
        scene = scene_cfg_cls(num_envs=num_envs, env_spacing=env_spacing)
        actions = ActionsCfg()
        observations = ObservationsCfg()
        events = EventCfg()
        rewards = RewardsCfg()
        terminations = TerminationsCfg()
        curriculum = CurriculumCfg()

        def __post_init__(self):
            # ---- Sim & PhysX ----
            self.decimation = 2
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

    return ManipEnvCfg


def load_scene_json(key: str) -> dict:
    """Return the raw scene dict from outputs/<key>/simulation/scene.json."""
    scene_path = Path.cwd() / "outputs" / key / "simulation" / "scene.json"
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene file not found: {scene_path}")
    with open(scene_path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_env(
    cfgs: dict,
    num_envs: int = 1,
    device: str = "cuda:0",
    bg_simplify: bool = False,
) -> Tuple["ManagerBasedRLEnv", "ManagerBasedRLEnvCfg"]:
    """
    Public entry to construct a ManagerBasedRLEnv from outputs/<key>/simulation/scene.json.
    
    Args:
        cfgs: Configuration dictionary containing 'scene_cfg' and 'physics_cfg'.
        num_envs: Number of parallel environments to create.
        device: Device to run simulation on (e.g., "cuda:0").
        bg_simplify: Whether to use a ground plane instead of background mesh.
    
    Returns:
        Tuple of (env, env_cfg).
    """
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg

    # Load scene json and initialize global SceneCtx
    scene = cfgs["scene_cfg"]
    init_scene_from_scene_dict(
        scene,
        cfgs=cfgs,
        use_ground_plane=bg_simplify,
    )

    # Build scene & env cfg
    SceneCfg = build_tabletop_scene_cfg()
    ManipEnvCfg = _build_manip_env_cfg(SceneCfg, num_envs=num_envs, env_spacing=2.5)
    env_cfg = ManipEnvCfg()
    env_cfg.sim.device = device
    env_cfg.scene.num_envs = num_envs  # Double safety

    env = ManagerBasedRLEnv(cfg=env_cfg)
    return env, env_cfg


# --------------------------------------------------------------------------------------
# Action/Observation/Reward/Termination/Curriculum config classes
# --------------------------------------------------------------------------------------
@configclass
class ActionsCfg:
    """Action specifications for the MDP (empty for calibration/testing)."""
    pass


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
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
    """Reward terms for the MDP (minimal)."""
    pass


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    pass


# Public symbols
__all__ = [
    # Scene init
    "init_scene_from_scene_dict",
    # Scene builders
    "build_tabletop_scene_cfg",
    # Manager config groups
    "ActionsCfg",
    "ObservationsCfg",
    "EventCfg",
    "RewardsCfg",
    "TerminationsCfg",
    "CurriculumCfg",
    # Env factory
    "make_env",
]
