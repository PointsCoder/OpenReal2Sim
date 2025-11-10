from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, List

from loguru import logger
import tyro

# Add parent directory to path
TOOLS_DIR = Path(__file__).resolve().parent
MUJOCO_DIR = TOOLS_DIR.parent
if str(MUJOCO_DIR) not in sys.path:
    sys.path.insert(0, str(MUJOCO_DIR))

from utils.scene_fusion import (
    SceneFusion,
    CollisionDefaults,
    ContactParameters,
    load_object_metadata,
    load_scene_config,
)
from utils.robot_placement import compute_robot_pose_from_scene


MUJOCO_ROOT = MUJOCO_DIR
DEFAULT_ROBOT_PATH = MUJOCO_ROOT / "assets/robots/franka_emika_panda"
DEFAULT_CONSTANTS_PATH = MUJOCO_ROOT / "config" / "constants.yaml"


def main(
    scene_name: Optional[str] = None,
    outputs_root: Path = Path("outputs"),
    demo_path: Optional[Path] = None,
    asset_path: Optional[Path] = None,
    robot_path: Path = DEFAULT_ROBOT_PATH,
    output: Optional[Path] = None,
    constants_path: Path = DEFAULT_CONSTANTS_PATH,
    object_mass: List[str] = [],
    default_mass: float = 0.1,
    groundplane_height: float = 0.0,
    robot_pose: Optional[List[float]] = None,
) -> int:
    """Fuse robot MJCF with reconstructed assets.

    Args:
        scene_name: Scene name in outputs/<scene_name>/simulation/
        outputs_root: Root directory containing outputs
        demo_path: Path to demo folder
        asset_path: Directory that stores MJCF asset packages
        robot_path: Directory that contains panda.xml and scene.xml
        output: Path for fused XML
        constants_path: Path to constants YAML file
        object_mass: Object masses as key=value pairs
        default_mass: Default mass for objects not specified
        groundplane_height: Height of the ground plane
        robot_pose: Manual robot pose [x, y, z, qw, qx, qy, qz]
    """
    # Scene mode: work with outputs/<scene_name>/simulation/
    if scene_name:
        outputs_root = outputs_root.expanduser().resolve()
        scene_dir = outputs_root / scene_name / "simulation"
        demo_path = scene_dir
        mjcf_dir = scene_dir / "mujoco" / "mjcf"
        asset_path = mjcf_dir
        if output is None:
            output = scene_dir / "mujoco" / "scene.xml"
    else:
        demo_path = demo_path.expanduser().resolve()
        asset_path = (asset_path or demo_path).expanduser().resolve()

    robot_path = robot_path.expanduser().resolve()
    robot_scene_xml = robot_path / "scene.xml"
    robot_panda_xml = robot_path / "panda.xml"

    scene_config = load_scene_config(demo_path)
    object_metadata = load_object_metadata(asset_path, scene_config)

    # Load constants from YAML
    import yaml
    constants_path = constants_path.expanduser().resolve()
    with open(constants_path, "r") as f:
        constants = yaml.safe_load(f)
    
    # Extract parameters from constants
    z_offset = constants["defaults"]["z_offset"]
    inertia_scale = constants["defaults"]["inertia_scale"]
    freejoint_damping = float(constants["joint"]["obj_freejoint_damping"])
    
    timestep = constants["simulation"]["timestep"]
    memory = constants["simulation"]["memory"]
    solver_iterations = constants["simulation"]["solver_iterations"]
    solver_ls_iterations = constants["simulation"]["solver_ls_iterations"]
    solver_noslip_iterations = constants["simulation"]["solver_noslip_iterations"]
    
    material_specular = constants["material"]["specular"]
    material_shininess = constants["material"]["shininess"]
    
    robot_asset_prefix = "../../../../openreal2sim/simulation/mujoco/assets/robots/franka_emika_panda"
    
    # Load collision defaults
    collision_defaults = CollisionDefaults(
        margin=constants["object_collision"]["margin"],
        solref=constants["object_collision"]["solref"],
        solimp=constants["object_collision"]["solimp"],
    )
    
    # Load contact parameters
    contact_params = ContactParameters(
        condim=constants["contact_pair"]["condim"],
        solref=constants["contact_pair"]["solref"],
        solimp=constants["contact_pair"]["solimp"],
        friction=constants["contact_pair"]["friction"],
    )

    # Parse object masses from CLI (key=value format)
    mass_dict = {}
    if object_mass:
        for item in object_mass:
            if "=" not in item:
                logger.error(f"Invalid mass format: '{item}'. Expected key=value format (e.g., spoon=0.05)")
                return 1
            key, value = item.split("=", 1)
            try:
                mass_dict[key] = float(value)
            except ValueError:
                logger.error(f"Invalid mass value for '{key}': '{value}' is not a number")
                return 1

    # Build masses dict for all objects
    masses = {}
    for obj_cfg in scene_config["objects"].values():
        obj_name = obj_cfg["name"]
        if obj_name in mass_dict:
            masses[obj_name] = mass_dict[obj_name]
            logger.info(f"Using mass from CLI for '{obj_name}': {mass_dict[obj_name]} kg")
        else:
            masses[obj_name] = default_mass
            logger.warning(f"No mass specified for '{obj_name}', using default: {default_mass} kg")

    # Compute or load robot pose
    if robot_pose:
        # Manual pose provided
        logger.info(f"Using manual robot pose: {robot_pose}")
    elif "robot_cfg" in scene_config and "robot_pose" in scene_config["robot_cfg"]:
        # Pose already in scene.json
        robot_pose = scene_config["robot_cfg"]["robot_pose"]
        logger.info(f"Using robot pose from scene.json: {robot_pose}")
    else:
        # Compute from heuristics
        robot_pose = compute_robot_pose_from_scene(scene_config, None)
        logger.info(f"Computed robot pose: {robot_pose}")

    # Store robot pose in scene_config for fusion
    if "robot_cfg" not in scene_config:
        scene_config["robot_cfg"] = {}
    scene_config["robot_cfg"]["robot_pose"] = robot_pose

    fusion = SceneFusion(
        asset_root=asset_path,
        masses=masses,
        z_offset=z_offset,
        inertia_scale=inertia_scale,
        freejoint_damping=freejoint_damping,
        robot_asset_prefix=robot_asset_prefix,
        collision_defaults=collision_defaults,
        contact_params=contact_params,
        groundplane_height=groundplane_height,
        timestep=timestep,
        memory=memory,
        solver_iterations=solver_iterations,
        solver_ls_iterations=solver_ls_iterations,
        solver_noslip_iterations=solver_noslip_iterations,
        material_specular=material_specular,
        material_shininess=material_shininess,
    )

    tree = fusion.fuse(
        robot_scene_path=robot_scene_xml,
        robot_panda_path=robot_panda_xml,
        scene_config=scene_config,
        object_metadata=object_metadata,
    )

    output_path = (output or (demo_path / "scene.xml")).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    logger.info(f"Fused scene written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(tyro.cli(main))
