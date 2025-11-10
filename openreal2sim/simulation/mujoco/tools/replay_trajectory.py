from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import mujoco
import mujoco.viewer
import numpy as np
import yaml
from loguru import logger
import tyro

# Add parent directory to path
TOOLS_DIR = Path(__file__).resolve().parent
MUJOCO_DIR = TOOLS_DIR.parent
REPO_ROOT = MUJOCO_DIR.parent.parent.parent  # OpenReal2Sim root

if str(MUJOCO_DIR) not in sys.path:
    sys.path.insert(0, str(MUJOCO_DIR))

from utils.scene_fusion import (
    SceneFusion,
    CollisionDefaults,
    ContactParameters,
    load_object_metadata,
)

DEFAULT_ROBOT_PATH = MUJOCO_DIR / "assets/robots/franka_emika_panda"
DEFAULT_CONSTANTS_PATH = MUJOCO_DIR / "config" / "constants.yaml"
DEFAULT_ROBOT_CONFIG_PATH = MUJOCO_DIR / "config" / "franka_panda_config.yaml"


def map_docker_path_to_local(path_str: str, repo_root: Path) -> Path:
    """Map docker paths (/app/...) to local filesystem."""
    if path_str.startswith("/app/"):
        rel_path = path_str[len("/app/"):]
        return repo_root / rel_path
    else:
        return Path(path_str)


def load_demo_config(demo_path: Path) -> dict:
    """Load config.json from demo directory."""
    config_path = demo_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {demo_path}")

    with open(config_path, "r") as f:
        return json.load(f)


def load_trajectory_from_npy(demo_path: Path) -> list[dict]:
    """Load trajectory from .npy files."""
    joint_pos_path = demo_path / "joint_pos.npy"
    joint_vel_path = demo_path / "joint_vel.npy"
    gripper_cmd_path = demo_path / "gripper_cmd.npy"

    if not joint_pos_path.exists():
        raise FileNotFoundError(f"joint_pos.npy not found in {demo_path}")
    if not joint_vel_path.exists():
        raise FileNotFoundError(f"joint_vel.npy not found in {demo_path}")
    if not gripper_cmd_path.exists():
        raise FileNotFoundError(f"gripper_cmd.npy not found in {demo_path}")

    joint_pos = np.load(joint_pos_path)
    joint_vel = np.load(joint_vel_path)
    gripper_cmd = np.load(gripper_cmd_path)

    # Convert gripper command to binary (same logic as convert_trajectory_format.py)
    gripper_binary = (gripper_cmd.sum(axis=-1) > 0.04).astype(int)

    trajectory = []
    for t in range(len(joint_pos)):
        trajectory.append({
            "qpos": joint_pos[t].tolist(),
            "qvel": joint_vel[t].tolist(),
            "gripper": int(gripper_binary[t])
        })

    logger.info(f"Loaded trajectory with {len(trajectory)} keyframes")
    return trajectory


def fuse_scene_from_config(
    demo_config: dict,
    repo_root: Path,
    robot_path: Path,
    constants_path: Path,
    output_xml: Path,
    mass_dict: dict[str, float],
    default_mass: float = 0.1,
) -> Path:
    """Fuse scene from demo config."""
    scene_cfg = demo_config["scene_cfg"]
    robot_cfg = demo_config["robot_cfg"]

    # Map paths from docker to local
    background_path = map_docker_path_to_local(
        scene_cfg["background"]["registered"], repo_root
    )
    # MJCF assets are in simulation/mujoco/mjcf/
    simulation_dir = background_path.parent
    asset_path = simulation_dir / "mujoco" / "mjcf"

    # Create scene_config compatible with scene_fusion
    scene_config = {
        "background": scene_cfg["background"],
        "objects": scene_cfg["objects"],
        "camera": scene_cfg.get("camera", {}),
        "robot_cfg": robot_cfg,
    }

    # Load object metadata
    object_metadata = load_object_metadata(asset_path, scene_config)

    # Load constants
    with open(constants_path, "r") as f:
        constants = yaml.safe_load(f)

    # Extract parameters
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

    # Create fusion object
    fusion = SceneFusion(
        asset_root=asset_path,
        masses=masses,
        z_offset=z_offset,
        inertia_scale=inertia_scale,
        freejoint_damping=freejoint_damping,
        robot_asset_prefix=robot_asset_prefix,
        collision_defaults=collision_defaults,
        contact_params=contact_params,
        groundplane_height=0.0,
        timestep=timestep,
        memory=memory,
        solver_iterations=solver_iterations,
        solver_ls_iterations=solver_ls_iterations,
        solver_noslip_iterations=solver_noslip_iterations,
        material_specular=material_specular,
        material_shininess=material_shininess,
    )

    # Fuse scene
    robot_scene_xml = robot_path / "scene.xml"
    robot_panda_xml = robot_path / "panda.xml"

    tree = fusion.fuse(
        robot_scene_path=robot_scene_xml,
        robot_panda_path=robot_panda_xml,
        scene_config=scene_config,
        object_metadata=object_metadata,
    )

    # Write output
    output_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)
    logger.info(f"Fused scene written to {output_xml}")

    return output_xml


def replay_trajectory_interactive(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    trajectory: list[dict],
    robot_config: dict,
    scene_config: dict,
    loop: bool = False,
):
    """Replay trajectory with PD control."""
    if len(trajectory) == 0:
        logger.error("Trajectory is empty")
        return

    logger.info(f"Loaded {len(trajectory)} keyframes")

    # Extract robot configuration
    joint_names = robot_config["joint_names"]
    gripper_joint_names = robot_config["gripper_joint_names"]
    control_frequency = robot_config["control_frequency"]
    arm_kp = np.array(robot_config["pd_gains"]["kp"][:7])
    arm_kd = np.array(robot_config["pd_gains"]["kd"][:7])
    gripper_kp = robot_config["gripper_control"]["kp"]
    gripper_kd = robot_config["gripper_control"]["kd"]
    gripper_max_opening = robot_config["gripper_control"]["max_opening"]

    # Find joint indices
    joint_qpos_idx = []
    joint_qvel_idx = []
    joint_actuator_idx = []

    for joint_name in joint_names:
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jnt_id == -1:
            raise ValueError(f"Joint '{joint_name}' not found in model")
        joint_qpos_idx.append(model.jnt_qposadr[jnt_id])
        joint_qvel_idx.append(model.jnt_dofadr[jnt_id])

        # Find actuator for this joint
        actuator_id = -1
        for act_id in range(model.nu):
            if model.actuator_trntype[act_id] == 0:  # mjTRN_JOINT
                if model.actuator_trnid[act_id, 0] == jnt_id:
                    actuator_id = act_id
                    break
        if actuator_id == -1:
            raise ValueError(f"No actuator found for joint '{joint_name}'")
        joint_actuator_idx.append(actuator_id)

    # Find gripper joint indices
    gripper_qpos_idx = []
    gripper_qvel_idx = []
    for gripper_joint_name in gripper_joint_names:
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, gripper_joint_name)
        if jnt_id == -1:
            logger.warning(f"Gripper joint '{gripper_joint_name}' not found")
            gripper_qpos_idx.append(-1)
            gripper_qvel_idx.append(-1)
            continue
        gripper_qpos_idx.append(model.jnt_qposadr[jnt_id])
        gripper_qvel_idx.append(model.jnt_dofadr[jnt_id])

    # Find gripper actuator (controls tendon)
    gripper_actuator_idx = -1
    for act_id in range(model.nu):
        if model.actuator_trntype[act_id] == 3:  # mjTRN_TENDON
            gripper_actuator_idx = act_id
            break

    # Initialize from first keyframe
    first_kf = trajectory[0]
    qpos = first_kf["qpos"]
    qvel = first_kf["qvel"]
    gripper = first_kf["gripper"]

    # Set initial joint positions and velocities
    for i in range(len(joint_qpos_idx)):
        data.qpos[joint_qpos_idx[i]] = qpos[i]
        data.qvel[joint_qvel_idx[i]] = qvel[i]

    # Set initial gripper position
    gripper_pos = gripper * gripper_max_opening
    for idx in gripper_qpos_idx:
        if idx != -1:
            data.qpos[idx] = gripper_pos

    # Initialize object positions from scene config
    # Objects have freejoint, need to set qpos: [x, y, z, qw, qx, qy, qz]
    for oid, obj_cfg in scene_config["objects"].items():
        obj_name = obj_cfg["name"]
        body_name = f"{oid}_{obj_name}_optimized"

        # Find body and its freejoint
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            logger.warning(f"Object body '{body_name}' not found in model")
            continue

        # Find the freejoint for this body
        # Look for a joint with qposadr that matches this body
        obj_joint_id = -1
        for jnt_id in range(model.njnt):
            if model.jnt_type[jnt_id] == 0 and model.jnt_bodyid[jnt_id] == body_id:  # free joint
                obj_joint_id = jnt_id
                break

        if obj_joint_id == -1:
            logger.warning(f"No freejoint found for object '{body_name}'")
            continue

        qpos_addr = model.jnt_qposadr[obj_joint_id]
        qvel_addr = model.jnt_dofadr[obj_joint_id]

        # In the scene XML, the inertial pos is set to object_center (relative to body frame)
        # So if we place the body frame at origin, the COM will be at object_center
        # This is the correct initialization - body at origin, COM offset by inertial pos

        # Set position to origin (x, y, z)
        data.qpos[qpos_addr:qpos_addr+3] = [0, 0, 0]
        # Set orientation to identity quaternion (qw, qx, qy, qz)
        data.qpos[qpos_addr+3:qpos_addr+7] = [1, 0, 0, 0]

        # IMPORTANT: Set velocities to zero to prevent bouncing
        data.qvel[qvel_addr:qvel_addr+6] = 0

        obj_center = obj_cfg["object_center"]
        logger.debug(f"Initialized object '{obj_name}' with COM at {obj_center}")

    # Forward kinematics to compute derived quantities
    mujoco.mj_forward(model, data)

    # Save initial state for reset
    initial_qpos = data.qpos.copy()
    initial_qvel = data.qvel.copy()

    # Control state
    class ControlState:
        def __init__(self):
            self.paused = False
            self.should_reset = False
            self.frame_idx = 0
            self.last_frame_time = time.time()
            self.target_qpos = qpos[:7]
            self.target_qvel = qvel[:7]
            self.target_gripper = gripper

    state = ControlState()

    # Keyboard callback - only set flags, don't modify state directly
    def key_callback(keycode):
        try:
            key = chr(keycode)
        except ValueError:
            return

        if key == ' ':  # Space: pause/resume
            state.paused = not state.paused
            status = "PAUSED" if state.paused else "PLAYING"
            logger.info(status)
        elif key == 'r' or key == 'R':  # R: restart
            state.should_reset = True
            logger.info("Restarting trajectory")

    # Launch viewer
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20
        viewer.cam.distance = 1.5
        viewer.cam.lookat[:] = [0, 0, 0.1]

        logger.info("="*60)
        logger.info("TRAJECTORY PLAYBACK")
        logger.info("="*60)
        logger.info(f"  Total frames: {len(trajectory)}")
        logger.info(f"  Loop: {loop}")
        logger.info(f"  Control frequency: {control_frequency} Hz")
        logger.info("="*60)
        logger.info("\nKEYBOARD CONTROLS:")
        logger.info("  SPACE     - Play/Pause")
        logger.info("  R         - Restart from beginning")
        logger.info("  ESC/Close - Exit")
        logger.info("="*60)

        while viewer.is_running():
            current_time = time.time()

            # Handle reset request (thread-safe with viewer lock)
            if state.should_reset:
                with viewer.lock():
                    data.qpos[:] = initial_qpos
                    data.qvel[:] = initial_qvel
                    mujoco.mj_forward(model, data)

                state.frame_idx = 0
                state.last_frame_time = time.time()
                state.target_qpos = qpos[:7]
                state.target_qvel = qvel[:7]
                state.target_gripper = gripper
                state.paused = False
                state.should_reset = False

            if not state.paused:
                # Check if we need to advance to next keyframe
                dt = 1.0 / control_frequency

                if current_time - state.last_frame_time >= dt:
                    state.frame_idx += 1
                    state.last_frame_time = current_time

                    if state.frame_idx >= len(trajectory):
                        if loop:
                            logger.info("Looping trajectory")
                            state.frame_idx = 0
                        else:
                            logger.info("Trajectory complete. Press R to restart or ESC to exit.")
                            state.paused = True
                            state.frame_idx = len(trajectory) - 1

                    if state.frame_idx < len(trajectory):
                        kf = trajectory[state.frame_idx]
                        state.target_qpos = kf["qpos"][:7]
                        state.target_qvel = kf["qvel"][:7]
                        state.target_gripper = kf["gripper"]

                        if state.frame_idx % 30 == 0:
                            logger.info(
                                f"Frame {state.frame_idx + 1}/{len(trajectory)} "
                                f"({100.0 * state.frame_idx / len(trajectory):.1f}%)"
                            )

                # Apply PD control with gravity compensation
                # ARM: PD control
                mujoco.mj_inverse(model, data)

                for i in range(7):
                    q_desired = state.target_qpos[i]
                    qvel_desired = state.target_qvel[i]
                    q_current = data.qpos[joint_qpos_idx[i]]
                    qvel_current = data.qvel[joint_qvel_idx[i]]
                    gravity_comp = data.qfrc_inverse[joint_qvel_idx[i]]

                    tau = (
                        arm_kp[i] * (q_desired - q_current) +
                        arm_kd[i] * (qvel_desired - qvel_current) +
                        gravity_comp
                    )
                    data.ctrl[joint_actuator_idx[i]] = tau

                # GRIPPER: PD control
                if gripper_actuator_idx != -1:
                    q_desired_gripper = state.target_gripper * gripper_max_opening

                    # Get average gripper position
                    q_current_gripper = 0
                    qvel_current_gripper = 0
                    valid_count = 0
                    for i, idx in enumerate(gripper_qpos_idx):
                        if idx != -1:
                            q_current_gripper += data.qpos[idx]
                            qvel_current_gripper += data.qvel[gripper_qvel_idx[i]]
                            valid_count += 1

                    if valid_count > 0:
                        q_current_gripper /= valid_count
                        qvel_current_gripper /= valid_count

                        tau_gripper = (
                            gripper_kp * (q_desired_gripper - q_current_gripper) -
                            gripper_kd * qvel_current_gripper
                        )
                        data.ctrl[gripper_actuator_idx] = tau_gripper

                # Step simulation
                mujoco.mj_step(model, data)

            # Sync viewer (always, to keep window responsive)
            viewer.sync()

            # Small sleep to prevent busy waiting
            time.sleep(0.001)

    logger.info("Viewer closed")


def main(
    demo_path: Path,
    constants_path: Path = DEFAULT_CONSTANTS_PATH,
    robot_config_path: Path = DEFAULT_ROBOT_CONFIG_PATH,
    object_mass: List[str] = [],
    default_mass: float = 0.1,
    loop: bool = False,
) -> int:
    """Replay trajectory from demo directory.

    Args:
        demo_path: Path to demo directory
        constants_path: Path to constants YAML file
        robot_config_path: Path to robot configuration YAML file
        object_mass: Object masses as key=value pairs
        default_mass: Default mass for objects not specified
        loop: Loop the trajectory playback
    """
    demo_path = demo_path.expanduser().resolve()
    constants_path = constants_path.expanduser().resolve()
    robot_config_path = robot_config_path.expanduser().resolve()
    robot_path = DEFAULT_ROBOT_PATH.expanduser().resolve()

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

    # Load demo config
    logger.info(f"Loading demo config from {demo_path}")
    demo_config = load_demo_config(demo_path)

    # Load robot config
    logger.info(f"Loading robot config from {robot_config_path}")
    with open(robot_config_path, "r") as f:
        robot_config = yaml.safe_load(f)

    # Load trajectory
    logger.info(f"Loading trajectory from {demo_path}")
    trajectory = load_trajectory_from_npy(demo_path)

    # Always fuse scene from demo config (don't use existing scene.xml)
    # Save to simulation/mujoco/replay_scene.xml to maintain correct relative paths
    # demo_path: outputs/<scene_name>/demos/demo_N/env_NNN
    demo_parts = demo_path.parts
    if "demos" in demo_parts:
        demos_idx = demo_parts.index("demos")
        scene_root = Path(*demo_parts[:demos_idx])
        output_xml = scene_root / "simulation" / "mujoco" / "replay_scene.xml"
    else:
        raise ValueError(f"Unexpected demo path structure: {demo_path}")

    logger.info("Fusing scene from demo config...")
    scene_xml = fuse_scene_from_config(
        demo_config=demo_config,
        repo_root=REPO_ROOT,
        robot_path=robot_path,
        constants_path=constants_path,
        output_xml=output_xml,
        mass_dict=mass_dict,
        default_mass=default_mass,
    )

    # Load MuJoCo model
    logger.info(f"Loading MuJoCo model from {scene_xml}")
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    logger.info(f"Model loaded: {model.ntex} textures, {model.nmesh} meshes")

    # Replay trajectory
    logger.info("Starting trajectory replay...")
    try:
        replay_trajectory_interactive(
            model=model,
            data=data,
            trajectory=trajectory,
            robot_config=robot_config,
            scene_config=demo_config["scene_cfg"],
            loop=loop,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    return 0


if __name__ == "__main__":
    raise SystemExit(tyro.cli(main))
