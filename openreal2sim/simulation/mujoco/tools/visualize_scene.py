#!/usr/bin/env python3
"""
MuJoCo Scene Viewer with Runtime Fusion and Trajectory Playback

Fuses robot and objects at runtime, then visualizes with trajectory playback.

Usage:
    python visualize_scene.py --demo-path outputs/genvideo/simulation --dump-xml scene_debug.xml
"""

import argparse
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict
import numpy as np
import mujoco
import mujoco.viewer
from loguru import logger

CURRENT_DIR = Path(__file__).resolve().parent
MUJOCO_DIR = CURRENT_DIR.parent
if str(MUJOCO_DIR) not in sys.path:
    sys.path.insert(0, str(MUJOCO_DIR))

from mujoco_asset_builder import constants


class SceneFuser:
    """Fuses robot and object MJCF files at runtime."""

    def __init__(self, demo_path: Path, asset_path: Path, robot_path: Path, object_masses: dict):
        self.demo_path = demo_path
        self.asset_path = asset_path
        self.robot_path = robot_path
        self.object_masses = object_masses
        self.z_offset = constants.DEFAULT_Z_OFFSET
        self.inertia_scale = constants.DEFAULT_INERTIA_SCALE

    def fuse_scene(self, scene_config: dict) -> str:
        logger.info("Fusing scene at runtime...")

        scene_tree = ET.parse(self.robot_path / constants.FILENAME_SCENE_XML)
        panda_tree = ET.parse(self.robot_path / constants.FILENAME_PANDA_XML)
        scene_root = scene_tree.getroot()
        root = panda_tree.getroot()

        compiler = root.find('compiler')
        if compiler is None:
            compiler = ET.Element('compiler')
            root.insert(0, compiler)

        old_meshdir = compiler.get('meshdir')
        if old_meshdir:
            mesh_prefix = f'{self.robot_path}/{old_meshdir}/'
            mesh_count = 0
            for mesh_elem in root.iter('mesh'):
                file_attr = mesh_elem.get('file')
                if file_attr and not file_attr.startswith(str(self.robot_path)):
                    mesh_elem.set('file', mesh_prefix + file_attr)
                    mesh_count += 1
            logger.debug(f"Updated {mesh_count} mesh file paths")

        compiler.set('meshdir', constants.MESHDIR_VFS)
        compiler.set('texturedir', constants.TEXTUREDIR_VFS)
        logger.debug(f"Set meshdir=\"{constants.MESHDIR_VFS}\" texturedir=\"{constants.TEXTUREDIR_VFS}\"")

        option_elem = root.find('option')
        if option_elem is not None and not option_elem.get('timestep'):
            option_elem.set('timestep', constants.DEFAULT_TIMESTEP)
            logger.debug(f"Added timestep=\"{constants.DEFAULT_TIMESTEP}\"")

        size_elem = root.find('size')
        if size_elem is None:
            size_elem = ET.Element('size')
            option_elem = root.find('option')
            insert_at = list(root).index(option_elem) + 1 if option_elem is not None else 0
            root.insert(insert_at, size_elem)
        size_elem.set('memory', constants.DEFAULT_MEMORY)
        logger.debug(f"Set <size memory=\"{constants.DEFAULT_MEMORY}\">")

        default_elem = root.find('default')
        if default_elem is None:
            default_elem = ET.SubElement(root, 'default')
            if compiler is not None:
                root.remove(default_elem)
                idx = list(root).index(compiler) + 1
                root.insert(idx, default_elem)

        obj_visual = ET.SubElement(default_elem, 'default', {'class': constants.CLASS_OBJ_VISUAL})
        ET.SubElement(obj_visual, 'geom', {
            'group': constants.GROUP_VISUAL,
            'type': constants.GEOM_TYPE_MESH,
            'contype': constants.CONTYPE_NONE,
            'conaffinity': constants.CONAFFINITY_NONE
        })

        obj_collision = ET.SubElement(default_elem, 'default', {'class': constants.CLASS_OBJ_COLLISION})
        ET.SubElement(obj_collision, 'geom', {
            'group': constants.GROUP_COLLISION,
            'type': constants.GEOM_TYPE_MESH,
            'margin': constants.OBJ_COLLISION_MARGIN,
            'solref': constants.OBJ_COLLISION_SOLREF,
            'solimp': constants.OBJ_COLLISION_SOLIMP
        })
        logger.debug("Added object collision classes")

        asset_elem = root.find('asset')
        insert_idx = list(root).index(asset_elem) if asset_elem is not None else len(root)

        for elem_name in ['statistic', 'visual']:
            elem = scene_root.find(elem_name)
            if elem is not None:
                existing = root.find(elem_name)
                if existing is not None:
                    root.remove(existing)
                root.insert(insert_idx, elem)
                insert_idx += 1
                logger.debug(f"Merged <{elem_name}>")

        scene_asset = scene_root.find('asset')
        if scene_asset is not None:
            if asset_elem is None:
                asset_elem = ET.SubElement(root, 'asset')
            for child in scene_asset:
                asset_elem.append(child)

        scene_worldbody = scene_root.find('worldbody')
        worldbody = root.find('worldbody')
        if worldbody is None:
            worldbody = ET.SubElement(root, 'worldbody')

        if scene_worldbody is not None:
            for child in scene_worldbody:
                worldbody.append(child)

        bg_metadata = self._get_object_metadata(f"{constants.ASSET_TYPE_BACKGROUND}_registered")
        if bg_metadata:
            self._add_background_assets(asset_elem, bg_metadata)
            self._add_background_geoms(worldbody, bg_metadata)
            logger.info(f"Added background ({bg_metadata['collision_parts']} collision parts)")

        objects = scene_config.get('objects', {})
        for oid, obj_data in sorted(objects.items()):
            obj_name = obj_data['name']
            body_name = f"{oid}_{obj_name}{constants.SUFFIX_OPTIMIZED}"
            metadata = self._get_object_metadata(body_name)

            if metadata:
                self._add_object_assets(asset_elem, oid, obj_name, metadata)
                self._add_object_body(worldbody, oid, obj_name, obj_data, metadata)
                logger.info(f"Added {body_name} ({metadata['collision_parts']} collision parts)")

        ET.indent(root, space='  ')
        xml_string = ET.tostring(root, encoding='unicode')
        logger.info("Scene fused successfully")
        return xml_string

    def _get_object_metadata(self, obj_name: str) -> dict:
        metadata_path = self.asset_path / obj_name / f"{obj_name}{constants.SUFFIX_METADATA}"
        if metadata_path.exists():
            with open(metadata_path) as f:
                return json.load(f)
        return None

    def _add_background_assets(self, asset_elem: ET.Element, metadata: dict):
        bg_name = f"{constants.ASSET_TYPE_BACKGROUND}_registered"
        base_path = self.asset_path / bg_name

        ET.SubElement(asset_elem, 'texture', {
            'type': '2d',
            'name': f'{bg_name}_{constants.MATERIAL_PREFIX}',
            'file': str(base_path / f'{constants.MATERIAL_PREFIX}{constants.EXT_PNG}')
        })
        ET.SubElement(asset_elem, 'material', {
            'name': f'{bg_name}_{constants.MATERIAL_PREFIX}',
            'texture': f'{bg_name}_{constants.MATERIAL_PREFIX}',
            'specular': constants.MATERIAL_SPECULAR,
            'shininess': constants.MATERIAL_SHININESS
        })
        ET.SubElement(asset_elem, 'mesh', {
            'file': str(base_path / f'{bg_name}{constants.EXT_OBJ}')
        })

        for i in range(metadata['collision_parts']):
            ET.SubElement(asset_elem, 'mesh', {
                'file': str(base_path / f'{bg_name}{constants.SUFFIX_COLLISION}{i}{constants.EXT_OBJ}')
            })

    def _add_background_geoms(self, worldbody: ET.Element, metadata: dict):
        bg_name = f"{constants.ASSET_TYPE_BACKGROUND}_registered"

        ET.SubElement(worldbody, 'geom', {
            'name': f'{bg_name}_visual',
            'class': constants.CLASS_OBJ_VISUAL,
            'mesh': bg_name,
            'material': f'{bg_name}_{constants.MATERIAL_PREFIX}'
        })

        for i in range(metadata['collision_parts']):
            ET.SubElement(worldbody, 'geom', {
                'name': f'{bg_name}{constants.SUFFIX_COLLISION}{i}',
                'mesh': f'{bg_name}{constants.SUFFIX_COLLISION}{i}',
                'class': constants.CLASS_OBJ_COLLISION
            })

    def _add_object_assets(self, asset_elem: ET.Element, oid: str, obj_name: str, metadata: dict):
        body_name = f"{oid}_{obj_name}{constants.SUFFIX_OPTIMIZED}"
        base_path = self.asset_path / body_name

        ET.SubElement(asset_elem, 'texture', {
            'type': '2d',
            'name': f'{obj_name}_{constants.MATERIAL_PREFIX}',
            'file': str(base_path / f'{constants.MATERIAL_PREFIX}{constants.EXT_PNG}')
        })
        ET.SubElement(asset_elem, 'material', {
            'name': f'{obj_name}_{constants.MATERIAL_PREFIX}',
            'texture': f'{obj_name}_{constants.MATERIAL_PREFIX}',
            'specular': constants.MATERIAL_SPECULAR,
            'shininess': constants.MATERIAL_SHININESS
        })
        ET.SubElement(asset_elem, 'mesh', {
            'file': str(base_path / f'{body_name}{constants.EXT_OBJ}')
        })

        for i in range(metadata['collision_parts']):
            ET.SubElement(asset_elem, 'mesh', {
                'file': str(base_path / f'{body_name}{constants.SUFFIX_COLLISION}{i}{constants.EXT_OBJ}')
            })

    def _add_object_body(self, worldbody: ET.Element, oid: str, obj_name: str, obj_data: dict, metadata: dict):
        body_name = f"{oid}_{obj_name}{constants.SUFFIX_OPTIMIZED}"
        obj_center = obj_data['object_center']

        if obj_name not in self.object_masses:
            raise ValueError(f"No mass specified for object '{obj_name}'")
        mass = self.object_masses[obj_name]

        inertia = mass * self.inertia_scale
        diag_inertia = f"{inertia:.6f} {inertia:.6f} {inertia:.6f}"

        body = ET.SubElement(worldbody, 'body', {
            'name': body_name,
            'pos': f'0 0 {self.z_offset}'
        })

        ET.SubElement(body, 'joint', {'type': constants.JOINT_TYPE_FREE, 'damping': constants.OBJ_FREEJOINT_DAMPING})
        ET.SubElement(body, 'inertial', {
            'pos': f"{obj_center[0]} {obj_center[1]} {obj_center[2]}",
            'mass': str(mass),
            'diaginertia': diag_inertia
        })
        ET.SubElement(body, 'geom', {
            'class': constants.CLASS_OBJ_VISUAL,
            'mesh': body_name,
            'material': f'{obj_name}_{constants.MATERIAL_PREFIX}'
        })

        for i in range(metadata['collision_parts']):
            ET.SubElement(body, 'geom', {
                'mesh': f'{body_name}{constants.SUFFIX_COLLISION}{i}',
                'class': constants.CLASS_OBJ_COLLISION
            })


class TrajectoryPlayer:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, scene_config: dict,
                 robot_config: dict, demo_path: Path):
        self.model = model
        self.data = data
        self.scene_config = scene_config
        self.robot_config = robot_config
        self.demo_path = demo_path

        self.joint_names = robot_config["joint_names"]
        self.gripper_joint_names = robot_config["gripper_joint_names"]
        self.control_frequency = robot_config["control_frequency"]
        self.kp = np.array(robot_config["pd_gains"]["kp"])
        self.kd = np.array(robot_config["pd_gains"]["kd"])
        self.gripper_max_opening = robot_config["gripper_control"]["max_opening"]

        self._find_joint_indices()
        self.robot_trajectory = self._load_robot_trajectory()
        self.object_trajectories = self._load_object_trajectories()

        self.trajectory_index = 0
        self.control_step_counter = 0
        self.simulation_frequency = 1.0 / model.opt.timestep
        self.control_decimation = int(self.simulation_frequency / self.control_frequency)

        logger.info(f"Trajectory player initialized")
        logger.info(f"Robot trajectory: {len(self.robot_trajectory)} frames")
        logger.info(f"Object trajectories: {list(self.object_trajectories.keys())}")
        logger.info(f"Control: {self.control_frequency} Hz, Simulation: {self.simulation_frequency:.0f} Hz")

    def _find_joint_indices(self):
        self.joint_qpos_indices = []
        self.joint_qvel_indices = []
        self.actuator_indices = []

        for joint_name in self.joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                self.joint_qpos_indices.append(self.model.jnt_qposadr[joint_id])
                self.joint_qvel_indices.append(self.model.jnt_dofadr[joint_id])
                actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
                if actuator_id >= 0:
                    self.actuator_indices.append(actuator_id)

        self.gripper_qpos_indices = []
        self.gripper_qvel_indices = []
        self.gripper_actuator_indices = []

        for gripper_joint_name in self.gripper_joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, gripper_joint_name)
            if joint_id >= 0:
                self.gripper_qpos_indices.append(self.model.jnt_qposadr[joint_id])
                self.gripper_qvel_indices.append(self.model.jnt_dofadr[joint_id])
                actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, gripper_joint_name)
                if actuator_id >= 0:
                    self.gripper_actuator_indices.append(actuator_id)

    def _load_robot_trajectory(self) -> np.ndarray:
        traj_path = self.demo_path / constants.FILENAME_TRAJECTORY
        with open(traj_path) as f:
            trajectory = json.load(f)
        return np.array(trajectory, dtype=np.float32)

    def _load_object_trajectories(self) -> Dict[str, np.ndarray]:
        trajectories = {}
        objects = self.scene_config.get("objects", {})

        for oid, obj_data in objects.items():
            obj_name = obj_data["name"]
            body_name = f"{oid}_{obj_name}{constants.SUFFIX_OPTIMIZED}"

            for traj_key in ["hybrid_trajs", "simple_trajs", "fdpose_trajs"]:
                if traj_key in obj_data:
                    traj_filename = Path(obj_data[traj_key]).name
                    traj_path = self.demo_path / traj_filename

                    if traj_path.exists():
                        trajectories[body_name] = np.load(traj_path)
                        logger.debug(f"Loaded {traj_key} for {body_name}: {trajectories[body_name].shape}")
                        break

        return trajectories

    def _set_object_pose(self, body_name: str, transform: np.ndarray):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            return

        for joint_id in range(self.model.njnt):
            if (self.model.jnt_bodyid[joint_id] == body_id and
                self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE):
                qpos_adr = self.model.jnt_qposadr[joint_id]

                pos = transform[:3, 3]
                from scipy.spatial.transform import Rotation
                rot_matrix = transform[:3, :3]
                quat_xyzw = Rotation.from_matrix(rot_matrix).as_quat()
                quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

                self.data.qpos[qpos_adr:qpos_adr+3] = pos
                self.data.qpos[qpos_adr+3:qpos_adr+7] = quat_wxyz
                break

    def _apply_pd_control(self, target_qpos: np.ndarray):
        for i, (qpos_idx, qvel_idx, act_idx) in enumerate(
            zip(self.joint_qpos_indices, self.joint_qvel_indices, self.actuator_indices)
        ):
            tau = self.kp[i] * (target_qpos[i] - self.data.qpos[qpos_idx]) - self.kd[i] * self.data.qvel[qvel_idx]
            self.data.ctrl[act_idx] = tau

        gripper_cmd = target_qpos[7]
        gripper_target = self.gripper_max_opening if gripper_cmd > 0.5 else 0.0

        for i, (qpos_idx, qvel_idx, act_idx) in enumerate(
            zip(self.gripper_qpos_indices, self.gripper_qvel_indices, self.gripper_actuator_indices)
        ):
            kp_idx = 7 + i
            tau = self.kp[kp_idx] * (gripper_target - self.data.qpos[qpos_idx]) - self.kd[kp_idx] * self.data.qvel[qvel_idx]
            self.data.ctrl[act_idx] = tau

    def step(self):
        self.control_step_counter += 1

        if self.control_step_counter >= self.control_decimation:
            self.control_step_counter = 0

            if self.trajectory_index < len(self.robot_trajectory):
                target_qpos = self.robot_trajectory[self.trajectory_index]
                self._apply_pd_control(target_qpos)

                for body_name, trajectory in self.object_trajectories.items():
                    if self.trajectory_index < len(trajectory):
                        self._set_object_pose(body_name, trajectory[self.trajectory_index])

                self.trajectory_index += 1
            else:
                self.trajectory_index = 0

        mujoco.mj_step(self.model, self.data)

    def reset(self):
        self.trajectory_index = 0
        self.control_step_counter = 0
        mujoco.mj_resetData(self.model, self.data)


def main():
    parser = argparse.ArgumentParser(description="MuJoCo scene viewer with runtime fusion")
    parser.add_argument("--demo-path", type=Path, required=True, help="Path to demo directory")
    parser.add_argument("--dump-xml", type=Path, help="Optional: dump fused XML to file for debugging")
    parser.add_argument("--asset-path", type=Path, help=f"Path to preprocessed assets (default: demo-path/{constants.DIR_MJCF_ASSETS})")
    parser.add_argument("--robot-path", type=Path, help="Path to robot XMLs (default: built-in)")
    args = parser.parse_args()

    if args.asset_path is None:
        args.asset_path = args.demo_path / constants.DIR_MJCF_ASSETS

    if args.robot_path is None:
        script_dir = Path(__file__).parent.parent
        args.robot_path = script_dir / constants.DIR_ROBOTS

    logger.info(f"Loading scene config: {args.demo_path / constants.FILENAME_SCENE}")
    with open(args.demo_path / constants.FILENAME_SCENE) as f:
        scene_config = json.load(f)

    robot_config_path = Path(__file__).parent.parent / constants.DIR_CONFIG / constants.FILENAME_PANDA_CONFIG
    logger.info(f"Loading robot config: {robot_config_path}")
    with open(robot_config_path) as f:
        robot_config = json.load(f)

    masses_path = Path(__file__).parent.parent / constants.DIR_CONFIG / constants.FILENAME_OBJECT_MASSES
    logger.info(f"Loading object masses: {masses_path}")
    with open(masses_path) as f:
        object_masses = json.load(f)

    fuser = SceneFuser(args.demo_path, args.asset_path, args.robot_path, object_masses)
    fused_xml = fuser.fuse_scene(scene_config)

    if args.dump_xml:
        args.dump_xml.write_text(fused_xml)
        logger.info(f"Dumped fused XML to: {args.dump_xml}")

    logger.info("Loading MuJoCo model...")
    with tempfile.NamedTemporaryFile(mode='w', suffix=constants.EXT_XML, delete=False) as f:
        f.write(fused_xml)
        temp_path = f.name

    try:
        model = mujoco.MjModel.from_xml_path(temp_path)
        data = mujoco.MjData(model)
        logger.info(f"Model loaded: {model.nq} qpos, {model.nv} qvel, {model.nu} actuators")

        player = TrajectoryPlayer(model, data, scene_config, robot_config, args.demo_path)

        logger.info("Launching viewer...")
        logger.info("Controls: Space=Pause/Resume, R=Reset, ESC=Exit")

        with mujoco.viewer.launch_passive(model, data) as viewer:
            player.reset()
            while viewer.is_running():
                player.step()
                viewer.sync()

    finally:
        Path(temp_path).unlink()


if __name__ == "__main__":
    main()
