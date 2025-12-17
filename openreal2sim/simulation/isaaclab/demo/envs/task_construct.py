### construct the task config from scene dict.

import json
import numpy as np
from enum import Enum
import shutil
from pathlib import Path
import os
import sys
from typing import List
file_path = Path(__file__).resolve()
sys.path.append(str(file_path.parent))
from .task_cfg import TaskCfg, TaskType, ObjectCfg, CameraInfo, BackgroundCfg, TrajectoryCfg, SuccessMetric, SuccessMetricType, RobotType

def get_next_id(folder: Path) -> int:
    if not folder.exists():
        os.makedirs(folder, exist_ok=True)
        return 0
    subfolders = [f for f in folder.iterdir() if f.is_dir()]
    task_num = len(subfolders)
    return task_num

def get_task_cfg(key: str, base_folder: Path) -> TaskCfg:
    json_path = base_folder / "task.json"
    return load_task_cfg(json_path)

def construct_task_config(key, scene_dict: dict, base_folder: Path):
    task_key = key
    task_id = get_next_id(base_folder)
    task_desc = scene_dict["task_desc"]
    base_folder = base_folder / key
    if base_folder.exists():
        shutil.rmtree(base_folder)
    base_folder.mkdir(parents=True, exist_ok=True)  # Create directory before copying files
    background_mesh_path = scene_dict["background"]["registered"]
    background_usd_path = scene_dict["background"]["usd"]
    shutil.copy(background_mesh_path, base_folder / "background.glb")
    shutil.copy(background_usd_path, base_folder / "background.usd")
    background_mesh_path = base_folder / "background.glb"
    background_usd_path = base_folder / "background.usd"
    background_rgb_path = scene_dict["background_image"]
    shutil.copy(background_rgb_path, base_folder / "bg_rgb.jpg")
    background_rgb_path = base_folder / "bg_rgb.jpg"
    background_point = scene_dict["groundplane_in_cam"]["point"]
    background_cfg = BackgroundCfg(str(background_rgb_path), str(background_mesh_path), str(background_usd_path), background_point)
    width = scene_dict["camera"]["width"]
    height = scene_dict["camera"]["height"]
    fx = scene_dict["camera"]["fx"]
    fy = scene_dict["camera"]["fy"]
    cx = scene_dict["camera"]["cx"]
    cy = scene_dict["camera"]["cy"]
    camera_opencv_to_world = scene_dict["camera"]["camera_opencv_to_world"]
    camera_position = scene_dict["camera"]["camera_position"]
    camera_heading_wxyz = scene_dict["camera"]["camera_heading_wxyz"]
    camera_info = CameraInfo(width, height, fx, fy, cx, cy, camera_opencv_to_world, camera_position, camera_heading_wxyz)

    objects = []
    for oid, obj in scene_dict["objects"].items():
        object_id = oid
        object_name = obj["name"]
        mesh_path = obj["optimized"]
        shutil.copy(mesh_path, base_folder / f"object_{object_id}.glb")
        mesh_path = base_folder / f"object_{object_id}.glb"
        usd_path = obj['usd']
        
        shutil.copy(usd_path, base_folder / Path(usd_path).name)
       
        cfg_path = Path(usd_path).parent / "config.yaml"
        asset_hash_path = Path(usd_path).parent / ".asset_hash"
        usd_path = base_folder / Path(usd_path).name
        shutil.copy(cfg_path, base_folder / "config.yaml")
        shutil.copy(asset_hash_path, base_folder / ".asset_hash")
        object_cfg = ObjectCfg(object_id, object_name, str(mesh_path), str(usd_path))
        objects.append(object_cfg)


    manipulated_oid = scene_dict["manipulated_oid"]
    start_related = scene_dict["start_related"]
    end_related = scene_dict["end_related"]

    if scene_dict["task_type"] == "targetted_pick_place":
        task_type = TaskType.TARGETTED_PICK_PLACE
    elif scene_dict["task_type"] == "simple_pick_place":
        task_type = TaskType.SIMPLE_PICK_PLACE
    elif scene_dict["task_type"] == "simple_pick":
        task_type = TaskType.SIMPLE_PICK
    else:
        raise ValueError(f"Invalid task type: {scene_dict['info']['task_type']}")
    task_config = TaskCfg(task_key, task_id, task_desc, task_type, background_cfg, camera_info, manipulated_oid, start_related, end_related, objects)
    json_path = base_folder / "task.json"
    with open(json_path, "w") as f:
        json.dump(serialize_task_cfg(task_config), f)
    return task_config, base_folder



def serialize_task_cfg(task_cfg):
    """
    Serialize TaskCfg and all nested fields (including numpy arrays) into pure Python dict/list/primitive types,
    so it can be safely saved as JSON.
    """

    def serialize(obj):
        # Handle None
        if obj is None:
            return None
        # Handle basic types
        elif isinstance(obj, (int, float, str, bool)):
            return obj
        # Handle numpy array
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # Handle enum
        elif hasattr(obj, 'name') and isinstance(obj, (Enum,)):
            return obj.name
        # Handle dict
        elif isinstance(obj, dict):
            return {serialize(k): serialize(v) for k, v in obj.items()}
        # Handle list/tuple
        elif isinstance(obj, (list, tuple)):
            return [serialize(i) for i in obj]
        # Handle dataclass/object with __dict__
        elif hasattr(obj, '__dict__'):
            data = {}
            for key, value in obj.__dict__.items():
                data[key] = serialize(value)
            return data
        # Handle class with __slots__
        elif hasattr(obj, '__slots__'):
            data = {}
            for slot in obj.__slots__:
                data[slot] = serialize(getattr(obj, slot))
            return data
        # Fallback (e.g. Path objects)
        elif hasattr(obj, '__str__'):
            return str(obj)
        else:
            raise TypeError(f"Cannot serialize object of type {type(obj)}: {repr(obj)}")

    return serialize(task_cfg)


def add_reference_trajectory(task_cfg: TaskCfg, reference_trajectory: List[TrajectoryCfg], base_folder: Path):
    task_cfg.reference_trajectory = task_cfg.reference_trajectory + reference_trajectory
    json_path = base_folder / "task.json"
    with open(json_path, "w") as f:
        json.dump(serialize_task_cfg(task_cfg), f)
    return task_cfg

def add_generated_trajectories(task_cfg: TaskCfg, generated_trajectories: List[TrajectoryCfg], base_folder: Path):
    original_generated_trajectories = task_cfg.generated_trajectories
    task_cfg.generated_trajectories = original_generated_trajectories + generated_trajectories
    json_path = base_folder / "task.json"
    with open(json_path, "w") as f:
        json.dump(serialize_task_cfg(task_cfg), f)
    return task_cfg


def load_task_cfg(json_path: Path) -> TaskCfg:
    """
    Load a TaskCfg from the given JSON file path and construct a TaskCfg instance.
    """
    with open(json_path, "r") as f:
        cfg_dict = json.load(f)

    # Handle all fields and reconstruct proper datatypes
    # Helper to reconstruct enums
    def parse_enum(enum_cls, val):
        if isinstance(val, enum_cls):
            return val
        elif isinstance(val, str):
            return enum_cls[val]
        else:
            raise ValueError(f"Unknown value {val} for enum {enum_cls}")

    # Parse SuccessMetric(s)
    def parse_success_metric(metric_dict):
        return SuccessMetric(
            success_metric_type=parse_enum(SuccessMetricType, metric_dict["success_metric_type"]),
            final_gripper_close=metric_dict["final_gripper_close"],
            lift_height=metric_dict.get("lift_height", None),
            ground_value=metric_dict.get("ground_value", None),
            end_pose=metric_dict.get("end_pose", None)
        )

    # Parse TrajectoryCfg(s)
    def parse_traj_cfg(traj_dict):
        return TrajectoryCfg(
            robot_pose=np.array(traj_dict["robot_pose"], dtype=np.float32).tolist(),
            object_poses={oid: np.array(pose, dtype=np.float32).tolist() for oid, pose in traj_dict["object_poses"].items()},
            object_trajectory=[np.array(m, dtype=np.float32).tolist() for m in traj_dict["object_trajectory"]],
            final_gripper_close=traj_dict.get("final_gripper_close", None),
            success_metric=parse_success_metric(traj_dict.get("success_metric", None)),
            pregrasp_pose=traj_dict.get("pregrasp_pose", None),
            grasp_pose=traj_dict.get("grasp_pose", None),
            robot_type=parse_enum(RobotType, traj_dict.get("robot_type", None))
        )

    def parse_camera_info(camera_dict):
        return CameraInfo(
            width=camera_dict["width"],
            height=camera_dict["height"],
            fx=camera_dict["fx"],
            fy=camera_dict["fy"],
            cx=camera_dict["cx"],
            cy=camera_dict["cy"],
            camera_opencv_to_world=np.array(camera_dict["camera_opencv_to_world"], dtype=np.float32).tolist(),
            camera_position=np.array(camera_dict["camera_position"], dtype=np.float32).tolist(),
            camera_heading_wxyz=np.array(camera_dict["camera_heading_wxyz"], dtype=np.float32).tolist(),
        )
    def parse_object_cfg(object_dict):
        return ObjectCfg(
            object_id=object_dict["object_id"],
            object_name=object_dict["object_name"],
            mesh_path=object_dict["mesh_path"],
            usd_path=object_dict["usd_path"]
        )
    def parse_background_cfg(background_dict):
        return BackgroundCfg(
            background_rgb_path=background_dict["background_rgb_path"],
            background_mesh_path=background_dict["background_mesh_path"],
            background_usd_path=background_dict["background_usd_path"],
            background_point=np.array(background_dict["background_point"], dtype=np.float32).tolist()
        )
    # Compose TaskCfg
    task_cfg = TaskCfg(
        task_id=cfg_dict["task_id"],
        task_desc=cfg_dict["task_desc"],
        task_key=cfg_dict["task_key"],
        task_type=parse_enum(TaskType, cfg_dict["task_type"]),
        background_cfg=parse_background_cfg(cfg_dict["background_cfg"]),
        camera_info=parse_camera_info(cfg_dict["camera_info"]),
        manipulated_oid=cfg_dict["manipulated_oid"],
        start_related=cfg_dict["start_related"],
        end_related=cfg_dict["end_related"],
        objects=[parse_object_cfg(obj) for obj in cfg_dict["objects"]],
        reference_trajectory=[
            parse_traj_cfg(traj) for traj in (cfg_dict.get("reference_trajectory") or [])
        ],
        generated_trajectories=[
            parse_traj_cfg(traj) for traj in (cfg_dict.get("generated_trajectories") or [])
        ]
    )

    return task_cfg