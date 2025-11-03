# -*- coding: utf-8 -*-
"""Scene configuration loader for OpenReal2Sim outputs."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class CameraConfig:
    """Camera configuration from scene.json."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    extrinsic_matrix: list  # 4x4 matrix as nested list
    intrinsic_matrix: list  # 3x3 matrix as nested list
    position: list  # [x, y, z]
    orientation_wxyz: list  # [w, x, y, z]


@dataclass
class ObjectConfig:
    """Object configuration from scene.json."""

    oid: int
    name: str
    mesh_path: str
    center: list  # [x, y, z]
    bbox_min: list  # [x, y, z]
    bbox_max: list  # [x, y, z]
    grasps: Optional[str] = None
    trajectory_path: Optional[str] = None


@dataclass
class SceneConfig:
    """Complete scene configuration."""

    background_mesh_path: str
    camera: CameraConfig
    objects: Dict[str, ObjectConfig]
    ground_plane_point: list  # [x, y, z]
    ground_plane_normal: list  # [x, y, z]
    scene_aabb_min: list
    scene_aabb_max: list


def load_scene_config(scene_json_path: str | Path) -> SceneConfig:
    """
    Load scene configuration from scene.json file.

    Args:
        scene_json_path: Path to scene.json file

    Returns:
        SceneConfig object containing all scene information

    Raises:
        FileNotFoundError: If scene.json doesn't exist
        ValueError: If scene.json is malformed
    """
    scene_json_path = Path(scene_json_path)

    if not scene_json_path.exists():
        raise FileNotFoundError(f"Scene JSON not found: {scene_json_path}")

    with open(scene_json_path, "r") as f:
        data = json.load(f)

    # Parse camera configuration
    cam_data = data.get("camera", {})
    intrinsic_matrix = np.array(
        [
            [cam_data["fx"], 0, cam_data["cx"]],
            [0, cam_data["fy"], cam_data["cy"]],
            [0, 0, 1],
        ]
    )
    camera = CameraConfig(
        width=int(cam_data["width"]),
        height=int(cam_data["height"]),
        fx=float(cam_data["fx"]),
        fy=float(cam_data["fy"]),
        cx=float(cam_data["cx"]),
        cy=float(cam_data["cy"]),
        extrinsic_matrix=cam_data["camera_opencv_to_world"],
        intrinsic_matrix=intrinsic_matrix.tolist(),
        position=cam_data["camera_position"],
        orientation_wxyz=cam_data["camera_heading_wxyz"],
    )

    # Parse objects
    objects = {}
    for obj_id, obj_data in data.get("objects", {}).items():
        # Use the optimized mesh if available, otherwise registered
        mesh_path = obj_data.get("optimized") or obj_data.get("registered")

        # get the output path:
        output_path = scene_json_path.parent.parent.parent.parent
        if mesh_path and mesh_path.startswith("/app/"):
            mesh_path = mesh_path.replace("/app/", str(output_path) + "/")

        grasp_path = obj_data.get("grasps").replace("/app/", str(output_path) + "/")
        objects[obj_id] = ObjectConfig(
            oid=obj_data["oid"],
            name=obj_data["name"],
            mesh_path=mesh_path,
            center=obj_data.get("object_center", [0, 0, 0]),
            bbox_min=obj_data.get("object_min", [0, 0, 0]),
            bbox_max=obj_data.get("object_max", [0, 0, 0]),
            grasps=grasp_path,
            trajectory_path=obj_data.get("hybrid_trajs")
            or obj_data.get("simple_trajs"),
        )

    # Parse background
    bg_data = data.get("background", {})
    bg_path = bg_data.get("registered") or bg_data.get("original")
    if bg_path and bg_path.startswith("/app/"):
        bg_path = bg_path.replace("/app/", str(output_path) + "/")

    # Parse ground plane (use simulation frame)
    ground_data = data.get("groundplane_in_sim", {})

    # Parse AABB
    aabb_data = data.get("aabb", {})

    return SceneConfig(
        background_mesh_path=bg_path,
        camera=camera,
        objects=objects,
        ground_plane_point=ground_data.get("point", [0, 0, 0]),
        ground_plane_normal=ground_data.get("normal", [0, 0, 1]),
        scene_aabb_min=aabb_data.get("scene_min", [-1, -1, -1]),
        scene_aabb_max=aabb_data.get("scene_max", [1, 1, 1]),
    )


def resolve_path(path: str, base_dir: Optional[Path] = None) -> Path:
    """
    Resolve a path from scene.json, handling container paths.

    Args:
        path: Path string from scene.json
        base_dir: Base directory to resolve relative paths (default: cwd)

    Returns:
        Resolved Path object
    """
    if base_dir is None:
        base_dir = Path.cwd()

    # Handle container paths
    if path.startswith("/app/"):
        path = path.replace("/app/", "")

    resolved = Path(path)

    # If not absolute, make it relative to base_dir
    if not resolved.is_absolute():
        resolved = base_dir / resolved

    return resolved
