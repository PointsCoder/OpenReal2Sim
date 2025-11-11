#!/usr/bin/env python3
"""Automated mesh simplification for entire scene."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional
import tyro
from loguru import logger

# Add parent directory to path
TOOLS_DIR = Path(__file__).resolve().parent
MUJOCO_DIR = TOOLS_DIR.parent
REPO_ROOT = MUJOCO_DIR.parent.parent.parent

if str(MUJOCO_DIR) not in sys.path:
    sys.path.insert(0, str(MUJOCO_DIR))

from utils.mesh_simplification import simplify_glb_in_place


def map_docker_path_to_local(path_str: str, repo_root: Path) -> Path:
    """Map docker paths (/app/...) to local filesystem."""
    if path_str.startswith("/app/"):
        rel_path = path_str[len("/app/"):]
        return repo_root / rel_path
    else:
        return Path(path_str)


def main(
    scene_name: str,
    outputs_root: Path = Path("outputs"),
    target_tris: int = 50000,
    background_target_tris: int = 200000,
    min_tris_to_simplify: int = 10000,
    min_target_tris: int = 500,
    smooth_iters: int = 0,
) -> int:
    """Simplify all meshes in a scene. WARNING: Overwrites GLB files in place.

    Args:
        scene_name: Scene name to process
        outputs_root: Root directory containing outputs
        target_tris: Target number of triangles for objects
        background_target_tris: Target number of triangles for background
        min_tris_to_simplify: Skip files with fewer triangles
        min_target_tris: Minimum target triangles
        smooth_iters: Number of smoothing iterations
    """
    outputs_root = outputs_root.expanduser().resolve()
    scene_dir = outputs_root / scene_name / "simulation"

    if not scene_dir.exists():
        logger.error(f"Scene directory not found: {scene_dir}")
        return 1

    # Load scene.json to get GLB files
    scene_json_path = scene_dir / "scene.json"
    if not scene_json_path.exists():
        logger.error(f"scene.json not found: {scene_json_path}")
        return 1

    with open(scene_json_path, "r") as f:
        scene_config = json.load(f)

    # Collect GLB files from scene.json
    background_files = []
    object_files = []

    # Add background
    if "background" in scene_config and "registered" in scene_config["background"]:
        bg_path_str = scene_config["background"]["registered"]
        bg_path = map_docker_path_to_local(bg_path_str, REPO_ROOT)
        if not bg_path.is_absolute():
            bg_path = scene_dir / bg_path.name
        if bg_path.exists():
            background_files.append(bg_path)
        else:
            logger.warning(f"Background GLB not found: {bg_path}")

    # Add objects
    for obj_id, obj_cfg in scene_config["objects"].items():
        obj_name = obj_cfg["name"]
        if "optimized" in obj_cfg:
            obj_path_str = obj_cfg["optimized"]
            obj_path = map_docker_path_to_local(obj_path_str, REPO_ROOT)
            if not obj_path.is_absolute():
                obj_path = scene_dir / obj_path.name
            if obj_path.exists():
                object_files.append(obj_path)
            else:
                logger.warning(f"Object GLB not found for '{obj_name}': {obj_path}")

    glb_files = background_files + object_files

    if not glb_files:
        logger.error(f"No GLB files found in scene.json")
        return 1

    logger.warning("WARNING: This will overwrite GLB files in place! Make sure you have backups.")

    logger.info(f"Found {len(glb_files)} GLB files to simplify:")
    for glb in background_files:
        logger.info(f"  [BACKGROUND] {glb.name} (target_tris={background_target_tris})")
    for glb in object_files:
        logger.info(f"  [OBJECT]     {glb.name} (target_tris={target_tris})")

    logger.info("")
    response = input("Proceed with simplification? (yes/no): ")
    if response.lower() not in ["yes", "y"]:
        logger.info("Aborted by user")
        return 0

    # Simplify objects
    for glb_path in object_files:
        logger.info(f"Simplifying object: {glb_path.name}")
        try:
            simplify_glb_in_place(
                glb_path=glb_path,
                target_tris=target_tris,
                min_tris_to_simplify=min_tris_to_simplify,
                min_target_tris=min_target_tris,
                smooth_iters=smooth_iters,
            )
            logger.success(f"  [OK] {glb_path.name}")
        except Exception as e:
            logger.error(f"  [FAIL] Failed to simplify {glb_path.name}: {e}")
            return 1

    # Simplify background
    for glb_path in background_files:
        logger.info(f"Simplifying background: {glb_path.name}")
        try:
            simplify_glb_in_place(
                glb_path=glb_path,
                target_tris=background_target_tris,
                min_tris_to_simplify=min_tris_to_simplify,
                min_target_tris=min_target_tris,
                smooth_iters=smooth_iters,
            )
            logger.success(f"  [OK] {glb_path.name}")
        except Exception as e:
            logger.error(f"  [FAIL] Failed to simplify {glb_path.name}: {e}")
            return 1

    logger.success(f"\nCompleted simplification of {len(glb_files)} files")

    return 0


if __name__ == "__main__":
    raise SystemExit(tyro.cli(main))
