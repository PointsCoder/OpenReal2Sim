#!/usr/bin/env python3
"""Automated mesh simplification for entire scene."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional
import tyro
from loguru import logger

# Add parent directory to path
TOOLS_DIR = Path(__file__).resolve().parent
MUJOCO_DIR = TOOLS_DIR.parent

if str(MUJOCO_DIR) not in sys.path:
    sys.path.insert(0, str(MUJOCO_DIR))

from utils.mesh_simplification import simplify_glb_in_place


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

    # Collect all GLB files (skip scene_optimized.glb)
    glb_files = sorted([
        f for f in scene_dir.glob("*.glb")
        if f.name != "scene_optimized.glb"
    ])

    if not glb_files:
        logger.error(f"No GLB files found in {scene_dir}")
        return 1

    logger.warning("WARNING: This will overwrite GLB files in place! Make sure you have backups.")

    logger.info(f"Found {len(glb_files)} GLB files to simplify:")
    background_files = []
    object_files = []

    for glb in glb_files:
        if "background" in glb.stem.lower():
            background_files.append(glb)
            logger.info(f"  [BACKGROUND] {glb.name} (target_tris={background_target_tris})")
        else:
            object_files.append(glb)
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
