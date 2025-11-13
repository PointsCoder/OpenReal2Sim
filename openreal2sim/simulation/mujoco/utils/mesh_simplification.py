from __future__ import annotations

import tempfile
from pathlib import Path

import pymeshlab
import trimesh
from loguru import logger


def simplify_glb_in_place(
    glb_path: Path,
    target_tris: int,
    min_tris_to_simplify: int = 10000,
    min_target_tris: int = 500,
    smooth_iters: int = 0,
) -> None:
    """Simplify GLB mesh in-place using quadric edge collapse."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir = Path(tmpdir)
        tmp_obj = tmp_dir / "mesh.obj"

        # Load GLB and convert to OBJ
        scene = trimesh.load(str(glb_path), force="scene")
        scene.export(str(tmp_obj), file_type="obj", include_texture=True)

        # Load into pymeshlab
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(tmp_obj))
        n_tris = ms.current_mesh().face_number()

        if n_tris < min_tris_to_simplify:
            logger.debug(f"Skip {glb_path.name}: {n_tris} tris < {min_tris_to_simplify}")
            return

        # If target is already larger than current, skip
        if target_tris >= n_tris:
            logger.debug(f"Skip {glb_path.name}: target {target_tris} >= current {n_tris}")
            return

        tgt = max(min_target_tris, min(int(target_tris), n_tris - 1))

        if tgt >= n_tris:
            logger.debug(f"Skip {glb_path.name}: computed target {tgt} >= current {n_tris}")
            return

        logger.debug(f"Simplify {glb_path.name}: {n_tris} -> {tgt} tris")

        # Simplify
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=tgt,
            preservetopology=True,
            preserveboundary=True,
            preservenormal=True,
            optimalplacement=True,
            planarquadric=True,
        )

        if smooth_iters > 0:
            ms.apply_coord_taubin_smoothing(stepsmoothnum=smooth_iters)

        # Save and convert back
        ms.save_current_mesh(str(tmp_obj))
        simplified_scene = trimesh.load(str(tmp_obj), force="scene")
        simplified_scene.export(str(glb_path))


def simplify_glb_by_target_tris(
    input_path: Path,
    output_path: Path,
    target_tris: int,
) -> None:
    """Simplify GLB mesh to target triangle count."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir = Path(tmpdir)
        tmp_obj = tmp_dir / "mesh.obj"

        logger.info(f"Loading {input_path.name}...")
        scene = trimesh.load(str(input_path), force="scene")
        scene.export(str(tmp_obj), file_type="obj", include_texture=True)

        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(tmp_obj))
        n_tris = ms.current_mesh().face_number()

        logger.info(f"Original: {n_tris:,} triangles")

        tgt = min(int(target_tris), n_tris - 1)
        if tgt >= n_tris:
            logger.info(
                f"Target {target_tris:,} >= current {n_tris:,}, "
                "copying without simplification"
            )
            scene.export(str(output_path))
            return

        logger.info(f"Simplifying to {tgt:,} triangles...")
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=tgt,
            preservetopology=True,
            preserveboundary=True,
            preservenormal=True,
            optimalplacement=True,
            planarquadric=True,
        )

        ms.save_current_mesh(str(tmp_obj))
        simplified_scene = trimesh.load(str(tmp_obj), force="scene")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        simplified_scene.export(str(output_path))

        final_tris = ms.current_mesh().face_number()
        reduction = (1 - final_tris / n_tris) * 100
        logger.success(f"Saved to {output_path}")
        logger.info(f"Final: {final_tris:,} triangles ({reduction:.1f}% reduction)")
