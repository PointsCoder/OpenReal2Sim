import json
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional
from loguru import logger
import tyro

CURRENT_DIR = Path(__file__).resolve().parent
MUJOCO_DIR = CURRENT_DIR.parent
if str(MUJOCO_DIR) not in sys.path:
    sys.path.insert(0, str(MUJOCO_DIR))

import trimesh
from trimesh.exchange.export import export_mesh

from mujoco_asset_builder.processing import (
    CoacdParams,
    ProcessingConfig,
    process_obj_inplace,
)


def _collect_meshes(scene: trimesh.Scene) -> list[trimesh.Trimesh]:
    meshes: list[trimesh.Trimesh] = []

    for node_name in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node_name]
        geometry = scene.geometry[geom_name]

        mesh = geometry.copy()
        mesh.apply_transform(transform)

        _ = mesh.vertex_normals
        mesh.update_faces(mesh.unique_faces())
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
        mesh.remove_infinite_values()
        mesh.fix_normals()

        meshes.append(mesh)

    return meshes


def export_visual_assets(glb_path: Path, out_dir: Path):
    scene = trimesh.load(glb_path, force="scene", skip_materials=False, process=False)
    if isinstance(scene, trimesh.Trimesh):
        scene = trimesh.Scene(scene)

    meshes = _collect_meshes(scene)
    if len(meshes) == 1:
        payload: trimesh.Trimesh | trimesh.Scene = meshes[0]
    else:
        payload = trimesh.Scene()
        for idx, mesh in enumerate(meshes):
            payload.add_geometry(mesh, node_name=f"mesh_{idx}")

    obj_path = out_dir / f"{glb_path.stem}.obj"
    export_mesh(
        payload,
        obj_path,
        file_type="obj",
        include_normals=True,
        include_texture=True,
        write_texture=True,
    )

    mtl_path = None
    with obj_path.open("r", encoding="utf-8", errors="ignore") as obj_file:
        for line in obj_file:
            if line.lower().startswith("mtllib"):
                parts = line.strip().split()
                if len(parts) >= 2:
                    candidate = out_dir / parts[1]
                    if candidate.exists():
                        mtl_path = candidate
                        break
    if mtl_path is None:
        candidates = sorted(out_dir.glob("*.mtl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            mtl_path = candidates[0]

    return obj_path, mtl_path, meshes


def compute_source_stats(meshes: Iterable[trimesh.Trimesh]) -> dict:
    verts = sum(len(m.vertices) for m in meshes)
    faces = sum(len(m.faces) for m in meshes)
    return {"source_vertices": verts, "source_faces": faces}


def compute_collision_stats(out_dir: Path, stem: str) -> dict:
    collision_files = sorted(
        out_dir.glob(f"{stem}_collision_*.obj"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    stats = {
        "collision_parts": len(collision_files),
        "collision_vertices": 0,
        "collision_faces": 0,
        "collision_files": [path.name for path in collision_files],
    }
    for path in collision_files:
        mesh = trimesh.load(path, force="mesh")
        stats["collision_vertices"] += len(mesh.vertices)
        stats["collision_faces"] += len(mesh.faces)
    return stats


def convert_glb(
    glb_path: Path,
    output_root: Path,
    cfg: ProcessingConfig,
    asset_type: str,
) -> dict:
    out_dir = output_root / glb_path.stem if output_root else glb_path.parent / glb_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob(f"{glb_path.stem}_collision_*"):
        if stale.is_file():
            stale.unlink()

    obj_path, mtl_path, meshes = export_visual_assets(glb_path, out_dir)
    source_stats = compute_source_stats(meshes)

    xml_path = process_obj_inplace(obj_path, cfg)

    metadata_path = out_dir / f"{glb_path.stem}_metadata.json"
    if metadata_path.exists():
        metadata_path.unlink()

    collision_stats = compute_collision_stats(out_dir, glb_path.stem)
    metadata = {
        "asset": glb_path.stem,
        "asset_type": asset_type,
        "visual_obj": obj_path.name,
        "visual_mtl": mtl_path.name,
        **source_stats,
        **collision_stats,
        "xml": xml_path.name,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return metadata


def expand_inputs(inputs: List[str]) -> List[Path]:
    glb_files: List[Path] = []
    for item in inputs:
        path = Path(item).expanduser()
        if path.is_dir():
            glb_files.extend(sorted(path.rglob("*.glb")))
            glb_files.extend(sorted(path.rglob("*.gltf")))
        elif path.is_file():
            glb_files.append(path)
    unique = []
    seen = set()
    for path in glb_files:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def build_processing_configs(
    add_free_joint: bool,
    coacd_threshold: float,
    coacd_max_hulls: int,
    coacd_resolution: int,
    background_threshold: float,
    background_max_hulls: int,
    background_resolution: int,
):
    object_cfg = ProcessingConfig(
        add_free_joint=add_free_joint,
        decompose=True,
        coacd=CoacdParams(
            threshold=coacd_threshold,
            max_convex_hull=coacd_max_hulls,
            resolution=coacd_resolution,
        ),
    )

    background_cfg = ProcessingConfig(
        add_free_joint=add_free_joint,
        decompose=True,
        coacd=CoacdParams(
            threshold=background_threshold,
            max_convex_hull=background_max_hulls,
            resolution=background_resolution,
            mcts_iterations=200,
            mcts_nodes=30,
        ),
    )

    return object_cfg, background_cfg


def main(
    inputs: List[str] = [],
    scene_name: Optional[str] = None,
    outputs_root: Path = Path("outputs"),
    output_dir: Optional[Path] = None,
    coacd_threshold: float = 0.05,
    coacd_max_hulls: int = 64,
    coacd_resolution: int = 4000,
    background_max_hulls: int = 512,
    background_threshold: float = 0.02,
    background_resolution: int = 12000,
    add_free_joint: bool = False,
) -> int:
    """Convert GLB/GLTF assets to MJCF with CoACD convex decomposition.

    Args:
        inputs: Input GLB/GLTF files or directories
        scene_name: Process scene from outputs/<scene_name>/simulation/
        outputs_root: Root directory containing outputs
        output_dir: Directory where results are written
        coacd_threshold: CoACD concavity threshold for objects
        coacd_max_hulls: Maximum convex hulls for objects
        coacd_resolution: Sampling resolution for objects
        background_max_hulls: Maximum convex hulls for background
        background_threshold: Concavity threshold for background
        background_resolution: Sampling resolution for background
        add_free_joint: Add a freejoint to the root body
    """
    import coacd  # noqa: F401

    # Scene mode: process GLBs from outputs/<scene_name>/simulation/
    if scene_name:
        outputs_root = outputs_root.expanduser().resolve()
        scene_dir = outputs_root / scene_name / "simulation"
        scene_json_path = scene_dir / "scene.json"
        with open(scene_json_path, "r") as f:
            scene_config = json.load(f)

        # Collect GLB files
        raw_inputs = []
        background_files = []
        object_files = []

        # Add background
        if "background" in scene_config and "registered" in scene_config["background"]:
            bg_path_str = scene_config["background"]["registered"]
            if bg_path_str.startswith("/app/"):
                bg_path_str = bg_path_str.replace("/app/", "")
            bg_path = Path(bg_path_str)
            if not bg_path.is_absolute():
                bg_path = scene_dir / bg_path.name
            raw_inputs.append(bg_path)
            background_files.append(bg_path)

        # Add objects
        for obj_id, obj_cfg in scene_config["objects"].items():
            obj_name = obj_cfg["name"]
            glb_name = f"{obj_id}_{obj_name}_optimized.glb"
            glb_path = scene_dir / glb_name
            raw_inputs.append(glb_path)
            object_files.append((obj_name, glb_path))

        # Set output directory
        output_root = scene_dir / "mujoco" / "mjcf"
        output_root.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing scene '{scene_name}'")
        logger.info(f"  Input: {scene_dir}")
        logger.info(f"  Output: {output_root}")
        logger.info(f"Found {len(raw_inputs)} GLB files to convert:")
        if background_files:
            logger.info(f"  Background: {len(background_files)} file(s)")
            for bg_file in background_files:
                logger.info(f"    - {bg_file.name}")
        if object_files:
            logger.info(f"  Objects: {len(object_files)} file(s)")
            for obj_name, obj_file in object_files:
                logger.info(f"    - {obj_name} ({obj_file.name})")
    else:
        # Standard mode: use provided inputs
        raw_inputs = [Path(p).expanduser() for p in inputs]
        output_root = output_dir.expanduser().resolve() if output_dir is not None else None

        if output_root is None and len(raw_inputs) > 1:
            candidate = raw_inputs[-1]
            if candidate.suffix.lower() not in {".glb", ".gltf"}:
                output_root = candidate
                raw_inputs = raw_inputs[:-1]

        if output_root is not None:
            output_root = output_root.resolve()
            output_root.mkdir(parents=True, exist_ok=True)

    glb_files = expand_inputs([str(p) for p in raw_inputs])
    object_cfg, background_cfg = build_processing_configs(
        add_free_joint, coacd_threshold, coacd_max_hulls, coacd_resolution,
        background_threshold, background_max_hulls, background_resolution
    )
    summaries = []

    for glb_path in glb_files:
        start = time.perf_counter()
        asset_type = "background" if "background" in glb_path.stem.lower() else "object"
        cfg = background_cfg if asset_type == "background" else object_cfg
        logger.info(f"Converting {glb_path.name} ({asset_type})...")
        metadata = convert_glb(glb_path, output_root or glb_path.parent, cfg, asset_type)
        elapsed = time.perf_counter() - start
        summaries.append((glb_path, elapsed, metadata, asset_type))
        logger.success(
            f"  [OK] {metadata['xml']} "
            f"(visual: {metadata['visual_obj']}, collisions: {metadata['collision_parts']} parts) "
            f"in {elapsed:.1f}s"
        )

    logger.info(f"\nCompleted {len(summaries)} conversions")
    logger.info("Summary:")
    for path, elapsed, metadata, asset_type in summaries:
        logger.info(f"  {path.stem:>30} [{asset_type:>10}]: {elapsed:>5.1f}s, {metadata['collision_parts']} collision parts")

    return 0


if __name__ == "__main__":
    sys.exit(tyro.cli(main))
