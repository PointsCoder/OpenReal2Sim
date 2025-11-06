#!/usr/bin/env python3

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable, List
from loguru import logger

CURRENT_DIR = Path(__file__).resolve().parent
MUJOCO_DIR = CURRENT_DIR.parent
if str(MUJOCO_DIR) not in sys.path:
    sys.path.insert(0, str(MUJOCO_DIR))

import trimesh
from trimesh.exchange.export import export_mesh

from mujoco_asset_builder import constants
from mujoco_asset_builder.processing import (
    CoacdParams,
    ProcessingConfig,
    process_obj_inplace,
)


def _collect_meshes(scene: trimesh.Scene, *, apply_units: bool, target_units: str | None) -> list[trimesh.Trimesh]:
    meshes: list[trimesh.Trimesh] = []

    for node_name in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node_name]
        geometry = scene.geometry[geom_name]

        mesh = geometry.copy()
        mesh.apply_transform(transform)

        if apply_units and target_units is not None:
            if mesh.units is None and scene.units is not None:
                mesh.units = scene.units
            if mesh.units is not None:
                mesh.convert_units(target_units, guess=False)

        _ = mesh.vertex_normals
        mesh.update_faces(mesh.unique_faces())
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
        mesh.remove_infinite_values()
        mesh.fix_normals()

        meshes.append(mesh)

    return meshes


def export_visual_assets(glb_path: Path, out_dir: Path, target_units: str | None):
    scene = trimesh.load(glb_path, force="scene", skip_materials=False, process=False)
    if isinstance(scene, trimesh.Trimesh):
        scene = trimesh.Scene(scene)

    meshes = _collect_meshes(scene, apply_units=target_units is not None, target_units=target_units)
    if not meshes:
        raise RuntimeError(f"No geometry nodes found in: {glb_path}")

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

    if not obj_path.exists():
        raise RuntimeError(f"OBJ export failed for: {glb_path}")

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

    if mtl_path is None or not mtl_path.exists():
        raise RuntimeError(f"MTL export failed for: {glb_path}")

    return obj_path, mtl_path, meshes


def compute_source_stats(meshes: Iterable[trimesh.Trimesh]) -> dict:
    verts = sum(len(m.vertices) for m in meshes)
    faces = sum(len(m.faces) for m in meshes)
    return {"source_vertices": verts, "source_faces": faces}


def compute_collision_stats(out_dir: Path, stem: str) -> dict:
    collision_files = sorted(
        out_dir.glob(f"{stem}{constants.SUFFIX_COLLISION}*{constants.EXT_OBJ}"),
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
    target_units: str | None,
    cfg: ProcessingConfig,
    asset_type: str,
) -> dict:
    out_dir = output_root / glb_path.stem if output_root else glb_path.parent / glb_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob(f"{glb_path.stem}{constants.SUFFIX_COLLISION}*"):
        if stale.is_file():
            stale.unlink()

    obj_path, mtl_path, meshes = export_visual_assets(glb_path, out_dir, target_units)
    source_stats = compute_source_stats(meshes)

    xml_path = process_obj_inplace(obj_path, cfg)

    metadata_path = out_dir / f"{glb_path.stem}{constants.SUFFIX_METADATA}"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert GLB/GLTF assets into MuJoCo-ready MJCF packages.")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input GLB/GLTF files or directories. The last positional value may be an output directory when --output-dir is omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where results are written. Each asset is placed under OUTPUT_DIR/<name>/...",
    )
    parser.add_argument(
        "--target-units",
        type=str,
        default=None,
        help="Optional target unit (e.g. 'm', 'cm').",
    )
    parser.add_argument(
        "--texture-resize",
        type=float,
        default=1.0,
        help="Resize textures by this factor (default 1.0).",
    )
    parser.add_argument(
        "--coacd-threshold",
        type=float,
        default=0.05,
        help="CoACD concavity threshold for object assets.",
    )
    parser.add_argument(
        "--coacd-max-hulls",
        type=int,
        default=64,
        help="Maximum convex hulls for object assets.",
    )
    parser.add_argument(
        "--coacd-resolution",
        type=int,
        default=4000,
        help="Sampling resolution for object assets.",
    )
    parser.add_argument(
        "--background-max-hulls",
        type=int,
        default=512,
        help="Maximum convex hulls for background assets.",
    )
    parser.add_argument(
        "--background-threshold",
        type=float,
        default=0.02,
        help="Concavity threshold for background assets.",
    )
    parser.add_argument(
        "--background-resolution",
        type=int,
        default=12000,
        help="Sampling resolution for background assets.",
    )
    parser.add_argument("--add-free-joint", action="store_true", help="Add a freejoint to the root body.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def expand_inputs(inputs: List[str]) -> List[Path]:
    glb_files: List[Path] = []
    for item in inputs:
        path = Path(item).expanduser()
        if path.is_dir():
            glb_files.extend(sorted(path.rglob(f"*{constants.EXT_GLB}")))
            glb_files.extend(sorted(path.rglob(f"*{constants.EXT_GLTF}")))
        elif path.is_file():
            glb_files.append(path)
        else:
            raise FileNotFoundError(f"Input path not found: {path}")
    unique = []
    seen = set()
    for path in glb_files:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def build_processing_configs(args: argparse.Namespace):
    object_cfg = ProcessingConfig(
        texture_resize_percent=args.texture_resize,
        add_free_joint=args.add_free_joint,
        decompose=True,
        coacd=CoacdParams(
            threshold=args.coacd_threshold,
            max_convex_hull=args.coacd_max_hulls,
            resolution=args.coacd_resolution,
        ),
    )

    background_cfg = ProcessingConfig(
        texture_resize_percent=args.texture_resize,
        add_free_joint=args.add_free_joint,
        decompose=True,
        coacd=CoacdParams(
            threshold=args.background_threshold,
            max_convex_hull=args.background_max_hulls,
            resolution=args.background_resolution,
            mcts_iterations=200,
            mcts_nodes=30,
        ),
    )

    return object_cfg, background_cfg


def main() -> int:
    args = parse_args()
    try:
        import coacd  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "The 'coacd' package is required. Install it with `pip install coacd`."
        ) from exc

    raw_inputs = [Path(p).expanduser() for p in args.inputs]
    output_root = args.output_dir.expanduser().resolve() if args.output_dir is not None else None

    if output_root is None and len(raw_inputs) > 1:
        candidate = raw_inputs[-1]
        if candidate.suffix.lower() not in {constants.EXT_GLB, constants.EXT_GLTF}:
            output_root = candidate
            raw_inputs = raw_inputs[:-1]

    if output_root is not None:
        output_root = output_root.resolve()
        output_root.mkdir(parents=True, exist_ok=True)

    try:
        inputs = expand_inputs([str(p) for p in raw_inputs])
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if not inputs:
        print("No GLB/GLTF files found.", file=sys.stderr)
        return 1

    object_cfg, background_cfg = build_processing_configs(args)
    summaries = []
    status = 0

    for glb_path in inputs:
        start = time.perf_counter()
        try:
            asset_type = constants.ASSET_TYPE_BACKGROUND if constants.ASSET_TYPE_BACKGROUND in glb_path.stem.lower() else constants.ASSET_TYPE_OBJECT
            cfg = background_cfg if asset_type == constants.ASSET_TYPE_BACKGROUND else object_cfg
            metadata = convert_glb(glb_path, output_root or glb_path.parent, args.target_units, cfg, asset_type)
            elapsed = time.perf_counter() - start
            summaries.append((glb_path, elapsed, metadata))
            print(
                f"[OK] {glb_path} -> {metadata['xml']} "
                f"(visual: {metadata['visual_obj']}, collisions: {metadata['collision_parts']} parts) "
                f"in {elapsed:.1f}s"
            )
        except Exception as exc:
            status = 1
            elapsed = time.perf_counter() - start
            print(f"[FAIL] {glb_path} after {elapsed:.1f}s: {exc}", file=sys.stderr)

    if summaries:
        print("\nProcessing times:")
        for path, elapsed, metadata in summaries:
            print(f"  {path.stem:>20}: {elapsed:.1f}s, {metadata['collision_parts']} collision parts")

    return status


if __name__ == "__main__":
    sys.exit(main())
