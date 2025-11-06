#!/usr/bin/env python3

import argparse
from pathlib import Path
import tempfile
import pymeshlab
import trimesh


def _simplify_one(
    in_path: Path,
    target_tris: int,
    *,
    min_tris_to_simplify: int = 10_000,
    min_target_tris: int = 500,
    smooth_iters: int = 0,
) -> None:
    """Simplify one GLB/GLTF mesh in-place."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir = Path(tmpdir)
        tmp_obj = tmp_dir / 'mesh.obj'

        # Load GLB and convert to OBJ
        scene = trimesh.load(str(in_path), force='scene')
        scene.export(str(tmp_obj), file_type='obj', include_texture=True)

        # Load into pymeshlab
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(tmp_obj))
        n_tris = ms.current_mesh().face_number()

        if n_tris < min_tris_to_simplify:
            print(f"Skip {in_path.name}: {n_tris} tris")
            return

        tgt = max(min_target_tris, min(int(target_tris), n_tris - 1))
        if tgt >= n_tris:
            return

        print(f"Simplify {in_path.name}: {n_tris} -> {tgt} tris")

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
        simplified_scene = trimesh.load(str(tmp_obj), force='scene')
        simplified_scene.export(str(in_path))


def simplify_dir(
    root: str,
    target_tris: int,
    *,
    exts: str = ".glb,.gltf",
    recursive: bool = True,
    min_tris_to_simplify: int = 10_000,
    min_target_tris: int = 500,
    smooth_iters: int = 0,
) -> None:
    """Simplify all GLB/GLTF files in a directory in-place."""
    root_p = Path(root)
    suffixes = {e.strip().lower() for e in exts.split(",") if e.strip()}
    it = root_p.rglob("*") if recursive else root_p.glob("*")

    for p in it:
        if not p.is_file():
            continue
        if p.suffix.lower() not in suffixes:
            continue
        _simplify_one(
            p,
            target_tris,
            min_tris_to_simplify=min_tris_to_simplify,
            min_target_tris=min_target_tris,
            smooth_iters=smooth_iters,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simplify GLB/GLTF files in-place")
    parser.add_argument("root", help="Directory containing GLB/GLTF files")
    parser.add_argument("target_tris", type=int, help="Target number of triangles")
    parser.add_argument("--exts", default=".glb,.gltf", help="File extensions (default: .glb,.gltf)")
    parser.add_argument("--no-recursive", action="store_true", help="Don't search recursively")
    parser.add_argument("--min-tris-to-simplify", type=int, default=10_000, help="Skip files with fewer triangles")
    parser.add_argument("--min-target-tris", type=int, default=500, help="Minimum target triangles")
    parser.add_argument("--smooth-iters", type=int, default=0, help="Smoothing iterations")

    args = parser.parse_args()

    simplify_dir(
        args.root,
        args.target_tris,
        exts=args.exts,
        recursive=not args.no_recursive,
        min_tris_to_simplify=args.min_tris_to_simplify,
        min_target_tris=args.min_target_tris,
        smooth_iters=args.smooth_iters,
    )
