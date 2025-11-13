from __future__ import annotations

import sys
from pathlib import Path

import tyro

# Add parent directory to path
TOOLS_DIR = Path(__file__).resolve().parent
MUJOCO_DIR = TOOLS_DIR.parent

if str(MUJOCO_DIR) not in sys.path:
    sys.path.insert(0, str(MUJOCO_DIR))

from utils.mesh_simplification import simplify_glb_by_target_tris


def main(
    input: Path,
    output: Path,
    target_tris: int,
) -> None:
    """Simplify a single GLB/GLTF file.

    Args:
        input: Input GLB/GLTF file
        output: Output GLB/GLTF file
        target_tris: Target number of triangles
    """
    simplify_glb_by_target_tris(input, output, target_tris)


if __name__ == "__main__":
    tyro.cli(main)
