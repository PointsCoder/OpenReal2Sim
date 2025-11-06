import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import trimesh
from PIL import Image
from loguru import logger

from . import constants
from .material import Material
from .mjcf_builder import MJCFBuilder


@dataclass
class CoacdParams:
    preprocess_resolution: int = 50
    threshold: float = 0.05
    max_convex_hull: int = -1
    mcts_iterations: int = 100
    mcts_max_depth: int = 3
    mcts_nodes: int = 20
    resolution: int = 2000
    pca: bool = False
    seed: int = 0


@dataclass
class ProcessingConfig:
    texture_resize_percent: float = 1.0
    add_free_joint: bool = False
    decompose: bool = True
    coacd: CoacdParams = field(default_factory=CoacdParams)
    overwrite: bool = True
    visual_only: bool = False


def resize_texture(filename: Path, resize_percent: float) -> None:
    if resize_percent == 1.0:
        return
    image = Image.open(filename)
    new_width = int(image.size[0] * resize_percent)
    new_height = int(image.size[1] * resize_percent)
    logger.info(f"Resizing {filename} to {new_width}x{new_height}")
    image = image.resize((new_width, new_height), Image.LANCZOS)
    image.save(filename)


def parse_mtl_name(lines: Iterable[str]) -> Optional[str]:
    mtl_regex = re.compile(r"^mtllib\s+(.+?\.mtl)(?:\s*#.*)?\s*\n?$")
    for line in lines:
        match = mtl_regex.match(line)
        if match is not None:
            return match.group(1)
    return None


def decompose_convex(obj_file: Path, work_dir: Path, params: CoacdParams) -> None:
    import coacd  # noqa: F401

    mesh = trimesh.load(obj_file.resolve(), force="mesh")
    coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)  # type: ignore
    parts = coacd.run_coacd(mesh=coacd_mesh, **params.__dict__)

    if not parts:
        raise RuntimeError(f"CoACD returned no parts for {obj_file}")

    for stale in work_dir.glob(f"{obj_file.stem}{constants.SUFFIX_COLLISION}*{constants.EXT_OBJ}"):
        stale.unlink()

    for i, (vs, fs) in enumerate(parts):
        submesh_name = work_dir / f"{obj_file.stem}{constants.SUFFIX_COLLISION}{i}{constants.EXT_OBJ}"
        trimesh.Trimesh(vs, fs).export(submesh_name.as_posix())


class VisualOnlyMJCFBuilder(MJCFBuilder):
    def __init__(self, *args, visual_only: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.visual_only = visual_only

    def add_collision_geometries(self, obj_body, asset_elem):
        if self.visual_only:
            return
        super().add_collision_geometries(obj_body, asset_elem)


def process_obj_inplace(obj_file: Path, cfg: ProcessingConfig) -> Path:
    if not obj_file.exists():
        raise FileNotFoundError(f"OBJ file not found: {obj_file}")

    work_dir = obj_file.parent
    logger.info(f"Processing {obj_file} in-place in {work_dir}")

    if cfg.decompose:
        decompose_convex(obj_file, work_dir, cfg.coacd)

    with obj_file.open("r") as f:
        mtl_name = parse_mtl_name(f.readlines())

    process_mtl = mtl_name is not None
    sub_mtls: List[List[str]] = []
    mtls: List[Material] = []

    if process_mtl:
        mtl_filename = obj_file.parent / mtl_name
        if not mtl_filename.exists():
            raise RuntimeError(f"MTL file {mtl_filename} referenced by {obj_file} is missing")

        with open(mtl_filename, "r") as f:
            lines = f.readlines()
        lines = [line for line in lines if not line.startswith(constants.MTL_COMMENT_CHAR)]
        lines = [line for line in lines if line.strip()]
        lines = [line.strip() for line in lines]

        for line in lines:
            if line.startswith("newmtl"):
                sub_mtls.append([])
            sub_mtls[-1].append(line)
        for sub_mtl in sub_mtls:
            mtls.append(Material.from_string(sub_mtl))

        for mtl in mtls:
            if mtl.map_Kd is None:
                continue
            texture_path = Path(mtl.map_Kd)
            texture_name = texture_path.name
            src_filename = obj_file.parent / texture_path
            if not src_filename.exists():
                raise RuntimeError(
                    f"Texture {src_filename} referenced by {mtl.name} does not exist"
                )
            dst_filename = work_dir / texture_name
            if dst_filename != src_filename:
                shutil.copy(src_filename, dst_filename)
            if texture_path.suffix.lower() in [constants.EXT_JPG, constants.EXT_JPEG]:
                image = Image.open(dst_filename)
                os.remove(dst_filename)
                dst_filename = (work_dir / texture_path.stem).with_suffix(constants.EXT_PNG)
                image.save(dst_filename)
                texture_name = dst_filename.name
                mtl.map_Kd = texture_name
            resize_texture(dst_filename, cfg.texture_resize_percent)

        with open(mtl_filename, "w") as f:
            for sub_mtl in sub_mtls:
                for line in sub_mtl:
                    if line.startswith("map_Kd"):
                        parts = line.split()
                        if len(parts) >= 2:
                            texture_path = Path(parts[1])
                            if texture_path.suffix.lower() in [constants.EXT_JPG, constants.EXT_JPEG]:
                                new_texture = texture_path.with_suffix(constants.EXT_PNG)
                                line = f"map_Kd {new_texture.name}"
                    f.write(line + "\n")

    mesh = trimesh.load(
        obj_file,
        split_object=True,
        group_material=True,
        process=False,
        maintain_order=False,
    )

    if isinstance(mesh, trimesh.base.Trimesh):
        mesh.export(obj_file.as_posix(), include_texture=True, header=None)
    else:
        obj_file.unlink()
        for i, geom in enumerate(mesh.geometry.values()):
            savename = work_dir / f"{obj_file.stem}_{i}{constants.EXT_OBJ}"
            geom.export(savename.as_posix(), include_texture=True, header=None)

    if isinstance(mesh, trimesh.base.Trimesh) and len(mtls) > 1:
        with open(obj_file, "r") as f:
            lines = f.readlines()
        mat_name = None
        for line in lines:
            if line.startswith("usemtl"):
                mat_name = line.split()[1]
                break
        if mat_name:
            for smtl in sub_mtls:
                if smtl[0].split()[1] == mat_name:
                    sub_mtls = [smtl]
                    mtls = [Material.from_string(smtl)]
                    break

    mtls = list({obj.name: obj for obj in mtls}.values())

    for file in [
        x
        for x in work_dir.glob("**/*")
        if x.is_file() and constants.MATERIAL_PREFIX in x.name and not x.name.endswith(constants.EXT_PNG)
    ]:
        file.unlink()

    builder = VisualOnlyMJCFBuilder(
        obj_file,
        mesh,
        mtls,
        work_dir=work_dir,
        decomp_success=cfg.decompose,
        visual_only=cfg.visual_only,
    )
    builder.build(add_free_joint=cfg.add_free_joint)
    builder.save_mjcf()

    return work_dir / f"{obj_file.stem}{constants.EXT_XML}"
