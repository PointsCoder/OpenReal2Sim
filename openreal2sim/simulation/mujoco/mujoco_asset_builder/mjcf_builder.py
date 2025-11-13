from pathlib import Path
from typing import Any, List, Union

import mujoco
import numpy as np
import trimesh
from lxml import etree
from loguru import logger

from .material import Material


class MJCFBuilder:
    """Build a MuJoCo XML model from meshes and materials."""

    def __init__(
        self,
        filename: Path,
        mesh: Union[trimesh.base.Trimesh, Any],
        materials: List[Material],
        work_dir: Path = Path(),
        decomp_success: bool = False,
    ):
        self.filename = filename
        self.mesh = mesh
        self.materials = materials
        self.decomp_success = decomp_success

        self.work_dir = work_dir if work_dir != Path() else filename.parent
        self.tree = None

    def add_visual_and_collision_default_classes(self, root: etree.Element):
        default_elem = etree.SubElement(root, "default")
        visual_default_elem = etree.SubElement(default_elem, "default")
        visual_default_elem.attrib["class"] = "visual"
        etree.SubElement(
            visual_default_elem,
            "geom",
            group="2",
            type="mesh",
            contype="0",
            conaffinity="0",
        )

        collision_default_elem = etree.SubElement(default_elem, "default")
        collision_default_elem.attrib["class"] = "collision"
        etree.SubElement(
            collision_default_elem,
            "geom",
            group="3",
            type="mesh",
            margin="0.006",
            solref="0.0005 1",
            solimp="0.998 0.9999 0.0005",
        )

    def add_assets(self, root: etree.Element, mtls: List[Material]) -> etree.Element:
        asset_elem = etree.SubElement(root, "asset")

        for material in mtls:
            if material.map_Kd is not None:
                texture = Path(material.map_Kd)
                etree.SubElement(
                    asset_elem,
                    "texture",
                    type="2d",
                    name=texture.stem,
                    file=texture.name,
                )
                etree.SubElement(
                    asset_elem,
                    "material",
                    name=material.name,
                    texture=texture.stem,
                    specular=material.mjcf_specular(),
                    shininess=material.mjcf_shininess(),
                )
            else:
                etree.SubElement(
                    asset_elem,
                    "material",
                    name=material.name,
                    specular=material.mjcf_specular(),
                    shininess=material.mjcf_shininess(),
                    rgba=material.mjcf_rgba(),
                )

        return asset_elem

    def add_visual_geometries(self, obj_body: etree.Element, asset_elem: etree.Element):
        mesh = self.mesh
        materials = self.materials
        filename = self.filename
        process_mtl = len(materials) > 0

        if isinstance(mesh, trimesh.base.Trimesh):
            meshname = Path(f"{filename.stem}.obj")
            etree.SubElement(asset_elem, "mesh", file=meshname.as_posix())
            geom_attrs = {"class": "visual", "mesh": meshname.stem}
            if process_mtl:
                geom_attrs["material"] = materials[0].name
            etree.SubElement(obj_body, "geom", **geom_attrs)
        else:
            for i, (name, geom) in enumerate(mesh.geometry.items()):
                meshname = Path(f"{filename.stem}_{i}.obj")
                etree.SubElement(asset_elem, "mesh", file=meshname.as_posix())
                geom_attrs = {"class": "visual", "mesh": meshname.stem}
                if process_mtl:
                    geom_attrs["material"] = name
                etree.SubElement(obj_body, "geom", **geom_attrs)

    def add_collision_geometries(self, obj_body: etree.Element, asset_elem: etree.Element):
        mesh = self.mesh
        work_dir = self.work_dir
        filename = self.filename

        collision_files = sorted(
            [x for x in work_dir.glob(f"{filename.stem}_collision_*.obj") if x.is_file()],
            key=lambda p: int(p.stem.split("_")[-1]),
        )

        if not collision_files:
            raise RuntimeError(
                f"No collision meshes found for {self.filename.stem}."
                " Run the convex decomposition step before building MJCFs."
            )

        for collision in collision_files:
            etree.SubElement(asset_elem, "mesh", file=collision.name)
            rgb = np.random.rand(3)
            etree.SubElement(
                obj_body,
                "geom",
                mesh=collision.stem,
                rgba=f"{rgb[0]} {rgb[1]} {rgb[2]} 1",
                **{"class": "collision"},
            )

    def build(self, add_free_joint: bool = False):
        filename = self.filename
        mtls = self.materials

        root = etree.Element("mujoco", model=filename.stem)
        self.add_visual_and_collision_default_classes(root)
        asset_elem = self.add_assets(root, mtls)

        worldbody_elem = etree.SubElement(root, "worldbody")
        obj_body = etree.SubElement(worldbody_elem, "body", name=filename.stem)
        if add_free_joint:
            etree.SubElement(
                obj_body,
                "joint",
                type="free",
                damping="0.1",
            )

        self.add_visual_geometries(obj_body, asset_elem)
        self.add_collision_geometries(obj_body, asset_elem)

        tree = etree.ElementTree(root)
        etree.indent(tree, space="  ", level=0)
        self.tree = tree

    def compile_model(self):
        if self.tree is None:
            raise ValueError("Tree has not been defined yet.")
        work_dir = self.work_dir
        tmp_path = work_dir / "tmp.xml"
        try:
            self.tree.write(tmp_path, encoding="utf-8")
            model = mujoco.MjModel.from_xml_path(tmp_path.as_posix())
            data = mujoco.MjData(model)
            mujoco.mj_step(model, data)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def save_mjcf(self):
        if self.tree is None:
            raise ValueError("Tree has not been defined yet.")
        xml_path = self.work_dir / f"{self.filename.stem}.xml"
        self.tree.write(xml_path.as_posix(), encoding="utf-8")
        logger.info(f"Saved MJCF to {xml_path}")
