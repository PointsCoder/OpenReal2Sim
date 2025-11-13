from __future__ import annotations

import json
import yaml
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict
import xml.etree.ElementTree as ET

from loguru import logger


@dataclass(slots=True)
class CollisionDefaults:
    """Contact parameters attached to runtime collision classes."""

    margin: str
    solref: str
    solimp: str


@dataclass(slots=True)
class ContactParameters:
    """Pair parameters between gripper pads and object meshes."""

    condim: int
    solref: str
    solimp: str
    friction: str


class SceneFusion:
    """Reproduces the runtime fusion logic used by the web demo in Python."""

    def __init__(
        self,
        *,
        asset_root: Path,
        masses: Dict[str, float],
        z_offset: float,
        inertia_scale: float,
        freejoint_damping: float,
        robot_asset_prefix: str,
        collision_defaults: CollisionDefaults,
        contact_params: ContactParameters,
        groundplane_height: float,
        timestep: str,
        memory: str,
        solver_iterations: int,
        solver_ls_iterations: int,
        solver_noslip_iterations: int,
        material_specular: str,
        material_shininess: str,
    ) -> None:
        self.asset_root = Path(asset_root)
        self.masses = masses
        self.z_offset = z_offset
        self.inertia_scale = inertia_scale
        self.freejoint_damping = freejoint_damping
        self.robot_asset_prefix = robot_asset_prefix
        self.collision_defaults = collision_defaults
        self.contact_params = contact_params
        self.groundplane_height = groundplane_height
        self.timestep = timestep
        self.memory = memory
        self.solver_iterations = solver_iterations
        self.solver_ls_iterations = solver_ls_iterations
        self.solver_noslip_iterations = solver_noslip_iterations
        self.material_specular = material_specular
        self.material_shininess = material_shininess

    def fuse(
        self,
        robot_scene_path: Path,
        robot_panda_path: Path,
        scene_config: dict,
        object_metadata: Dict[str, dict],
    ) -> ET.ElementTree:
        """Fuse robot template MJCFs with reconstructed objects."""

        scene_tree = ET.parse(robot_scene_path)
        panda_tree = ET.parse(robot_panda_path)

        scene_root = scene_tree.getroot()
        root = panda_tree.getroot()

        self._rewrite_robot_mesh_paths(
            root,
            robot_asset_root=robot_panda_path.parent,
            robot_asset_prefix=self.robot_asset_prefix,
        )
        self._configure_root(root)
        self._merge_sections(root, scene_root)

        asset_elem = root.find("asset")
        if asset_elem is None:
            asset_elem = ET.SubElement(root, "asset")
        self._merge_scene_assets(asset_elem, scene_root.find("asset"))

        worldbody = root.find("worldbody")
        if worldbody is None:
            worldbody = ET.SubElement(root, "worldbody")
        self._merge_scene_worldbody(worldbody, scene_root.find("worldbody"))

        self._append_background_assets(asset_elem, object_metadata)
        self._append_object_assets(asset_elem, scene_config, object_metadata)
        self._insert_background_geometries(worldbody, object_metadata)
        self._insert_object_bodies(worldbody, scene_config, object_metadata)
        self._apply_robot_pose(root, scene_config)
        self._update_groundplane_height(worldbody)
        self._clear_keyframes(root)
        self._ensure_contact_pairs(root, scene_config, object_metadata)

        ET.indent(panda_tree, space="  ")
        return panda_tree

    # ------------------------------------------------------------------
    # XML helpers
    # ------------------------------------------------------------------
    def _rewrite_robot_mesh_paths(
        self,
        root: ET.Element,
        *,
        robot_asset_root: Path,
        robot_asset_prefix: str | None,
    ) -> None:
        compiler = root.find("compiler")
        if compiler is None:
            compiler = ET.Element("compiler")
            root.insert(0, compiler)

        old_meshdir = compiler.get("meshdir")
        if old_meshdir:
            source_meshes = list(root.iter("mesh"))
            if robot_asset_prefix:
                prefix = PurePosixPath(robot_asset_prefix.strip("/")) / PurePosixPath(old_meshdir.strip("/"))

                def _rewrite_relative(value: str) -> str:
                    if value.startswith("../") or value.startswith("../../"):
                        return value
                    return f"{prefix.as_posix()}/{value}"

                for mesh in source_meshes:
                    file_attr = mesh.get("file")
                    if file_attr:
                        mesh.set("file", _rewrite_relative(file_attr))
            else:
                base_dir = (robot_asset_root / old_meshdir).resolve()

                def _rewrite_absolute(value: str) -> str:
                    if value.startswith("/"):
                        return value
                    return str((base_dir / value).resolve())

                for mesh in source_meshes:
                    file_attr = mesh.get("file")
                    if file_attr:
                        mesh.set("file", _rewrite_absolute(file_attr))

        compiler.set("meshdir", ".")
        compiler.set("texturedir", ".")

    def _configure_root(self, root: ET.Element) -> None:
        option = root.find("option")
        if option is None:
            option = ET.SubElement(root, "option")
        option.set("timestep", self.timestep)
        option.set("iterations", str(self.solver_iterations))
        option.set("ls_iterations", str(self.solver_ls_iterations))
        option.set("noslip_iterations", str(self.solver_noslip_iterations))

        size_elem = root.find("size")
        if size_elem is None:
            size_elem = ET.SubElement(root, "size")
        size_elem.set("memory", self.memory)

        default_elem = root.find("default")
        if default_elem is None:
            compiler = root.find("compiler")
            default_elem = ET.Element("default")
            if compiler is not None:
                root.insert(list(root).index(compiler) + 1, default_elem)
            else:
                root.insert(0, default_elem)

        self._ensure_default_class(
            default_elem,
            class_name="obj_visual",
            geom_kwargs={
                "group": "2",
                "type": "mesh",
                "contype": "0",
                "conaffinity": "0",
            },
        )

        self._ensure_default_class(
            default_elem,
            class_name="obj_collision",
            geom_kwargs={
                "group": "3",
                "type": "mesh",
                "margin": self.collision_defaults.margin,
                "solref": self.collision_defaults.solref,
                "solimp": self.collision_defaults.solimp,
            },
        )

    def _merge_sections(self, root: ET.Element, scene_root: ET.Element) -> None:
        asset_elem = root.find("asset")
        insert_idx = list(root).index(asset_elem) if asset_elem is not None else len(root)

        for tag in ("statistic", "visual"):
            scene_elem = scene_root.find(tag)
            if scene_elem is None:
                continue
            existing = root.find(tag)
            if existing is not None:
                root.remove(existing)
            root.insert(insert_idx, scene_elem)
            insert_idx += 1

    def _merge_scene_assets(self, asset_elem: ET.Element, scene_asset: ET.Element | None) -> None:
        if scene_asset is None:
            return
        for child in scene_asset:
            asset_elem.append(child)

    def _merge_scene_worldbody(self, worldbody: ET.Element, scene_worldbody: ET.Element | None) -> None:
        if scene_worldbody is None:
            return
        for child in list(scene_worldbody):
            worldbody.insert(0, child)

    def _append_background_assets(self, asset_elem: ET.Element, metadata: Dict[str, dict]) -> None:
        bg_meta = metadata.get("background")
        if not bg_meta:
            return

        bg_name = f"background_registered"
        texture_name = f"{bg_name}_material_0"

        ET.SubElement(
            asset_elem,
            "texture",
            {
                "type": "2d",
                "name": texture_name,
                "file": self._asset_path(bg_name, f"material_0.png"),
            },
        )
        ET.SubElement(
            asset_elem,
            "material",
            {
                "name": texture_name,
                "texture": texture_name,
                "specular": "0.4",
                "shininess": "0.001",
            },
        )
        ET.SubElement(
            asset_elem,
            "mesh",
            {"file": self._asset_path(bg_name, f"{bg_name}.obj")},
        )

        for idx in range(bg_meta.get("collision_parts", 0)):
            ET.SubElement(
                asset_elem,
                "mesh",
                {
                    "file": self._asset_path(
                        bg_name,
                        f"{bg_name}_collision_{idx}.obj",
                    )
                },
            )

    def _append_object_assets(
        self,
        asset_elem: ET.Element,
        scene_config: dict,
        metadata: Dict[str, dict],
    ) -> None:
        objects = scene_config.get("objects", {})
        for oid in self._sorted_object_ids(objects):
            obj_cfg = objects[oid]
            obj_name = obj_cfg["name"]
            meta = metadata[oid]
            body_name = f"{oid}_{obj_name}_optimized"
            material_name = f"{obj_name}_material_0"

            ET.SubElement(
                asset_elem,
                "texture",
                {
                    "type": "2d",
                    "name": material_name,
                    "file": self._asset_path(body_name, "material_0.png"),
                },
            )
            ET.SubElement(
                asset_elem,
                "material",
                {
                    "name": material_name,
                    "texture": material_name,
                    "specular": self.material_specular,
                    "shininess": self.material_shininess,
                },
            )
            ET.SubElement(
                asset_elem,
                "mesh",
                {"file": self._asset_path(body_name, f"{body_name}.obj")},
            )

            collision_parts = meta.get("collision_parts")
            if collision_parts is None:
                raise KeyError(f"Missing 'collision_parts' in metadata for object '{oid}'")
            for idx in range(collision_parts):
                ET.SubElement(
                    asset_elem,
                    "mesh",
                    {
                        "file": self._asset_path(
                            body_name,
                            f"{body_name}_collision_{idx}.obj",
                        )
                    },
                )

    def _insert_background_geometries(self, worldbody: ET.Element, metadata: Dict[str, dict]) -> None:
        bg_meta = metadata.get("background")
        if not bg_meta:
            return

        bg_name = "background_registered"
        material_name = f"{bg_name}_material_0"
        geoms: list[ET.Element] = []

        visual_geom = ET.Element(
            "geom",
            {
                "name": f"{bg_name}_visual",
                "class": "obj_visual",
                "mesh": bg_name,
                "material": material_name,
            },
        )
        geoms.append(visual_geom)

        for idx in range(bg_meta["collision_parts"]):
            geoms.append(
                ET.Element(
                    "geom",
                    {
                        "name": f"{bg_name}_collision_{idx}",
                        "mesh": f"{bg_name}_collision_{idx}",
                        "class": "obj_collision",
                    },
                )
            )

        bodies = list(worldbody)
        insertion_index = next((i for i, elem in enumerate(bodies) if elem.tag == "body"), len(bodies))
        for geom in geoms:
            worldbody.insert(insertion_index, geom)

    def _insert_object_bodies(
        self,
        worldbody: ET.Element,
        scene_config: dict,
        metadata: Dict[str, dict],
    ) -> None:
        objects = scene_config["objects"]
        for oid in self._sorted_object_ids(objects):
            obj_cfg = objects[oid]
            obj_name = obj_cfg["name"]
            meta = metadata[oid]
            mass = self.masses[obj_name]
            inertia = mass * self.inertia_scale
            diag_inertia = f"{inertia:.6f} {inertia:.6f} {inertia:.6f}"

            body_name = f"{oid}_{obj_name}_optimized"
            material_name = f"{obj_name}_material_0"
            obj_center = obj_cfg["object_center"]

            body_elem = ET.SubElement(
                worldbody,
                "body",
                {
                    "name": body_name,
                    "pos": f"0 0 {self.z_offset}",
                },
            )
            ET.SubElement(
                body_elem,
                "joint",
                {
                    "type": "free",
                    "damping": f"{self.freejoint_damping:.6f}",
                },
            )
            ET.SubElement(
                body_elem,
                "inertial",
                {
                    "pos": f"{obj_center[0]} {obj_center[1]} {obj_center[2]}",
                    "mass": f"{mass}",
                    "diaginertia": diag_inertia,
                },
            )
            ET.SubElement(
                body_elem,
                "geom",
                {
                    "class": "obj_visual",
                    "mesh": body_name,
                    "material": material_name,
                },
            )

            for idx in range(meta["collision_parts"]):
                ET.SubElement(
                    body_elem,
                    "geom",
                    {
                        "name": f"{body_name}_collision_{idx}",
                        "mesh": f"{body_name}_collision_{idx}",
                        "class": "obj_collision",
                    },
                )

    def _apply_robot_pose(self, root: ET.Element, scene_config: dict) -> None:
        robot_pose = scene_config["robot_cfg"]["robot_pose"]
        worldbody = root.find("worldbody")
        if worldbody is None:
            return
        link0 = worldbody.find(f"body[@name='link0']")
        if link0 is None:
            logger.warning("Could not locate robot base body 'link0'")
            return

        pos = robot_pose[:3]
        quat = robot_pose[3:]
        link0.set("pos", f"{pos[0]} {pos[1]} {pos[2]}")
        link0.set("quat", f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}")

    def _update_groundplane_height(self, worldbody: ET.Element) -> None:
        """Update ground plane z-position."""
        if self.groundplane_height == 0.0:
            return
        
        # Find ground plane geom (typically named "ground" or has type="plane")
        for geom in worldbody.findall("geom"):
            geom_type = geom.get("type")
            geom_name = geom.get("name", "")
            
            if geom_type == "plane" or "ground" in geom_name.lower():
                pos = geom.get("pos", "0 0 0")
                pos_parts = pos.split()
                if len(pos_parts) == 3:
                    pos_parts[2] = str(self.groundplane_height)
                    geom.set("pos", " ".join(pos_parts))
                    logger.info(f"Updated ground plane height to {self.groundplane_height}")
                    return

    def _clear_keyframes(self, root: ET.Element) -> None:
        keyframe = root.find("keyframe")
        if keyframe is None:
            return
        for key in list(keyframe):
            keyframe.remove(key)

    def _ensure_contact_pairs(self, root: ET.Element, scene_config: dict, metadata: Dict[str, dict]) -> None:
        objects = scene_config.get("objects", {})
        if not objects:
            return

        contact = root.find("contact")
        worldbody = root.find("worldbody")
        if worldbody is None:
            return
        if contact is None:
            contact = ET.Element("contact")
            root.insert(list(root).index(worldbody), contact)

        for oid, obj_cfg in objects.items():
            obj_name = obj_cfg["name"]
            meta = metadata[oid]
            for idx in range(meta["collision_parts"]):
                geom_name = f"{oid}_{obj_name}_optimized_collision_{idx}"
                for finger in ("finger1", "finger2"):
                    for pad_idx in range(1, 6):
                        ET.SubElement(
                            contact,
                            "pair",
                            {
                                "geom1": geom_name,
                                "geom2": f"{finger}_fingertip_pad_collision_{pad_idx}",
                                "condim": str(self.contact_params.condim),
                                "solref": self.contact_params.solref,
                                "solimp": self.contact_params.solimp,
                                "friction": self.contact_params.friction,
                            },
                        )

    def _ensure_default_class(self, default_elem: ET.Element, *, class_name: str, geom_kwargs: dict) -> None:
        for child in default_elem.findall("default"):
            if child.get("class") == class_name:
                return
        sub = ET.SubElement(default_elem, "default", {"class": class_name})
        ET.SubElement(sub, "geom", geom_kwargs)

    def _asset_path(self, *parts: str) -> str:
        """Return relative asset path in POSIX format."""
        relative = Path("mjcf", *parts)
        return PurePosixPath(relative).as_posix()

    def _sorted_object_ids(self, objects: dict) -> list[str]:
        def sort_key(value: str) -> tuple[int, str]:
            try:
                return (0, int(value))
            except (TypeError, ValueError):
                return (1, str(value))

        return sorted(objects.keys(), key=sort_key)


# ----------------------------------------------------------------------
# Loading helpers
# ----------------------------------------------------------------------
def load_scene_config(demo_path: Path) -> dict:
    scene_path = Path(demo_path) / "scene.json"
    with open(scene_path, "r") as f:
        return json.load(f)


def load_object_metadata(asset_root: Path, scene_config: dict) -> Dict[str, dict]:
    metadata: Dict[str, dict] = {}
    asset_root = Path(asset_root)

    bg_name = "background_registered"
    bg_meta_path = asset_root / bg_name / f"{bg_name}_metadata.json"
    if bg_meta_path.exists():
        with open(bg_meta_path, "r", encoding="utf-8") as f:
            metadata["background"] = json.load(f)

    objects = scene_config["objects"]
    for oid, obj_cfg in objects.items():
        obj_name = obj_cfg["name"]
        body_name = f"{oid}_{obj_name}_optimized"
        meta_path = asset_root / body_name / f"{body_name}_metadata.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata[oid] = json.load(f)

    return metadata


def load_object_masses(config_path: Path, scene_config: dict, default_mass: float) -> Dict[str, float]:
    """Load object masses from YAML config.

    Args:
        config_path: Path to YAML file with name: mass mapping
        scene_config: Scene configuration dictionary
        default_mass: Default mass to use for objects not in config (with warning)

    Returns:
        Dictionary mapping object names to masses
    """
    with open(config_path, "r", encoding="utf-8") as f:
        mass_config = yaml.safe_load(f)

    masses: Dict[str, float] = {}
    for obj_cfg in scene_config["objects"].values():
        name = obj_cfg["name"]
        if name in mass_config:
            masses[name] = float(mass_config[name])
        else:
            masses[name] = default_mass
            logger.warning(f"Object '{name}' not found in mass config, using default mass: {default_mass} kg")

    return masses
