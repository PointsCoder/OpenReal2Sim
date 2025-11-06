import argparse
import json
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
from loguru import logger

CURRENT_DIR = Path(__file__).resolve().parent
MUJOCO_DIR = CURRENT_DIR.parent
if str(MUJOCO_DIR) not in sys.path:
    sys.path.insert(0, str(MUJOCO_DIR))

from mujoco_asset_builder import constants


class SceneFusion:
    def __init__(self, masses: dict[str, float] | None = None, z_offset: float | None = None, inertia_scale: float | None = None):
        self.masses = masses if masses is not None else {}
        self.z_offset = z_offset if z_offset is not None else constants.DEFAULT_Z_OFFSET
        self.inertia_scale = inertia_scale if inertia_scale is not None else constants.DEFAULT_INERTIA_SCALE

        if self.inertia_scale == constants.DEFAULT_INERTIA_SCALE:
            logger.warning(
                f"Using default inertia_scale={constants.DEFAULT_INERTIA_SCALE}. For better physics, set explicit inertia values.")

    def generate_asset_section(self, scene_config, object_metadata):
        """Generate asset XML for all objects"""
        lines = []

        if constants.ASSET_TYPE_BACKGROUND in object_metadata:
            bg = object_metadata[constants.ASSET_TYPE_BACKGROUND]
            bg_name = f"{constants.ASSET_TYPE_BACKGROUND}_registered"
            lines.append(f'    <!-- {constants.ASSET_TYPE_BACKGROUND.capitalize()} assets -->')
            lines.append(
                f'    <texture type="2d" name="{bg_name}_{constants.MATERIAL_PREFIX}" file="{bg_name}/{constants.MATERIAL_PREFIX}{constants.EXT_PNG}"/>')
            lines.append(
                f'    <material name="{bg_name}_{constants.MATERIAL_PREFIX}" texture="{bg_name}_{constants.MATERIAL_PREFIX}" specular="{constants.MATERIAL_SPECULAR}" shininess="{constants.MATERIAL_SHININESS}"/>')
            lines.append(
                f'    <mesh file="{bg_name}/{bg_name}{constants.EXT_OBJ}"/>')

            for i in range(bg['collision_parts']):
                lines.append(
                    f'    <mesh file="{bg_name}/{bg_name}{constants.SUFFIX_COLLISION}{i}{constants.EXT_OBJ}"/>')

        # Object assets
        objects = scene_config.get('objects', {})
        for oid in sorted(objects.keys()):
            obj_data = objects[oid]
            obj_name = obj_data['name']
            metadata = object_metadata.get(oid)

            if not metadata:
                logger.warning(
                    f"No metadata found for object {oid} ({obj_name})")
                continue

            body_name = f"{oid}_{obj_name}{constants.SUFFIX_OPTIMIZED}"
            lines.append(f'\n    <!-- {obj_name.capitalize()} assets -->')
            lines.append(
                f'    <texture type="2d" name="{obj_name}_{constants.MATERIAL_PREFIX}" file="{body_name}/{constants.MATERIAL_PREFIX}{constants.EXT_PNG}"/>')
            lines.append(
                f'    <material name="{obj_name}_{constants.MATERIAL_PREFIX}" texture="{obj_name}_{constants.MATERIAL_PREFIX}" specular="{constants.MATERIAL_SPECULAR}" shininess="{constants.MATERIAL_SHININESS}"/>')
            lines.append(
                f'    <mesh file="{body_name}/{body_name}{constants.EXT_OBJ}"/>')

            for i in range(metadata['collision_parts']):
                lines.append(
                    f'    <mesh file="{body_name}/{body_name}{constants.SUFFIX_COLLISION}{i}{constants.EXT_OBJ}"/>')

        return '\n'.join(lines)

    def generate_background_geoms(self, object_metadata):
        """Generate static background geometry"""
        bg = object_metadata.get(constants.ASSET_TYPE_BACKGROUND)
        if not bg:
            return ''

        bg_name = f"{constants.ASSET_TYPE_BACKGROUND}_registered"
        lines = []
        lines.append(
            f'    <!-- {constants.ASSET_TYPE_BACKGROUND.capitalize()} (static) - directly in worldbody -->')
        lines.append(
            f'    <geom name="{bg_name}_visual" class="{constants.CLASS_OBJ_VISUAL}" mesh="{bg_name}" material="{bg_name}_{constants.MATERIAL_PREFIX}"/>')

        for i in range(bg['collision_parts']):
            lines.append(
                f'    <geom name="{bg_name}{constants.SUFFIX_COLLISION}{i}" mesh="{bg_name}{constants.SUFFIX_COLLISION}{i}" class="{constants.CLASS_OBJ_COLLISION}"/>')

        return '\n'.join(lines)

    def generate_object_bodies(self, scene_config, object_metadata):
        """Generate dynamic object bodies"""
        lines = []
        objects = scene_config.get('objects', {})

        for oid in sorted(objects.keys()):
            obj_data = objects[oid]
            obj_name = obj_data['name']
            obj_center = obj_data['object_center']
            metadata = object_metadata.get(oid)

            if not metadata:
                continue

            if obj_name not in self.masses:
                raise ValueError(
                    f"No mass specified for object '{obj_name}'. "
                    f"Pass it to SceneFusion constructor: masses={{'{obj_name}': <value>}}"
                )
            mass = self.masses[obj_name]

            # Calculate diagonal inertia based on mass using configured scale factor
            inertia = mass * self.inertia_scale
            diag_inertia = f"{inertia:.6f} {inertia:.6f} {inertia:.6f}"

            body_name = f"{oid}_{obj_name}{constants.SUFFIX_OPTIMIZED}"
            lines.append(
                f'\n    <!-- {obj_name.capitalize()} (object {oid}) -->')
            lines.append(
                f'    <body name="{body_name}" pos="0 0 {self.z_offset}">')
            lines.append(f'      <joint type="{constants.JOINT_TYPE_FREE}" damping="{constants.OBJ_FREEJOINT_DAMPING}"/>')
            lines.append(
                f'      <inertial pos="{obj_center[0]} {obj_center[1]} {obj_center[2]}" mass="{mass}" diaginertia="{diag_inertia}"/>')
            lines.append(
                f'      <geom class="{constants.CLASS_OBJ_VISUAL}" mesh="{body_name}" material="{obj_name}_{constants.MATERIAL_PREFIX}"/>')

            for i in range(metadata['collision_parts']):
                lines.append(
                    f'      <geom mesh="{body_name}{constants.SUFFIX_COLLISION}{i}" class="{constants.CLASS_OBJ_COLLISION}"/>')

            lines.append('    </body>')

        return '\n'.join(lines)

    def fuse_scene(self, robot_scene_path, robot_panda_path, scene_config, object_metadata):
        """Fuse robot scene with objects"""
        logger.info("Fusing scene...")

        # Parse both XMLs
        scene_tree = ET.parse(robot_scene_path)
        panda_tree = ET.parse(robot_panda_path)

        scene_root = scene_tree.getroot()
        panda_root = panda_tree.getroot()

        # Start from panda.xml structure
        root = panda_root

        # Update compiler meshdir/texturedir and update all panda mesh paths
        compiler = root.find('compiler')
        if compiler is None:
            compiler = ET.Element('compiler')
            root.insert(0, compiler)

        old_meshdir = compiler.get('meshdir')
        if old_meshdir:
            # Update only mesh FILE paths in asset section (not geom mesh references!)
            mesh_prefix = f'../../robots/franka_emika_panda/{old_meshdir}/'

            mesh_count = 0
            for mesh_elem in root.iter('mesh'):
                file_attr = mesh_elem.get('file')
                if file_attr and not file_attr.startswith('../../'):
                    mesh_elem.set('file', mesh_prefix + file_attr)
                    mesh_count += 1

            logger.debug(f"Updated {mesh_count} mesh file paths with prefix {mesh_prefix}")

        compiler.set('meshdir', constants.MESHDIR_VFS)
        compiler.set('texturedir', constants.TEXTUREDIR_VFS)
        logger.debug(f"Set meshdir=\"{constants.MESHDIR_VFS}\" texturedir=\"{constants.TEXTUREDIR_VFS}\"")

        option_elem = root.find('option')
        if option_elem is not None and not option_elem.get('timestep'):
            option_elem.set('timestep', constants.DEFAULT_TIMESTEP)
            logger.debug(f"Added timestep=\"{constants.DEFAULT_TIMESTEP}\" to option")

        size_elem = root.find('size')
        if size_elem is None:
            size_elem = ET.Element('size')
            option_elem = root.find('option')
            insert_at = list(root).index(option_elem) + \
                1 if option_elem is not None else 0
            root.insert(insert_at, size_elem)
        size_elem.set('memory', constants.DEFAULT_MEMORY)
        logger.debug(f"Set <size memory=\"{constants.DEFAULT_MEMORY}\"> for MuJoCo WASM stack allocation")

        # Add object classes to defaults
        default_elem = root.find('default')
        if default_elem is None:
            default_elem = ET.SubElement(root, 'default')
            if compiler is not None:
                # Move default after compiler
                root.remove(default_elem)
                idx = list(root).index(compiler) + 1
                root.insert(idx, default_elem)

        obj_visual = ET.SubElement(default_elem, 'default', {
                                   'class': constants.CLASS_OBJ_VISUAL})
        ET.SubElement(obj_visual, 'geom', {
            'group': constants.GROUP_VISUAL,
            'type': constants.GEOM_TYPE_MESH,
            'contype': constants.CONTYPE_NONE,
            'conaffinity': constants.CONAFFINITY_NONE
        })
        logger.debug(f"Added {constants.CLASS_OBJ_VISUAL} class")

        obj_collision = ET.SubElement(default_elem, 'default', {
                                      'class': constants.CLASS_OBJ_COLLISION})
        ET.SubElement(obj_collision, 'geom', {
            'group': constants.GROUP_COLLISION,
            'type': constants.GEOM_TYPE_MESH,
            'margin': constants.OBJ_COLLISION_MARGIN,
            'solref': constants.OBJ_COLLISION_SOLREF,
            'solimp': constants.OBJ_COLLISION_SOLIMP
        })
        logger.debug(f"Added {constants.CLASS_OBJ_COLLISION} class")

        # Insert statistic and visual before asset (correct MJCF order)
        asset_elem = root.find('asset')
        insert_idx = list(root).index(
            asset_elem) if asset_elem is not None else len(root)

        for elem_name in ['statistic', 'visual']:
            elem = scene_root.find(elem_name)
            if elem is not None:
                # Remove existing if present
                existing = root.find(elem_name)
                if existing is not None:
                    root.remove(existing)
                root.insert(insert_idx, elem)
                insert_idx += 1
                logger.debug(f"Merged <{elem_name}>")

        # Merge scene.xml assets
        scene_asset = scene_root.find('asset')
        if scene_asset is not None:
            if asset_elem is None:
                asset_elem = ET.SubElement(root, 'asset')
            for child in scene_asset:
                asset_elem.append(child)
                logger.debug(
                    f"Merged asset: <{child.tag} name=\"{child.get('name', '')}\">")
        scene_worldbody = scene_root.find('worldbody')
        worldbody = root.find('worldbody')
        if worldbody is None:
            worldbody = ET.SubElement(root, 'worldbody')

        if scene_worldbody is not None:
            for child in scene_worldbody:
                worldbody.append(child)
                logger.debug(
                    f"Merged worldbody: <{child.tag} name=\"{child.get('name', '')}\">")
        asset_xml = self.generate_asset_section(scene_config, object_metadata)
        if asset_xml:
            temp_xml = f'<temp>{asset_xml}</temp>'
            temp_root = ET.fromstring(temp_xml)
            for child in temp_root:
                asset_elem.append(child)
            logger.debug(f"Added {len(temp_root)} object assets")

        # Add background geoms at beginning of worldbody
        bg_xml = self.generate_background_geoms(object_metadata)
        if bg_xml:
            temp_xml = f'<temp>{bg_xml}</temp>'
            temp_root = ET.fromstring(temp_xml)
            for child in reversed(list(temp_root)):
                worldbody.insert(0, child)
            logger.debug(f"Added {len(temp_root)} background geoms")

        # Add object bodies at end of worldbody
        obj_xml = self.generate_object_bodies(scene_config, object_metadata)
        if obj_xml:
            temp_xml = f'<temp>{obj_xml}</temp>'
            temp_root = ET.fromstring(temp_xml)
            for child in temp_root:
                worldbody.append(child)
            logger.debug(f"Added {len(temp_root)} object bodies")

        logger.info("Scene fused successfully")
        return panda_tree


def main():
    parser = argparse.ArgumentParser(
        description='Fuse robot scene with objects')
    parser.add_argument('--demo', required=True,
                        help='Demo name (e.g., demo_genvideo)')
    parser.add_argument('--output', required=True, help='Output XML path')
    parser.add_argument('--z-offset', type=float, default=constants.DEFAULT_Z_OFFSET,
                        help=f'Z offset for objects (default: {constants.DEFAULT_Z_OFFSET}m)')
    args = parser.parse_args()

    # Setup paths
    repo_root = Path(__file__).parent.parent
    demo_path = repo_root / 'public' / 'demos' / args.demo
    robot_path = repo_root / 'public' / 'robots' / 'franka_emika_panda'

    scene_config_path = demo_path / constants.FILENAME_SCENE
    logger.info(f"Loading scene config: {scene_config_path}")
    with open(scene_config_path) as f:
        scene_config = json.load(f)

    object_metadata = {}

    bg_name = f"{constants.ASSET_TYPE_BACKGROUND}_registered"
    bg_meta_path = demo_path / bg_name / f"{bg_name}{constants.SUFFIX_METADATA}"
    if bg_meta_path.exists():
        with open(bg_meta_path) as f:
            object_metadata[constants.ASSET_TYPE_BACKGROUND] = json.load(f)
        logger.debug(
            f"{constants.ASSET_TYPE_BACKGROUND}: {object_metadata[constants.ASSET_TYPE_BACKGROUND]['collision_parts']} collision parts")

    objects = scene_config.get('objects', {})
    for oid, obj_data in objects.items():
        obj_name = obj_data['name']
        body_name = f"{oid}_{obj_name}{constants.SUFFIX_OPTIMIZED}"
        meta_path = demo_path / body_name / f"{body_name}{constants.SUFFIX_METADATA}"
        with open(meta_path) as f:
            object_metadata[oid] = json.load(f)
        logger.debug(
            f"{oid}: {object_metadata[oid]['collision_parts']} collision parts")

    # Fuse scene with explicit masses
    masses = {
        'spoon': 0.05,
        'plate': 0.15,
        'cup': 0.1,
        'bowl': 0.2,
    }
    fusion = SceneFusion(z_offset=args.z_offset, masses=masses)
    fused_tree = fusion.fuse_scene(
        robot_path / constants.FILENAME_SCENE_XML,
        robot_path / constants.FILENAME_PANDA_XML,
        scene_config,
        object_metadata
    )

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ET.indent(fused_tree, space=constants.XML_INDENTATION)
    fused_tree.write(output_path, encoding='utf-8', xml_declaration=True)

    logger.info(f"Fused scene written to: {output_path}")
    logger.info(f"File size: {output_path.stat().st_size} bytes")


if __name__ == '__main__':
    main()
