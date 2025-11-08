########################################################
## Commandline arguments (IsaacLab only; no --key)
########################################################
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
# NOTE: no --key here; we will batch keys from config/config.yaml
AppLauncher.add_app_launcher_args(parser)
args_isaaclab = parser.parse_args()
# Force GUI unless you toggle via IsaacLab flags
args_isaaclab.headless = True
app_launcher = AppLauncher(args_isaaclab)
simulation_app = app_launcher.app

########################################################
## Imports
########################################################
from dataclasses import dataclass
from typing import Literal, List
from pathlib import Path
import numpy as np
from PIL import Image
import trimesh
import json
import yaml
import contextlib
import os
import xml.etree.ElementTree as ET
from collections import defaultdict

import carb
import isaacsim.core.utils.stage as stage_utils
import omni.kit.app
from loguru import logger as log
from isaaclab.sim.converters import (
    MeshConverter,
    MeshConverterCfg,
    UrdfConverter,
    UrdfConverterCfg,
)
from isaaclab.sim.schemas import schemas_cfg
from isaaclab.utils.assets import check_file_path
from pxr import Usd, UsdPhysics
from rich.logging import RichHandler

# Extra USD/PhysX schemas + tokens
from pxr import PhysxSchema, Sdf, UsdGeom

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

########################################################
## Small utilities
########################################################
base_dir = Path.cwd()
out_dir = base_dir / "outputs"

def glb_to_usd(filename: str) -> str:
    assert filename.endswith(".glb")
    return filename.replace(".glb", ".usd")

def convert_glb_to_obj(glb_path):
    """
    Convert a GLB file to an OBJ file with texture.
    (Kept from your original file; unused by the flow unless you point inputs to .glb->.obj)
    """
    obj_name = str(glb_path).split("/")[-1].split(".")[0]
    obj_path = Path(glb_path).parent / f"{obj_name}.obj"
    mtl_path = Path(glb_path).parent / f"{obj_name}.mtl"
    texture_path = Path(glb_path).parent / f"{obj_name}.jpg"

    mesh = trimesh.load(glb_path)

    if hasattr(mesh.visual, 'material') and hasattr(mesh.visual.material, 'image'):
        image = mesh.visual.material.image
        if image is not None:
            Image.fromarray(np.asarray(image)).save(texture_path)

    with open(mtl_path, "w") as mtl_file:
        mtl_file.write(
            "newmtl material_0\nKa 1.000 1.000 1.000\nKd 1.000 1.000 1.000\n"
            "Ks 0.000 0.000 0.000\nNs 10.000\nd 1.0\nillum 2\n"
        )
        mtl_file.write(f"map_Kd {texture_path.name}\n")

    with open(obj_path, "w") as obj_file:
        obj_file.write(f"mtllib {mtl_path.name}\n")
        for v in mesh.vertices:
            obj_file.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for uv in mesh.visual.uv:
            obj_file.write(f"vt {uv[0]} {1 - uv[1]}\n")
        for f in mesh.faces:
            obj_file.write(f"f {f[0]+1}/{f[0]+1} {f[1]+1}/{f[1]+1} {f[2]+1}/{f[2]+1}\n")

    print(f"Converted GLB to OBJ: {obj_path}")
    return str(obj_path)

########################################################
## Args container (same fields as before)
########################################################
@dataclass
class Args:
    input: List[str]
    output: List[str]
    make_instanceable: bool = False
    # We will enforce convexDecomposition for both bg and objects in run_conversion()
    collision_approximation: Literal["convexHull", "convexDecomposition", "meshSimplification", "none"] = "convexDecomposition"
    mass: float | None = 1.0
    disable_gravity: bool = False
    kinematic_enabled: bool = False
    headless: bool = False
    exit_on_finish: bool = True

# We'll assign a per-key Args instance to this global so helper functions keep working unchanged.
args = None  # type: ignore

########################################################
## Converters and USD post-processing (unchanged)
########################################################
def convert_obj_to_usd(obj_path, usd_path):
    log.info(f"Converting {obj_path}")

    if not os.path.isabs(obj_path):
        obj_path = os.path.abspath(obj_path)
    if not check_file_path(obj_path):
        raise ValueError(f"Invalid mesh file path: {obj_path}")

    usd_path = os.path.abspath(usd_path)

    mass_props = schemas_cfg.MassPropertiesCfg(mass=args.mass) if args.mass is not None else None
    rigid_props = schemas_cfg.RigidBodyPropertiesCfg(
        disable_gravity=args.disable_gravity,
        kinematic_enabled=args.kinematic_enabled,
    )
    collision_props = schemas_cfg.CollisionPropertiesCfg(
        collision_enabled=args.collision_approximation != "none",
        contact_offset=0.006,
        rest_offset=-0.002,
        torsional_patch_radius=0.05,
        min_torsional_patch_radius=0.005,
    )

    mesh_converter_cfg = MeshConverterCfg(
        mass_props=mass_props,
        rigid_props=rigid_props,
        collision_props=collision_props,
        asset_path=obj_path,
        force_usd_conversion=True,
        usd_dir=os.path.dirname(usd_path),
        usd_file_name=os.path.basename(usd_path),
        make_instanceable=args.make_instanceable,
        collision_approximation="convexDecomposition",  # enforce
    )

    log.info(f"Conversion configuration: {mesh_converter_cfg}")
    MeshConverter(mesh_converter_cfg)

def ensure_rigidbody_and_set_kinematic(usd_path: str, disable_gravity: bool = True):
    stage = Usd.Stage.Open(usd_path)
    prim = stage.GetDefaultPrim()
    if not prim or not prim.IsValid():
        for p in stage.GetPseudoRoot().GetChildren():
            prim = p
            break
        if not prim or not prim.IsValid():
            raise RuntimeError(f"[ensure_rigidbody_and_set_kinematic] No valid prim in stage: {usd_path}")

    UsdPhysics.RigidBodyAPI.Apply(prim)
    try:
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        if hasattr(rb, "CreateKinematicEnabledAttr"):
            rb.CreateKinematicEnabledAttr(True)
        else:
            prim.CreateAttribute("physxRigidBody:kinematicEnabled", Sdf.ValueTypeNames.Bool).Set(True)
        if hasattr(rb, "CreateDisableGravityAttr"):
            rb.CreateDisableGravityAttr(bool(disable_gravity))
        else:
            prim.CreateAttribute("physxRigidBody:disableGravity", Sdf.ValueTypeNames.Bool).Set(bool(disable_gravity))
    except Exception:
        prim.CreateAttribute("physxRigidBody:kinematicEnabled", Sdf.ValueTypeNames.Bool).Set(True)
        prim.CreateAttribute("physxRigidBody:disableGravity", Sdf.ValueTypeNames.Bool).Set(bool(disable_gravity))
    try:
        UsdPhysics.RigidBodyAPI(prim).CreateDisableGravityAttr(bool(disable_gravity))
    except Exception:
        prim.CreateAttribute("physics:disableGravity", Sdf.ValueTypeNames.Bool).Set(bool(disable_gravity))
    stage.Save()

def enable_ccd_and_iters(usd_path: str, pos_iters: int = 14, vel_iters: int = 4,
                         enable_ccd: bool = True, enable_speculative_ccd: bool = True):
    stage = Usd.Stage.Open(usd_path)
    prim = stage.GetDefaultPrim()
    if not prim or not prim.IsValid():
        for p in stage.Traverse():
            prim = p
            break
        if not prim or not prim.IsValid():
            raise RuntimeError(f"[enable_ccd_and_iters] No valid prim in stage: {usd_path}")

    UsdPhysics.RigidBodyAPI.Apply(prim)
    try:
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        if enable_ccd:
            (rb.CreateEnableCCDAttr(True) if hasattr(rb, "CreateEnableCCDAttr")
             else prim.CreateAttribute("physxRigidBody:enableCCD", Sdf.ValueTypeNames.Bool).Set(True))
        if enable_speculative_ccd:
            (rb.CreateEnableSpeculativeCCDAttr(True) if hasattr(rb, "CreateEnableSpeculativeCCDAttr")
             else prim.CreateAttribute("physxRigidBody:enableSpeculativeCCD", Sdf.ValueTypeNames.Bool).Set(True))
        (rb.CreateSolverPositionIterationCountAttr(int(pos_iters)) if hasattr(rb, "CreateSolverPositionIterationCountAttr")
         else prim.CreateAttribute("physxRigidBody:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(int(pos_iters)))
        (rb.CreateSolverVelocityIterationCountAttr(int(vel_iters)) if hasattr(rb, "CreateSolverVelocityIterationCountAttr")
         else prim.CreateAttribute("physxRigidBody:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(int(vel_iters)))
    except Exception:
        if enable_ccd:
            prim.CreateAttribute("physxRigidBody:enableCCD", Sdf.ValueTypeNames.Bool).Set(True)
        if enable_speculative_ccd:
            prim.CreateAttribute("physxRigidBody:enableSpeculativeCCD", Sdf.ValueTypeNames.Bool).Set(True)
        prim.CreateAttribute("physxRigidBody:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(int(pos_iters))
        prim.CreateAttribute("physxRigidBody:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(int(vel_iters))
    stage.Save()

def set_convex_decomposition_params(usd_path: str,
                                    voxel_resolution: int = 500_000,
                                    max_convex_hulls: int = 24,
                                    max_vertices_per_hull: int = 64,
                                    concavity: float = 0.002,
                                    shrink_wrap: bool = True,
                                    overlap: bool = False,
                                    contact_offset: float = 0.006,
                                    rest_offset: float = -0.002):
    stage = Usd.Stage.Open(usd_path)
    root = stage.GetDefaultPrim()
    if not root or not root.IsValid():
        return

    for prim in stage.Traverse():
        if not prim or not prim.IsValid():
            continue
        if not prim.GetPath().HasPrefix(root.GetPath()):
            continue
        if not prim.IsA(UsdGeom.Mesh):
            continue

        UsdPhysics.CollisionAPI.Apply(prim)
        mesh_col = UsdPhysics.MeshCollisionAPI.Apply(prim)

        try:
            mesh_col.CreateApproximationAttr(PhysxSchema.Tokens.convexDecomposition)
        except Exception:
            try:
                mesh_col.GetApproximationAttr().Set("convexDecomposition")
            except Exception:
                prim.CreateAttribute("physics:approximation", Sdf.ValueTypeNames.Token).Set("convexDecomposition")

        try:
            cd_api = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
            if hasattr(cd_api, "CreateVoxelResolutionAttr"):
                cd_api.CreateVoxelResolutionAttr(int(voxel_resolution))
            else:
                prim.CreateAttribute("physxConvexDecomposition:voxelResolution", Sdf.ValueTypeNames.Int).Set(int(voxel_resolution))
            if hasattr(cd_api, "CreateMaxConvexHullsAttr"):
                cd_api.CreateMaxConvexHullsAttr(int(max_convex_hulls))
            else:
                prim.CreateAttribute("physxConvexDecomposition:maxConvexHulls", Sdf.ValueTypeNames.Int).Set(int(max_convex_hulls))
            if hasattr(cd_api, "CreateMaxNumVerticesPerCHAttr"):
                cd_api.CreateMaxNumVerticesPerCHAttr(int(max_vertices_per_hull))
            else:
                prim.CreateAttribute("physxConvexDecomposition:maxNumVerticesPerCH", Sdf.ValueTypeNames.Int).Set(int(max_vertices_per_hull))
            if hasattr(cd_api, "CreateConcavityAttr"):
                cd_api.CreateConcavityAttr(float(concavity))
            else:
                prim.CreateAttribute("physxConvexDecomposition:concavity", Sdf.ValueTypeNames.Float).Set(float(concavity))
            if hasattr(cd_api, "CreateShrinkWrapAttr"):
                cd_api.CreateShrinkWrapAttr(bool(shrink_wrap))
            else:
                prim.CreateAttribute("physxConvexDecomposition:shrinkWrap", Sdf.ValueTypeNames.Bool).Set(bool(shrink_wrap))
            if hasattr(cd_api, "CreateOverlapAttr"):
                cd_api.CreateOverlapAttr(bool(overlap))
            else:
                prim.CreateAttribute("physxConvexDecomposition:overlap", Sdf.ValueTypeNames.Bool).Set(bool(overlap))
        except Exception:
            prim.CreateAttribute("physxConvexDecomposition:voxelResolution", Sdf.ValueTypeNames.Int).Set(int(voxel_resolution))
            prim.CreateAttribute("physxConvexDecomposition:maxConvexHulls", Sdf.ValueTypeNames.Int).Set(int(max_convex_hulls))
            prim.CreateAttribute("physxConvexDecomposition:maxNumVerticesPerCH", Sdf.ValueTypeNames.Int).Set(int(max_vertices_per_hull))
            prim.CreateAttribute("physxConvexDecomposition:concavity", Sdf.ValueTypeNames.Float).Set(float(concavity))
            prim.CreateAttribute("physxConvexDecomposition:shrinkWrap", Sdf.ValueTypeNames.Bool).Set(bool(shrink_wrap))
            prim.CreateAttribute("physxConvexDecomposition:overlap", Sdf.ValueTypeNames.Bool).Set(bool(overlap))

        try:
            col_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            if hasattr(col_api, "CreateContactOffsetAttr"):
                col_api.CreateContactOffsetAttr(float(contact_offset))
            else:
                prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(float(contact_offset))
            if hasattr(col_api, "CreateRestOffsetAttr"):
                col_api.CreateRestOffsetAttr(float(rest_offset))
            else:
                prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(float(rest_offset))
        except Exception:
            prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(float(contact_offset))
            prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(float(rest_offset))

        try:
            UsdGeom.Mesh(prim).CreateDoubleSidedAttr(True)
        except Exception:
            pass

    stage.Save()

def make_visual_names_unique(xml_string: str) -> str:
    tree = ET.ElementTree(ET.fromstring(xml_string))
    root = tree.getroot()
    elements_with_name = root.findall(".//visual")
    name_counts = defaultdict(int)
    for element in elements_with_name:
        name = element.get("name")
        name_counts[name] += 1
    for element in elements_with_name:
        name = element.get("name")
        if name_counts[name] > 1:
            count = name_counts[name]
            new_name = f"{name}{count}"
            element.set("name", new_name)
            name_counts[name] -= 1
    return ET.tostring(root, encoding="unicode")

def make_urdf_with_unique_visual_names(urdf_path: str) -> str:
    xml_str = open(urdf_path, "r").read()
    xml_str = '<?xml version="1.0"?>' + "\n" + make_visual_names_unique(xml_str)
    urdf_unique_path = urdf_path.replace(".urdf", "_unique.urdf")
    with open(urdf_unique_path, "w") as f:
        f.write(xml_str)
    return urdf_unique_path

def make_obj_without_slashes(obj_path: str) -> str:
    obj_str = open(obj_path, "r").read()
    obj_str = obj_str.replace("/", "")
    obj_clean_path = obj_path.replace(".obj", "_clean.obj")
    with open(obj_clean_path, "w") as f:
        f.write(obj_str)
    return obj_clean_path

def convert_urdf_to_usd(urdf_path, usd_path):
    log.info(f"Converting {urdf_path}")
    if not os.path.isabs(urdf_path):
        urdf_path = os.path.abspath(urdf_path)
    urdf_converter_cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=os.path.dirname(usd_path),
        usd_file_name=os.path.basename(usd_path),
        make_instanceable=args.make_instanceable,
        force_usd_conversion=True,
        fix_base=True,
    )
    UrdfConverter(urdf_converter_cfg)

def is_articulation(usd_path: str):
    joint_count = 0
    stage = Usd.Stage.Open(usd_path)
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            joint_count += 1
    return joint_count > 0

def apply_rigidbody_api(usd_path):
    stage = Usd.Stage.Open(usd_path)
    defaultPrim = stage.GetDefaultPrim()
    UsdPhysics.RigidBodyAPI.Apply(defaultPrim)
    stage.Save()

def count_rigid_api(usd_path):
    stage = Usd.Stage.Open(usd_path)
    rigid_api_count = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_api_count += 1
    return rigid_api_count

########################################################
## Per-key conversion (batch)
########################################################
from typing import Optional
def run_conversion_for_key(key: str, args_cli:Optional[argparse.Namespace] = None):
    global args
    if args_cli is None:
        args_cli = args_isaaclab
    scene_json = out_dir / key / "simulation" / "scene.json"
    assert scene_json.exists(), f"[{key}] scene.json not found: {scene_json}"
    scene_dict = json.load(open(scene_json, "r"))

    # Build IO lists (background always first). Prefer "refined" if present.
    input_list, output_list = [scene_dict["background"]["registered"]], [glb_to_usd(scene_dict["background"]["registered"])]
    scene_dict["background"]["usd"] = glb_to_usd(scene_dict["background"]["registered"])
    for idx, obj in scene_dict["objects"].items():
        input_list.append(obj["optimized"])
        output_list.append(glb_to_usd(obj["optimized"]))
        scene_dict["objects"][idx]["usd"] = glb_to_usd(obj["optimized"])

    # Persist USD paths back to scene.json
    with open(scene_json, 'w') as f:
        json.dump(scene_dict, f, indent=2)

    # Create a fresh Args for this key
    args = Args(input=input_list, output=output_list, make_instanceable=False,
                collision_approximation="convexDecomposition",
                mass=1.0, disable_gravity=False, kinematic_enabled=False,
                headless=args_cli.headless, exit_on_finish=True)

    # Run the original main conversion logic for this key
    log.info(f"[{key}] converting {len(args.input)} assets...")
    run_conversion(scene_dict)

def run_conversion(scene_dict: dict):
    """This is the body of your original `main()` but parameterized and reused per key."""
    if isinstance(args.input, list) and isinstance(args.output, list):
        for input_path, output_path in zip(args.input, args.output):
            is_background = (input_path == scene_dict["background"]["registered"])

            if input_path.endswith(".obj") or input_path.endswith(".glb"):
                # Save current, then enforce per-asset overrides
                prev_mass = args.mass
                prev_approx = args.collision_approximation
                prev_disable_grav = args.disable_gravity
                prev_kinematic = args.kinematic_enabled

                args.collision_approximation = "convexDecomposition"

                if is_background:
                    args.mass = None
                    args.disable_gravity = True
                    args.kinematic_enabled = True
                else:
                    args.mass = 1.0
                    args.disable_gravity = False
                    args.kinematic_enabled = False

                convert_obj_to_usd(input_path, output_path)

                if is_background:
                    apply_rigidbody_api(output_path)
                    ensure_rigidbody_and_set_kinematic(output_path, disable_gravity=True)
                    set_convex_decomposition_params(
                        output_path,
                        voxel_resolution=600_000,
                        max_convex_hulls=512,
                        max_vertices_per_hull=128,
                        concavity=0.0005,
                        shrink_wrap=True,
                        overlap=False,
                        contact_offset=0.006,
                        rest_offset=-0.002
                    )
                else:
                    apply_rigidbody_api(output_path)
                    enable_ccd_and_iters(output_path, pos_iters=14, vel_iters=4,
                                         enable_ccd=True, enable_speculative_ccd=True)
                    set_convex_decomposition_params(
                        output_path,
                        voxel_resolution=600_000,
                        max_convex_hulls=24,
                        max_vertices_per_hull=64,
                        concavity=0.002,
                        shrink_wrap=True,
                        overlap=False,
                        contact_offset=0.006,
                        rest_offset=-0.002
                    )

                # Restore global args
                args.mass = prev_mass
                args.collision_approximation = prev_approx
                args.disable_gravity = prev_disable_grav
                args.kinematic_enabled = prev_kinematic

                log.info(f'Rigid body count: {count_rigid_api(output_path)}')
                log.info(f"Saved USD file to {os.path.abspath(output_path)}")

            elif input_path.endswith(".urdf"):
                urdf_unique_path = make_urdf_with_unique_visual_names(input_path)
                convert_urdf_to_usd(urdf_unique_path, output_path)

                if is_background:
                    apply_rigidbody_api(output_path)
                    ensure_rigidbody_and_set_kinematic(output_path, disable_gravity=True)
                    set_convex_decomposition_params(
                        output_path,
                        voxel_resolution=600_000,
                        max_convex_hulls=28,
                        max_vertices_per_hull=64,
                        concavity=0.0015,
                        shrink_wrap=True,
                        overlap=False,
                        contact_offset=0.006,
                        rest_offset=-0.002
                    )
                elif not is_articulation(output_path):
                    apply_rigidbody_api(output_path)
                    enable_ccd_and_iters(output_path, pos_iters=14, vel_iters=4,
                                         enable_ccd=True, enable_speculative_ccd=True)
                    set_convex_decomposition_params(
                        output_path,
                        voxel_resolution=600_000,
                        max_convex_hulls=24,
                        max_vertices_per_hull=64,
                        concavity=0.002,
                        shrink_wrap=True,
                        overlap=False,
                        contact_offset=0.006,
                        rest_offset=-0.002
                    )

                log.info(f'Rigid body count: {count_rigid_api(output_path)}')
                log.info(f"Saved USD file to {os.path.abspath(output_path)}")

    else:
        # Single-path branches kept for completeness (not used in batch flow)
        input_path = args.input
        output_path = args.output
        if input_path.endswith(".obj") or input_path.endswith(".glb"):
            convert_obj_to_usd(input_path, output_path)
            apply_rigidbody_api(output_path)
            enable_ccd_and_iters(output_path, pos_iters=14, vel_iters=4,
                                 enable_ccd=True, enable_speculative_ccd=True)
            set_convex_decomposition_params(
                output_path,
                voxel_resolution=600_000,
                max_convex_hulls=24,
                max_vertices_per_hull=64,
                concavity=0.002,
                shrink_wrap=True,
                overlap=False,
                contact_offset=0.006,
                rest_offset=-0.002
            )
            log.info(f'Rigid body count: {count_rigid_api(output_path)}')
            log.info(f"Saved USD file to {os.path.abspath(output_path)}")

        elif input_path.endswith(".urdf"):
            urdf_unique_path = make_urdf_with_unique_visual_names(input_path)
            convert_urdf_to_usd(urdf_unique_path, output_path)
            apply_rigidbody_api(output_path)
            enable_ccd_and_iters(output_path, pos_iters=14, vel_iters=4,
                                 enable_ccd=True, enable_speculative_ccd=True)
            set_convex_decomposition_params(
                output_path,
                voxel_resolution=600_000,
                max_convex_hulls=24,
                max_vertices_per_hull=64,
                concavity=0.002,
                shrink_wrap=True,
                overlap=False,
                contact_offset=0.006,
                rest_offset=-0.002
            )
            log.info(f'Rigid body count: {count_rigid_api(output_path)}')
            log.info(f"Saved USD file to {os.path.abspath(output_path)}")


def usd_conversion(keys, args_cli):
    for key in keys:
        log.info(f"\n========== [USD Convert] key: {key} ==========")
        run_conversion_for_key(key, args_cli)
    #simulation_app.close()
    print('[Info] USD conversion completed.')

########################################################
## Batch entry
########################################################
def main():
    # Load keys from config/config.yaml and run per key
    cfg = yaml.safe_load((base_dir / "config" / "config.yaml").open("r"))
    keys = cfg["keys"]
    for key in keys:
        log.info(f"\n========== [USD Convert] key: {key} ==========")
        run_conversion_for_key(key)

    # Exit or keep GUI (same behavior as your original main)
    if args and args.exit_on_finish:
        return

    if args and not args.headless:
        carb_settings_iface = carb.settings.get_settings()
        local_gui = carb_settings_iface.get("/app/window/enabled")
        livestream_gui = carb_settings_iface.get("/app/livestream/enabled")
        if local_gui or livestream_gui:
            # If you want to open the last output USD in a viewer, uncomment below
            # stage_utils.open_stage(args.output[-1] if isinstance(args.output, list) else args.output)
            app = omni.kit.app.get_app_interface()
            with contextlib.suppress(KeyboardInterrupt):
                while app.is_running():
                    app.update()

if __name__ == "__main__":
    main()
    simulation_app.close()
