#!/usr/bin/env python3
"""Per-scene check of a finished reconstruction.

Measures, per object, the four things the agent-vs-pipeline experiment showed the pipeline
gets wrong (`tasks/agent_vs_pipeline/REPORT.md`, Tables 2 and 3): silhouette IoU and centroid
error of the optimized mesh against the frame-0 annotation mask; the thickness ratio, height
over convex-hull diameter measured the same way on the mesh and on the `fg_depth`
back-projection of that mask; a drop test over the ground plane and over
`background_registered.glb`; and mesh well-formedness.

Run as `python openreal2sim/steps/reconstruction/tools/scene_check.py --key <key>`, or import
`scene_check(scene_dir)` / `check_key(key)`: both return the dict that is also written to
`simulation/check.json`, next to a `simulation/check.png` figure.
"""

import os

# pyrender picks its backend at import time, and the hosts are headless.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import contextlib
import json
import math
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyrender
import trimesh
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion
from scipy.spatial import ConvexHull

# The pipeline ships neither a mass nor a friction coefficient, so the drop test answers
# "does this pose settle", not "how heavy is it".
MASS_KG = 0.2
FRICTION = 0.6
SIM_HZ = 480
DROP_SECONDS = 3.0
VHACD_RESOLUTION = 100_000
OCCLUSION_TOL_M = 2e-3          # a coincident surface in front must not count as a miss
MASK_ERODE_PX = 2               # the depth map is unreliable on the mask border
DEPTH_HEIGHT_PCT = 99.0         # ... and has outliers, so take a high percentile
CONTACT_XY_RADIUS_M = 5e-3      # background vertices counted as "under" an object
CONTACT_N_VERTS = 30


# --- loading ---

def as_mesh(loaded):
    """Concatenate every geometry of a GLB, instead of keeping only the first one."""
    if isinstance(loaded, trimesh.Scene):
        parts = [g for g in loaded.dump() if isinstance(g, trimesh.Trimesh)]
        if not parts:
            raise ValueError("no triangle mesh in scene")
        return trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
    return loaded


def resolve_asset(path, scene_dir):
    """Find an asset recorded as `/app/outputs/<key>/...` in a directory on this host."""
    if not path:
        return None
    name = Path(path).name
    for candidate in (Path(path), scene_dir / "simulation" / name,
                      scene_dir / "reconstruction" / "objects" / name,
                      scene_dir / "reconstruction" / name):
        if candidate.is_file():
            return candidate
    return None


def load_world_mesh(path, scene_dir):
    """Load a vertex-baked, world-frame GLB written by the pipeline."""
    resolved = resolve_asset(path, scene_dir)
    if resolved is None:
        return None, None
    return as_mesh(trimesh.load(resolved, process=False)), resolved


def camera_matrices(info):
    """Intrinsics and OpenCV camera-to-world of the frame-0 pipeline camera."""
    cam = info["camera"]
    K = np.array([[cam["fx"], 0.0, cam["cx"]], [0.0, cam["fy"], cam["cy"]], [0.0, 0.0, 1.0]])
    T = np.asarray(cam["camera_opencv_to_world"], dtype=float)
    return K, T, int(round(cam["width"])), int(round(cam["height"]))


def frame0_mask(scene, oid):
    """The annotation mask of one object in frame 0, whichever key type the pkl used."""
    frame = scene["mask"][0]
    entry = frame.get(oid, frame.get(int(oid), frame.get(str(oid))))
    if entry is not None:
        return np.asarray(entry["mask"], dtype=bool)
    mask = np.asarray(scene["objects"][oid]["mask"], dtype=bool)
    return mask[0] if mask.ndim == 3 else mask


# --- geometry ---

def mesh_report(mesh):
    """Well-formedness. Textured GLBs are split along every UV seam, so the verdict that
    matters is the one after merging those duplicated vertices."""
    merged = mesh.copy()
    merged.merge_vertices(merge_tex=True, merge_norm=True)
    merged.update_faces(merged.nondegenerate_faces())
    merged.remove_unreferenced_vertices()
    return {"faces": int(len(mesh.faces)), "watertight_raw": bool(mesh.is_watertight),
            "watertight_merged": bool(merged.is_watertight), "volume_m3": float(merged.volume),
            "euler_merged": int(merged.euler_number),
            "winding_consistent": bool(merged.is_winding_consistent)}


def _hull_xy(points_xy):
    """Convex-hull vertices of a 2D point set, or the set itself when it is degenerate."""
    if len(points_xy) < 3:
        return points_xy
    try:
        return points_xy[ConvexHull(points_xy).vertices]
    except Exception:
        return points_xy


def hull_diameter_xy(points_xy):
    """Largest distance between two hull vertices of the XY projection."""
    hull = _hull_xy(np.asarray(points_xy, dtype=float))
    if len(hull) < 2:
        return 0.0
    return float(np.sqrt(((hull[:, None, :] - hull[None, :, :]) ** 2).sum(-1)).max())


def footprint_box_xy(points_xy):
    """Rotating-caliper minimum-area box of the XY projection, long side first."""
    hull = _hull_xy(np.asarray(points_xy, dtype=float))
    if len(hull) < 3:
        extent = hull.max(0) - hull.min(0) if len(hull) else np.zeros(2)
        return float(max(extent)), float(min(extent))
    best = None
    for i in range(len(hull)):
        edge = hull[(i + 1) % len(hull)] - hull[i]
        length = np.linalg.norm(edge)
        if length < 1e-9:
            continue
        along = edge / length
        across = np.array([-along[1], along[0]])
        dims = (np.ptp(hull @ along), np.ptp(hull @ across))
        if best is None or dims[0] * dims[1] < best[0]:
            best = (dims[0] * dims[1], sorted(dims, reverse=True))
    return float(best[1][0]), float(best[1][1])


def back_project(depth, mask, K, T_c2w):
    """World-frame points of the depth map inside a mask."""
    height, width = depth.shape
    us, vs = np.meshgrid(np.arange(width), np.arange(height))
    valid = mask & np.isfinite(depth) & (depth > 0)
    z = depth[valid].astype(float)
    x = (us[valid] - K[0, 2]) / K[0, 0] * z
    y = (vs[valid] - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, z], -1) @ T_c2w[:3, :3].T + T_c2w[:3, 3]


def erode_mask(mask, pixels=MASK_ERODE_PX):
    eroded = binary_erosion(mask, iterations=int(pixels)) if pixels > 0 else mask
    return eroded if eroded.any() else mask  # a thin object would erode away entirely


def aabb_overlaps(world_meshes):
    """Axis-aligned overlap between every pair of placed objects, in mm."""
    found, names = [], list(world_meshes)
    for index, a in enumerate(names):
        for b in names[index + 1:]:
            low = np.maximum(world_meshes[a].bounds[0], world_meshes[b].bounds[0])
            high = np.minimum(world_meshes[a].bounds[1], world_meshes[b].bounds[1])
            if (high - low > 0).all():
                found.append({"a": a, "b": b, "overlap_mm": ((high - low) * 1000).tolist()})
    return found


def gap_to_support_mesh(obj_mesh, support_mesh):
    """Signed gap in mm between an object's underside and a support mesh: the highest support
    vertex within a small XY radius of each of the object's lowest vertices. Negative means
    the object already cuts into the support."""
    lowest = obj_mesh.vertices[np.argsort(obj_mesh.vertices[:, 2])[:CONTACT_N_VERTS]]
    support = support_mesh.vertices
    gaps = []
    for vertex in lowest:
        near = support[np.linalg.norm(support[:, :2] - vertex[:2], axis=1) <= CONTACT_XY_RADIUS_M]
        if len(near):
            gaps.append(vertex[2] - near[:, 2].max())
    return float(min(gaps) * 1000) if gaps else None


# --- rendering ---

class PinholeCamera(pyrender.camera.Camera):
    """OpenGL projection from an OpenCV intrinsics matrix: pyrender grew `IntrinsicsCamera`
    only after the version the pipeline image pins, and the check has to render the same
    pixels in the image and in the host venv."""

    def __init__(self, K, width, height, znear=0.01, zfar=50.0):
        super().__init__(znear=znear, zfar=zfar)
        self.K, self.width, self.height = np.asarray(K, dtype=float), width, height

    def get_projection_matrix(self, width=None, height=None):
        width = self.width if width is None else width
        height = self.height if height is None else height
        near, far = self.znear, self.zfar
        P = np.zeros((4, 4))
        P[0, 0] = 2.0 * self.K[0, 0] / width
        P[1, 1] = 2.0 * self.K[1, 1] / height
        P[0, 2] = 1.0 - 2.0 * self.K[0, 2] / width
        P[1, 2] = 2.0 * self.K[1, 2] / height - 1.0
        P[2, 2] = (far + near) / (near - far)
        P[2, 3] = (2 * far * near) / (near - far)
        P[3, 2] = -1.0
        return P


def render(meshes, K, T_c2w, width, height):
    """Colour and depth of world-frame meshes seen by an OpenCV camera."""
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.6, 0.6, 0.6])
    for mesh in meshes:
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
    pose = T_c2w @ np.diag([1.0, -1.0, -1.0, 1.0])  # pyrender looks down -Z, OpenCV down +Z
    scene.add(PinholeCamera(K, width, height), pose=pose)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=pose)
    renderer = pyrender.OffscreenRenderer(width, height)
    try:
        return renderer.render(scene)
    finally:
        renderer.delete()


def iou(a, b):
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else float("nan")


def centroid_px(mask):
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean(), ys.mean()]) if len(xs) else np.array([np.nan, np.nan])


# --- physics ---

def _quat_angle_deg(orientation):
    """Rotation away from identity, in degrees."""
    return math.degrees(2 * math.acos(min(abs(float(orientation[3])), 1.0)))


@contextlib.contextmanager
def _quiet_stdout():
    """V-HACD writes its progress straight to fd 1 from C; keep it out of the stage log."""
    saved = os.dup(1)
    try:
        with open(os.devnull, "w") as null:
            os.dup2(null.fileno(), 1)
        yield
    finally:
        os.dup2(saved, 1)
        os.close(saved)


def prepare_bodies(world_meshes, tmp_dir):
    """Export each placed mesh centred on its centre of mass, with a convex decomposition.
    Done once and reused by both drop tests, because V-HACD is by far the slow part."""
    import pybullet

    prepared = {}
    for name, mesh in world_meshes.items():
        body = mesh.copy()
        body.merge_vertices()
        centre = body.center_mass if body.is_volume else body.centroid
        if not np.isfinite(centre).all():
            centre = body.centroid
        body.apply_translation(-centre)
        path = os.path.join(tmp_dir, "".join(c if c.isalnum() else "_" for c in name) + ".obj")
        body.export(path)
        convex = path.replace(".obj", "_vhacd.obj")
        try:
            with _quiet_stdout():
                pybullet.vhacd(path, convex, os.path.join(tmp_dir, "vhacd.log"),
                               resolution=VHACD_RESOLUTION)
            if os.path.getsize(convex) > 0:
                path = convex
        except Exception:
            pass
        prepared[name] = (path, np.asarray(centre))
    return prepared


def drop_test(prepared, support, seconds=DROP_SECONDS):
    """Let the placed meshes fall for `seconds` and report how far they moved. `support` is
    ("plane", z) or ("mesh", path to an OBJ); the support mesh stays concave."""
    import pybullet

    client = pybullet.connect(pybullet.DIRECT)
    try:
        pybullet.setGravity(0, 0, -9.81, physicsClientId=client)
        pybullet.setPhysicsEngineParameter(fixedTimeStep=1.0 / SIM_HZ, numSolverIterations=100,
                                           physicsClientId=client)
        kind, value = support
        if kind == "plane":
            shape = pybullet.createCollisionShape(pybullet.GEOM_PLANE, planeNormal=[0, 0, 1],
                                                  physicsClientId=client)
            pybullet.createMultiBody(0, shape, basePosition=[0, 0, float(value)],
                                     physicsClientId=client)
        else:
            shape = pybullet.createCollisionShape(pybullet.GEOM_MESH, fileName=str(value),
                                                  flags=pybullet.GEOM_FORCE_CONCAVE_TRIMESH,
                                                  physicsClientId=client)
            pybullet.createMultiBody(0, shape, physicsClientId=client)

        bodies = {}
        for name, (path, centre) in prepared.items():
            shape = pybullet.createCollisionShape(pybullet.GEOM_MESH, fileName=path,
                                                  physicsClientId=client)
            body = pybullet.createMultiBody(MASS_KG, shape, basePosition=centre.tolist(),
                                            baseOrientation=[0, 0, 0, 1], physicsClientId=client)
            pybullet.changeDynamics(body, -1, lateralFriction=FRICTION, restitution=0.0,
                                    physicsClientId=client)
            bodies[name] = body

        pybullet.performCollisionDetection(physicsClientId=client)
        penetration = {name: -min([0.0] + [c[8] for c in pybullet.getContactPoints(
            bodyA=body, physicsClientId=client)]) * 1000 for name, body in bodies.items()}

        for _ in range(int(seconds * SIM_HZ)):
            pybullet.stepSimulation(physicsClientId=client)

        result = {}
        for name, body in bodies.items():
            position, orientation = pybullet.getBasePositionAndOrientation(
                body, physicsClientId=client)
            delta = np.asarray(position) - prepared[name][1]
            result[name] = {"drift_total_mm": float(np.linalg.norm(delta) * 1000),
                            "drift_xy_mm": float(np.linalg.norm(delta[:2]) * 1000),
                            "drift_z_mm": float(delta[2] * 1000),
                            "rot_deg": _quat_angle_deg(orientation),
                            "initial_penetration_mm": float(penetration[name])}
        return result
    finally:
        pybullet.disconnect(physicsClientId=client)


def run_drop_tests(world_meshes, plane_z, background, seconds):
    """Both drop tests, or a stated reason why neither of them ran."""
    try:
        import pybullet  # noqa: F401
    except ImportError:
        skipped = {"skipped": "pybullet is not installed"}
        return skipped, (skipped if background is not None else None)
    with tempfile.TemporaryDirectory(prefix="scene_check_") as tmp_dir:
        prepared = prepare_bodies(world_meshes, tmp_dir)
        on_plane = drop_test(prepared, ("plane", plane_z), seconds)
        on_background = None
        if background is not None:
            support = os.path.join(tmp_dir, "background.obj")
            background.export(support)
            on_background = drop_test(prepared, ("mesh", support), seconds)
    return on_plane, on_background


# --- the four checks ---

def silhouette_report(mesh, mask, scene_depth, K, T_c2w, width, height):
    """IoU and centroid error of one object's visible silhouette against its mask."""
    _, depth = render([mesh], K, T_c2w, width, height)
    visible = (depth > 0) & (depth <= scene_depth + OCCLUSION_TOL_M)
    mask_centroid, render_centroid = centroid_px(mask), centroid_px(visible)
    return visible, {"iou": iou(visible, mask), "mask_px": int(mask.sum()),
                     "render_px": int(visible.sum()),
                     "centroid_mask_px": mask_centroid.tolist(),
                     "centroid_render_px": render_centroid.tolist(),
                     "centroid_err_px": float(np.linalg.norm(render_centroid - mask_centroid))}


def thickness_report(mesh, points, plane_z):
    """Height over convex-hull diameter, for the mesh and for the depth cloud of its mask.
    A ratio above 1 means the object is taller than it is wide, so the mask's XY hull is a
    side view rather than a footprint: the comparison is then flagged, not trusted."""
    if len(points) < 3:
        return {"depth_points": int(len(points)),
                "flag": "too few depth points inside the eroded mask"}
    depth_diameter = hull_diameter_xy(points[:, :2])
    depth_height = float(np.percentile(points[:, 2] - plane_z, DEPTH_HEIGHT_PCT))
    depth_l, depth_w = footprint_box_xy(points[:, :2])
    mesh_diameter = hull_diameter_xy(mesh.vertices[:, :2])
    mesh_height = float(np.ptp(mesh.vertices[:, 2]))
    mesh_l, mesh_w = footprint_box_xy(mesh.vertices[:, :2])
    depth_ratio = depth_height / depth_diameter if depth_diameter > 0 else None
    mesh_ratio = mesh_height / mesh_diameter if mesh_diameter > 0 else None
    report = {"depth_points": int(len(points)), "depth_height_m": depth_height,
              "depth_hull_diameter_m": depth_diameter, "depth_thickness_ratio": depth_ratio,
              "mesh_height_m": mesh_height, "mesh_hull_diameter_m": mesh_diameter,
              "mesh_thickness_ratio": mesh_ratio,
              "thickness_ratio_mesh_over_depth":
                  mesh_ratio / depth_ratio if (depth_ratio and mesh_ratio) else None,
              "footprint_ratio_L": mesh_l / depth_l if depth_l > 0 else None,
              "footprint_ratio_W": mesh_w / depth_w if depth_w > 0 else None,
              "height_ratio_mesh_over_depth":
                  mesh_height / depth_height if depth_height > 0 else None,
              "standing": bool(mesh_ratio is not None and mesh_ratio > 1.0)}
    if report["standing"]:
        report["flag"] = ("standing object: the mask XY hull is a side view, not the "
                          "footprint, so the thickness ratio is not evidence of thickening")
    return report


def write_figure(path, photo, render_rgb, silhouettes, captions):
    """One row per object: photo, scene render, mask-versus-render overlay."""
    height, width = photo.shape[:2]
    rows = [np.concatenate([photo, render_rgb,
                            np.stack([mask, visible, np.zeros_like(mask)], -1) * 255], axis=1)
            for mask, visible in (silhouettes[label] for label in captions)]
    image = Image.fromarray(np.concatenate(rows, axis=0).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    for index, (label, caption) in enumerate(captions.items()):
        draw.text((6, index * height + 4), label, fill=(255, 255, 0))
        draw.text((width + 6, index * height + 4), "render", fill=(255, 255, 0))
        draw.text((2 * width + 6, index * height + 4), f"R mask / G render, {caption}",
                  fill=(255, 255, 0))
    image.save(path)


def scene_check(scene_dir, out_dir=None, seconds=DROP_SECONDS, write=True):
    """Check one reconstructed scene and return the result dict. `scene_dir` holds
    `scene/scene.pkl` and `simulation/`; `check.json` and `check.png` go to `out_dir`,
    by default `<scene_dir>/simulation`."""
    scene_dir = Path(scene_dir).resolve()
    out_dir = Path(out_dir) if out_dir else scene_dir / "simulation"
    with open(scene_dir / "scene" / "scene.pkl", "rb") as handle:
        scene = pickle.load(handle)

    info = scene["info"]
    K, T_c2w, width, height = camera_matrices(info)
    plane_z = float(np.asarray(info["groundplane_in_sim"]["point"])[2])
    fg_depth = np.asarray(scene["recon"]["fg_depth"], dtype=np.float32)
    photo = np.asarray(scene["images"][0])[..., :3].astype(np.uint8)

    objects, world_meshes, masks = {}, {}, {}
    for oid, obj in info["objects"].items():
        label = f"{obj['oid']}_{obj['name']}"
        mesh, path = load_world_mesh(obj.get("optimized"), scene_dir)
        if mesh is None:
            objects[label] = {"error": "optimized mesh not found",
                              "recorded_path": obj.get("optimized")}
            continue
        world_meshes[label] = mesh
        masks[label] = frame0_mask(scene, oid)
        objects[label] = {"oid": int(obj["oid"]), "name": obj["name"], "mesh": str(path),
                          "mesh_report": mesh_report(mesh)}
    background, background_path = load_world_mesh(
        (info.get("background") or {}).get("registered"), scene_dir)

    # The whole scene is rendered once so that a silhouette is only counted where the object
    # is actually the nearest surface.
    scene_meshes = ([background] if background is not None else []) + list(world_meshes.values())
    render_rgb, scene_depth = render(scene_meshes, K, T_c2w, width, height)
    silhouettes = {}
    for label, mesh in world_meshes.items():
        visible, report = silhouette_report(mesh, masks[label], scene_depth, K, T_c2w,
                                            width, height)
        objects[label]["silhouette"] = report
        silhouettes[label] = (masks[label], visible)
        points = back_project(fg_depth, erode_mask(masks[label]), K, T_c2w)
        objects[label]["thickness"] = thickness_report(mesh, points, plane_z)

    on_plane, on_background = run_drop_tests(world_meshes, plane_z, background, seconds)
    physics = {"mass_kg": MASS_KG, "friction": FRICTION, "seconds": seconds,
               "ground_plane_z_m": plane_z, "plane": on_plane,
               "aabb_overlaps": aabb_overlaps(world_meshes)}
    if background is not None:
        physics["background_mesh"] = str(background_path)
        physics["background"] = on_background
    for label, mesh in world_meshes.items():
        resting = {"gap_to_plane_mm": float((mesh.vertices[:, 2].min() - plane_z) * 1000)}
        if background is not None:
            resting["gap_to_background_mm"] = gap_to_support_mesh(mesh, background)
        objects[label]["resting"] = resting

    result = {"scene_dir": str(scene_dir),
              "camera": {"fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2],
                         "width": width, "height": height},
              "objects": objects, "physics": physics}

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "check.json", "w") as handle:
            json.dump(result, handle, indent=1, default=float)
        result["check_json"] = str(out_dir / "check.json")
        if silhouettes:
            captions = {label: "IoU %.3f, centroid %.1f px" % (
                objects[label]["silhouette"]["iou"],
                objects[label]["silhouette"]["centroid_err_px"]) for label in silhouettes}
            write_figure(out_dir / "check.png", photo, render_rgb, silhouettes, captions)
            result["check_png"] = str(out_dir / "check.png")
    return result


def check_key(key, base_dir=None, **kwargs):
    """Check `outputs/<key>`, relative to the current directory by default."""
    return scene_check(Path(base_dir or Path.cwd()) / "outputs" / key, **kwargs)


def _fmt(value, spec="%.3f"):
    return spec % value if isinstance(value, (int, float)) else "n/a"


def summarise(result):
    """One line per object: the numbers that decide whether a pipeline change helped."""
    lines = []
    for label, entry in result["objects"].items():
        if "silhouette" not in entry:
            lines.append(f"  {label:<18} {entry.get('error', 'skipped')}")
            continue
        thickness, plane = entry["thickness"], result["physics"]["plane"].get(label, {})
        lines.append(
            "  %-18s IoU %s  centroid %s px  thickness %s mesh / %s depth = x%s%s  "
            "drift %s mm  rot %s deg  gap %s mm  watertight %s" % (
                label, _fmt(entry["silhouette"]["iou"]),
                _fmt(entry["silhouette"]["centroid_err_px"], "%.1f"),
                _fmt(thickness.get("mesh_thickness_ratio")),
                _fmt(thickness.get("depth_thickness_ratio")),
                _fmt(thickness.get("thickness_ratio_mesh_over_depth")),
                " [standing]" if thickness.get("standing") else "",
                _fmt(plane.get("drift_total_mm"), "%.1f"), _fmt(plane.get("rot_deg"), "%.1f"),
                _fmt(entry["resting"]["gap_to_plane_mm"], "%+.1f"),
                entry["mesh_report"]["watertight_merged"]))
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Check one reconstructed scene.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--key", help="check outputs/<key>")
    target.add_argument("--dir", help="check a directory holding scene/ and simulation/")
    parser.add_argument("--out-dir", default=None, help="where check.json and check.png go")
    parser.add_argument("--seconds", type=float, default=DROP_SECONDS, help="drop-test duration")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    scene_dir = Path.cwd() / "outputs" / args.key if args.key else Path(args.dir)
    result = scene_check(scene_dir, out_dir=args.out_dir, seconds=args.seconds)
    print(f"[Info] scene check for {result['scene_dir']}")
    print(summarise(result))
    print(f"[Info] wrote {result['check_json']} and {result.get('check_png', 'no figure')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
