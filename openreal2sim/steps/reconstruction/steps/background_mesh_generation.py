#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate the textured background mesh.

Two modes, selected by cfg["bg_table_mode"]:
  - "slab" (default): the table is built like an object. A plane is fitted on the
    recon.ground_mask pixels of the background depth, the mask outline is projected onto it to
    give a footprint (extended past the image border by a margin so objects cannot fall
    off), and the footprint is extruded downward into a flat slab. The top face carries
    the inpainted background image through UVs from the camera projection; the sides, the
    bottom and the part of the top that reaches past the image are a flat colour sampled
    from the table. Whatever the slab does not cover (walls, the far side of the table)
    can be kept as a thickened per-pixel height-map mesh in a second node
    ("background_depth"), off by default -- see the note below.
  - "depth": the old behaviour, the whole background including the tabletop meshed from
    per-pixel depth and thickened along the opposite direction of the ground normal. With
    the default config this reproduces the pre-slab mesh exactly.

Inputs:
    - outputs/{key}/scene/scene.pkl (must contain the "recon" key background_pixel_inpainting
      writes: the inpainted background image and the ground and object masks)
Outputs:
    - outputs/{key}/recon/background_textured_mesh.glb # background textured mesh
      ("slab" mode: node "table_slab", plus node "background_depth" when
       bg_table_keep_depth_layer is on; "depth" mode: a single mesh, as before)
    - outputs/{key}/recon/ground_normal.ply # estimated ground plane and normal direction (debug purpose)
    - outputs/{key}/recon/mesh_thickness_debug.ply # background mesh and pointmap ("depth" mode only, debug purpose)
Note:
    - added keys in "recon": "info", "normal"
    - the "info" key contains lightweight scenario info that can be exported to json
        "info": {
            "background": {
                "original": # original background mesh path,
            },
            "groundplane_in_cam": {
                "point":  # a point on the ground plane [x,y,z],
                "normal": # the normal of the ground plane [x,y,z],
            }
        }
    - nothing downstream distinguishes collision from visual geometry: every consumer of
      background_registered.glb (scenario_collision_optimization, Isaac Lab, ManiSkill,
      MuJoCo) feeds the whole file to its collider builder, so the "background_depth"
      node is collision geometry too. That is why it is off by default, and why it is
      thickened into a closed body (bg_mesh_thickness) when it is turned on.
"""

from pathlib import Path
import numpy as np
import open3d as o3d
import trimesh
import yaml
import pickle
from PIL import Image

from openreal2sim.steps.reconstruction.utils.table_slab import (
    boundary_edges, build_table_slab, clean_ground_mask, color_pointmap,
    complete_background_depth, depth_to_points, fit_plane_trimmed, flat_color_uv,
    footprint_polygon, plane_frame, plane_rms, slab_shadow_mask)

# used when cfg["bg_mesh_discontinuity_rel"] is left at None: the filter is on for the slab's
# depth layer and off in "depth" mode, which keeps that mode an exact reproduction of the
# pre-slab background mesh
SLAB_DEPTH_LAYER_DISCONTINUITY_REL = 0.02

# ──────────────────── util ─────────────────────
def add_thickness_below_mesh_preserve_texture(mesh: trimesh.Trimesh,
                                              thickness: float = 0.02,
                                              direction: np.ndarray | None = None
                                              ) -> trimesh.Trimesh:
    dir_vec = np.array([0, 0, -1], np.float32) if direction is None else direction.astype(np.float32)
    dir_vec /= (np.linalg.norm(dir_vec) + 1e-8)

    top_v  = mesh.vertices
    bot_v  = top_v + dir_vec * thickness
    top_f  = mesh.faces
    bot_f  = top_f[:, ::-1] + len(top_v)

    side_faces = []
    for v0, v1 in boundary_edges(mesh.faces):
        v0b, v1b = v0 + len(top_v), v1 + len(top_v)
        side_faces += [[v0, v1, v1b], [v0, v1b, v0b]]
    faces = np.vstack([top_f, bot_f, np.asarray(side_faces, int)])
    verts = np.vstack([top_v, bot_v])

    if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
        uv_top = mesh.visual.uv
        uv_all = np.zeros((len(verts), 2), uv_top.dtype)
        uv_all[:len(uv_top)] = uv_top
        new_vis = trimesh.visual.TextureVisuals(uv=uv_all,
                                                image=mesh.visual.material.image)
    else:
        new_vis = mesh.visual

    return trimesh.Trimesh(vertices=verts, faces=faces, visual=new_vis, process=False)

# ──────────────────── height-map aware simplification ─────────────────────
def _uniform_idx(N: int, step: int) -> np.ndarray:
    idx = np.arange(0, N, step, dtype=np.int64)
    if idx[-1] != N - 1:
        idx = np.concatenate([idx, [N - 1]])
    return idx

def _calc_step_for_target_faces(H: int, W: int, target_faces: int) -> int:
    orig = max(2 * (H - 1) * (W - 1), 1)
    if target_faces >= orig:
        return 1
    ratio = orig / max(target_faces, 1)
    step = int(np.ceil(np.sqrt(ratio)))
    return max(1, step)

def simplify_heightmap_pmap(pmap: np.ndarray,
                            step: int | None = None,
                            target_faces: int | None = None) -> tuple[np.ndarray, int, int, int]:
    H, W = pmap.shape[:2]
    if step is None:
        if target_faces is None or target_faces <= 0:
            step = 1
        else:
            step = _calc_step_for_target_faces(H, W, target_faces)
    step = max(1, int(step))

    if step == 1:
        return pmap, H, W, step

    r_idx = _uniform_idx(H, step)
    c_idx = _uniform_idx(W, step)
    pmap_ds = pmap[np.ix_(r_idx, c_idx)]
    H2, W2 = pmap_ds.shape[:2]
    return pmap_ds, H2, W2, step

# ──────────────────── depth discontinuity filter ─────────────────────
def depth_continuous_faces(verts: np.ndarray, faces: np.ndarray, rel_thresh: float) -> np.ndarray:
    """Mask of faces whose vertex depths agree within `rel_thresh` (relative to the depth).

    A quad that straddles a depth cliff (e.g. table edge -> far wall) is otherwise
    triangulated into long triangles that render as ramps between foreground and background.
    """
    fz = verts[:, 2][faces]
    zmin, zmax = fz.min(axis=1), fz.max(axis=1)
    keep = (zmax - zmin) <= rel_thresh * np.abs(zmin)
    return keep & np.isfinite(zmin) & np.isfinite(zmax)

def per_pixel_background_mesh(pmap: np.ndarray, img: Image.Image, cfg: dict,
                              normal: np.ndarray, signed: float,
                              drop_mask: np.ndarray | None = None
                              ) -> tuple[trimesh.Trimesh, dict]:
    """The per-pixel height-map mesh (the original background mesher).

    `drop_mask` (H,W bool) removes the faces that touch those pixels, which is how the
    tabletop is cut out of this layer when the slab replaces it.
    """
    H, W = pmap.shape[:2]
    simplify_step = cfg["bg_mesh_simplify_step"]
    target_faces = cfg["bg_mesh_target_faces"]
    pmap_s, H2, W2, grid_step = simplify_heightmap_pmap(
        pmap, step=(simplify_step if simplify_step and simplify_step > 1 else None),
        target_faces=(target_faces if target_faces and target_faces > 0 else None)
    )
    if (H2, W2) != (H, W):
        orig_F = 2 * (H - 1) * (W - 1)
        new_F = 2 * (H2 - 1) * (W2 - 1)
        print(f"[i] simplified grid: ({H},{W}) -> ({H2},{W2}) faces ~ {orig_F} -> {new_F}")

    verts = pmap_s[..., :3].reshape(-1, 3)
    faces = []
    for i in range(H2 - 1):
        base = i * W2
        for j in range(W2 - 1):
            faces += [[base + j, base + j + 1, base + j + W2],
                      [base + j + 1, base + j + W2 + 1, base + j + W2]]
    faces = np.asarray(faces, dtype=np.int32)
    diag = {"grid_faces": int(len(faces))}

    # drop the triangles that bridge a depth discontinuity (they render as ramps)
    # the tolerance scales with the grid step, since a coarser grid samples larger depth steps
    disc_rel = cfg["bg_mesh_discontinuity_rel"]
    if disc_rel and disc_rel > 0:
        keep = depth_continuous_faces(verts, faces, disc_rel * grid_step)
        print(f"[i] depth discontinuity filter (> {100 * disc_rel * grid_step:.1f}% relative depth jump): "
              f"faces {len(faces)} -> {int(keep.sum())}")
        faces = faces[keep]
        diag["faces_after_discontinuity"] = int(len(faces))

    if drop_mask is not None:
        drop_s = drop_mask[np.ix_(_uniform_idx(H, grid_step), _uniform_idx(W, grid_step))] \
            if grid_step > 1 else drop_mask
        drop_v = drop_s.reshape(-1)
        keep = ~drop_v[faces].any(axis=1)
        print(f"[i] tabletop cut out of the per-pixel layer: faces {len(faces)} -> {int(keep.sum())}")
        faces = faces[keep]
        diag["faces_after_drop_mask"] = int(len(faces))

    uu, vv = np.meshgrid(np.linspace(0, 1, W2, dtype=np.float32),
                         np.linspace(1, 0, H2, dtype=np.float32))
    uv = np.stack([uu, vv], -1).reshape(-1, 2)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    face_normals = mesh.face_normals
    if len(face_normals) > 0:
        dot_products = np.dot(face_normals, normal)
        if np.mean(dot_products) < 0:
            print("[Info] Flipping face orientation to align with ground normal")
            mesh.faces = np.flip(mesh.faces, axis=1)

    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, image=img)

    # flipping mesh normals for correct texturing
    if signed > 0:
        mesh.faces = mesh.faces[:, ::-1]
    return mesh, diag

def background_pointmap(recon: dict, bg_depth: np.ndarray, K: np.ndarray,
                        img_rgb: np.ndarray) -> np.ndarray:
    """The (H,W,6) xyzrgb map of the completed background, which only the per-pixel
    layers need: taken from the point completion step when it ran, else built here."""
    if "bg_pts" in recon:
        return recon["bg_pts"]
    return color_pointmap(bg_depth, K, img_rgb)

def visualize_plane_normal(plane_pts, normal, filename, num_arrow_pts=200, normal_length=0.5):
    plane_pts = np.asarray(plane_pts)
    normal = np.asarray(normal, dtype=float)
    normal = normal / (np.linalg.norm(normal) + 1e-8)

    plane_colors = np.tile(np.array([[0.6, 0.6, 0.6]]), (plane_pts.shape[0], 1))
    origin = plane_pts.mean(axis=0)
    arrow_points = (np.linspace(0, normal_length, num_arrow_pts).reshape(-1, 1) * normal) + origin
    arrow_colors = np.tile(np.array([[1.0, 0.0, 0.0]]), (num_arrow_pts, 1))

    all_pts = np.vstack([plane_pts, arrow_points])
    all_colors = np.vstack([plane_colors, arrow_colors])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)
    o3d.io.write_point_cloud(str(filename), pcd)

def visualize_mesh_thickness_with_pointmap(scene_dir: Path,
                                           mesh: trimesh.Trimesh,
                                           pmap_xyz: np.ndarray,
                                           out_name: str = "mesh_thickness_debug.ply",
                                           ds_stride: int = 4):
    xyz = pmap_xyz.reshape(-1, 3)
    if ds_stride > 1:
        H, W, _ = pmap_xyz.shape
        r = np.arange(0, H, ds_stride)
        c = np.arange(0, W, ds_stride)
        idx = (r[:, None] * W + c[None, :]).reshape(-1)
        idx = idx[idx < xyz.shape[0]]
        xyz_bg = xyz[idx]
    else:
        xyz_bg = xyz
    xyz_bg = xyz_bg[np.all(np.isfinite(xyz_bg), axis=1)]
    col_bg = np.tile(np.array([[1.0, 0, 0]]), (xyz_bg.shape[0], 1))

    verts = np.asarray(mesh.vertices)
    nv = verts.shape[0] // 2
    vtop = verts[:nv]
    vbot = verts[nv:]
    col_top = np.tile(np.array([[1.0, 0.0, 0.0]]), (vtop.shape[0], 1))
    col_bot = np.tile(np.array([[0.0, 0.0, 1.0]]), (vbot.shape[0], 1))

    all_pts = np.vstack([xyz_bg, vtop, vbot])
    all_cols = np.vstack([col_bg, col_top, col_bot])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)
    pcd.colors = o3d.utility.Vector3dVector(all_cols)

    out_dir = scene_dir / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / out_name
    o3d.io.write_point_cloud(str(out_file), pcd)
    print(f"✓ thickness debug cloud saved: {out_file}")


# ──────────────────── core ─────────────────────
def background_mesh_generation(keys, key_scene_dicts, key_cfgs):
    base_dir = Path.cwd()
    for key in keys:
        print(f"[Info] Processing {key}...\n")
        scene_dict = key_scene_dicts[key]
        cfg = key_cfgs[key]
        recon_dir = base_dir / f'outputs/{key}/reconstruction'
        recon_dir.mkdir(parents=True, exist_ok=True)
        recon = scene_dict.get("recon", {})
        missing = [k for k in ("background", "ground_mask", "object_mask") if k not in recon]
        if missing:
            print(f"[Error] No {missing} in 'recon' of {key}/scene/scene.pkl, please run background_pixel_inpainting.py first!")
            continue
        img_rgb = np.ascontiguousarray(recon["background"])
        img = Image.fromarray(img_rgb, mode="RGB")
        H, W = img_rgb.shape[:2]
        K = np.asarray(scene_dict["intrinsics"], np.float64)
        K32 = np.asarray(scene_dict["intrinsics"], np.float32)
        ground_mask = recon["ground_mask"]
        obj_mask = recon["object_mask"]
        depth0 = scene_dict["depths"][0]

        # the background depth: the frame-0 depth with the segmented objects replaced by the
        # ground plane behind them. The point completion step writes it when that opt-in step
        # runs, which is the only way to get a non-planar fill; otherwise it is the plane
        # fill, which is all the plane fit and the slab need
        if "bg_depth" in recon:
            bg_depth = np.asarray(recon["bg_depth"])
        else:
            bg_depth = complete_background_depth(depth0, K32, ground_mask, obj_mask, cfg)
        bg_xyz = depth_to_points(np.asarray(bg_depth, np.float64), K)

        # plane estimation: fitted on the ground mask pixels of the background depth, used
        # both for the mesh thickening direction and for the table slab
        plane_pts = bg_xyz[ground_mask].reshape(-1, 3)
        plane_pts = plane_pts[np.all(np.isfinite(plane_pts), axis=1)]
        normal, plane_offset, inliers = fit_plane_trimmed(
            plane_pts, dist_thresh=0.01, ransac_n=10, num_iterations=2000)
        print(f"[Info] [{key}] plane fit on {len(plane_pts)} ground points: "
              f"inliers {int(inliers.sum())} ({100 * inliers.mean():.1f}%), "
              f"RMS all {1000 * plane_rms(plane_pts, normal, plane_offset):.2f} mm, "
              f"RMS inliers {1000 * plane_rms(plane_pts[inliers], normal, plane_offset):.2f} mm")

        # we need to determine the normal direction (up or down)
        # we will assume most object points are above the ground plane, and use this hints to flip normal if needed
        fg_xyz = recon["fg_pts"][..., :3] if "fg_pts" in recon \
            else depth_to_points(depth0, K32).astype(np.float32)
        pts = fg_xyz[obj_mask].reshape(-1, 3)
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        obj_pts = pts
        assert len(obj_pts) > 0, f"[Error] No valid object points found in {key}! We need them to determine the ground normal direction."
        # a point on the fitted plane, closest to the centroid of the ground points
        p0 = plane_pts.mean(axis=0)
        p0 = p0 - (p0 @ normal + plane_offset) * normal
        signed = np.median((obj_pts - p0) @ normal)
        # np.mean((obj_pts - p0) @ normal) is always positive
        if signed < 0:
            print(f"[Info] Flipping normal direction for {key} because the median signed distance is negative")
            normal = -normal
            plane_offset = -plane_offset
            # `signed` keeps its sign on purpose: the per-pixel path uses it below to pick
            # the winding that textures correctly

        save_plane_normal_path = recon_dir / "ground_normal.ply"
        visualize_plane_normal(plane_pts, normal, save_plane_normal_path,
                            num_arrow_pts=200, normal_length=0.5)

        # background mesh thicken direction is opposite to ground normal (downward)
        direction = -normal
        print(f"[Info] plane normal = {normal}, thicken dir = {direction}")

        table_mode = cfg["bg_table_mode"]
        if table_mode not in ("slab", "depth"):
            raise ValueError(f"unknown bg_table_mode: {table_mode}")
        if cfg["bg_mesh_discontinuity_rel"] is None:
            cfg = dict(cfg, bg_mesh_discontinuity_rel=(
                0.0 if table_mode == "depth" else SLAB_DEPTH_LAYER_DISCONTINUITY_REL))

        if table_mode == "slab":
            # ── the table is built like an object: a flat textured slab ──
            e1, e2 = plane_frame(normal, p0)
            poly2, fdiag = footprint_polygon(
                ground_mask, K, normal, plane_offset, p0, e1, e2,
                mode=cfg["bg_table_footprint_mode"],
                simplify_px=cfg["bg_table_simplify_px"],
                border_px=cfg["bg_table_border_px"],
                margin=cfg["bg_table_margin"])
            if len(poly2) < 3:
                raise RuntimeError(f"[Error] [{key}] could not build a table footprint from ground_mask")
            flat_uv = flat_color_uv(img_rgb, clean_ground_mask(ground_mask), H, W)
            slab, sdiag = build_table_slab(
                poly2, p0, e1, e2, normal, K, (H, W),
                thickness=cfg["bg_table_thickness"], max_edge=cfg["bg_table_max_edge"],
                flat_uv=flat_uv, img=img)
            print(f"[Info] [{key}] table footprint: {fdiag}")
            print(f"[Info] [{key}] table slab: {sdiag}")

            geoms = {"table_slab": slab}
            if cfg["bg_table_keep_depth_layer"]:
                # everything the slab does not cover stays a per-pixel mesh, in its own node.
                # it is thickened like the old background mesh, because downstream it is a
                # collider as well: nothing separates visual from collision geometry
                shadow = slab_shadow_mask(bg_xyz[..., 2], K,
                                          normal, plane_offset, p0, e1, e2, poly2,
                                          ground_mask=clean_ground_mask(ground_mask))
                print(f"[Info] [{key}] slab hides {100 * shadow.mean():.1f}% of the pixels")
                layer, ldiag = per_pixel_background_mesh(
                    background_pointmap(recon, bg_depth, K32, img_rgb),
                    img, cfg, normal, signed, drop_mask=shadow)
                if len(layer.faces):
                    # most of the pixel grid now belongs to the slab, so drop the vertices
                    # the remaining faces do not use before thickening doubles them
                    layer.remove_unreferenced_vertices()
                    layer = add_thickness_below_mesh_preserve_texture(
                        layer, cfg["bg_mesh_thickness"], direction)
                    ldiag["thickened_faces"] = int(len(layer.faces))
                    geoms["background_depth"] = layer
                    print(f"[Info] [{key}] background depth layer: {ldiag}")
                else:
                    print(f"[Info] [{key}] the slab covers the whole background, "
                          f"no depth layer to keep")
            # always a Scene, so the layers are named in the file; note that trimesh loads
            # any glb as a Scene anyway, single mesh or not
            mesh_path = recon_dir / "background_textured_mesh.glb"
            out_scene = trimesh.Scene()
            for name, geom in geoms.items():
                out_scene.add_geometry(geom, node_name=name, geom_name=name)
            out_scene.export(mesh_path)
            print(f"[Info] [{key}] background textured mesh saved: {mesh_path} "
                  f"(nodes: {list(geoms)})")
        else:
            # ── old behaviour: the whole background, tabletop included, from per-pixel depth ──
            pmap = background_pointmap(recon, bg_depth, K32, img_rgb)
            mesh, _ = per_pixel_background_mesh(pmap, img, cfg, normal, signed)
            thickness = cfg["bg_mesh_thickness"]
            mesh_thick = add_thickness_below_mesh_preserve_texture(mesh, thickness, direction)
            mesh_path = recon_dir / "background_textured_mesh.glb"
            mesh_thick.export(mesh_path)
            visualize_mesh_thickness_with_pointmap(recon_dir, mesh_thick, pmap[..., :3])
            print(f"[Info] [{key}] background textured mesh saved: {mesh_path}")

        # update scene_dict
        scene_dict["info"] = scene_dict.get("info", {})
        scene_dict["info"]["background"] = {
            "original": str(mesh_path),
        }
        scene_dict["info"]["groundplane_in_cam"] = {
            "point": p0.tolist(),
            "normal": normal.tolist(),
        }
        scene_dict["recon"]["normal"] = normal.astype(np.float32) # store normal direction
        key_scene_dicts[key] = scene_dict
        with open(base_dir / f'outputs/{key}/scene/scene.pkl', 'wb') as f:
            pickle.dump(scene_dict, f)

        print(f"[Info] [{key}] scene_dict updated.")

    
    return key_scene_dicts

# ──────────────────── batch main ─────────────────────
if __name__ == "__main__":
    base_dir = Path.cwd()
    cfg_path = base_dir / "config" / "config.yaml"
    cfg = yaml.safe_load(cfg_path.open("r"))
    keys = cfg["keys"]
    from openreal2sim.utils.compose_config import compose_configs
    key_cfgs = {key: compose_configs(key, cfg, "reconstruction") for key in keys}
    key_scene_dicts = {}
    for key in keys:
        scene_pkl = base_dir / f'outputs/{key}/scene/scene.pkl'
        with open(scene_pkl, 'rb') as f:
            scene_dict = pickle.load(f)
        key_scene_dicts[key] = scene_dict

    background_mesh_generation(keys, key_scene_dicts, key_cfgs)
