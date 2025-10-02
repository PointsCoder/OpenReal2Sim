#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate a textured background height-map mesh using background pointmap.
and optionally thicken it along the opposite direction of the ground normal
Inputs:
    - outputs/{key_name}/scene/scene.pkl (must contain a "bg_pts" key in "recon" containing background pointmap)
Outputs: 
    - outputs/{key_name}/recon/background_textured_mesh.glb # background textured mesh
    - outputs/{key_name}/recon/ground_normal.ply # estimated ground plane and normal direction (debug purpose)
    - outputs/{key_name}/recon/mesh_thickness_debug.ply # background mesh and pointmap (debug purpose)
Note:
    - added keys in "recon": "bg_mesh_path", "normal"
"""

from pathlib import Path
import numpy as np
import open3d as o3d
import trimesh
import yaml
import pickle

# ──────────────────── util ─────────────────────
def get_boundary_edges(mesh: trimesh.Trimesh):
    edges = np.vstack([
        mesh.faces[:, [0, 1]],
        mesh.faces[:, [1, 2]],
        mesh.faces[:, [2, 0]]
    ])
    edges = np.sort(edges, axis=1)
    edges_unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = edges_unique[counts == 1]
    return boundary_edges

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
    for v0, v1 in get_boundary_edges(mesh):
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
                            target_faces: int | None = None) -> tuple[np.ndarray, int, int]:
    H, W = pmap.shape[:2]
    if step is None:
        if target_faces is None or target_faces <= 0:
            step = 1
        else:
            step = _calc_step_for_target_faces(H, W, target_faces)
    step = max(1, int(step))

    if step == 1:
        return pmap, H, W

    r_idx = _uniform_idx(H, step)
    c_idx = _uniform_idx(W, step)
    pmap_ds = pmap[np.ix_(r_idx, c_idx)]
    H2, W2 = pmap_ds.shape[:2]
    return pmap_ds, H2, W2

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
    for key in keys:
        scene_dict = key_scene_dicts[key]
        cfg = key_cfgs[key]
        recon_dir = base_dir / f'outputs/{key}/reconstruction'
        recon_dir.mkdir(parents=True, exist_ok=True)
        recon = scene_dict.get("recon", {})
        if "bg_pts" not in recon:
            print(f"[Error] No 'bg_pts' in 'recon' of {key}/scene/scene.pkl, please run background_pixel_inpainting.py first!")
            continue
        pmap = recon["bg_pts"]  # (H,W,6) float32
        fg_pmap = recon["fg_pts"] # (H,W,6) float32

        H, W = pmap.shape[:2]
        img = recon["background"]

        # plane estimation to determine mesh thickening direction
        ground_mask = recon["ground_mask"]
        plane_pts = pmap[..., :3][ground_mask].reshape(-1, 3)
        plane_pts = plane_pts[np.all(np.isfinite(plane_pts), axis=1)]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(plane_pts)
        plane_model, _ = pcd.segment_plane(distance_threshold=0.01,
                                           ransac_n=3, num_iterations=1000)
        a, b, c, d = plane_model
        normal = np.array([a, b, c], np.float64)
        normal /= (np.linalg.norm(normal) + 1e-12)

        # we need to determine the normal direction (up or down)
        # we will assume most object points are above the ground plane, and use this hints to flip normal if needed
        obj_mask = recon["object_mask"]
        pts = fg_pmap[..., :3][obj_mask].reshape(-1, 3)
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        obj_pts = pts
        assert len(obj_pts) > 0, f"[Error] No valid object points found in {key}! We need them to determine the ground normal direction."
        p0 = plane_pts.mean(axis=0)
        signed = np.median((obj_pts - p0) @ normal)
        if signed < 0:
            normal = -normal

        save_plane_normal_path = recon_dir / "ground_normal.ply"
        visualize_plane_normal(plane_pts, normal, save_plane_normal_path,
                               num_arrow_pts=200, normal_length=0.5)

        # background mesh thicken direction is opposite to ground normal (downward)
        direction = -normal
        print(f"[Info] plane normal = {normal}, thicken dir = {direction}")

        # since we are generating a mesh with every pixel as a vertex, the vertices might be too many for large images
        # perform mesh simplification if needed
        simplify_step = cfg["bg_mesh_simplify_step"]
        target_faces = cfg["bg_mesh_target_faces"]
        pmap_s, H2, W2 = simplify_heightmap_pmap(
            pmap, step=(simplify_step if simplify_step and simplify_step > 1 else None),
            target_faces=(target_faces if target_faces and target_faces > 0 else None)
        )
        if (H2, W2) != (H, W):
            orig_F = 2 * (H - 1) * (W - 1)
            new_F  = 2 * (H2 - 1) * (W2 - 1)
            print(f"[i] simplified grid: ({H},{W}) -> ({H2},{W2}) "
                  f"faces ~ {orig_F} -> {new_F}")

        # construct mesh
        verts = pmap_s[..., :3].reshape(-1, 3)
        faces = []
        for i in range(H2 - 1):
            base = i * W2
            for j in range(W2 - 1):
                faces += [[base + j, base + j + 1, base + j + W2],
                          [base + j + 1, base + j + W2 + 1, base + j + W2]]
        faces = np.asarray(faces, dtype=np.int32)

        uu, vv = np.meshgrid(np.linspace(0, 1, W2, dtype=np.float32),
                             np.linspace(1, 0, H2, dtype=np.float32))
        uv = np.stack([uu, vv], -1).reshape(-1, 2)

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv, image=img)

        thickness = cfg["bg_mesh_thickness"]
        mesh_thick = add_thickness_below_mesh_preserve_texture(mesh, thickness, direction)

        mesh_path = recon_dir / "background_textured_mesh.glb"
        mesh_thick.export(mesh_path)
        visualize_mesh_thickness_with_pointmap(recon_dir, mesh_thick, pmap[..., :3])
        print(f"[Info] [{key}] background textured mesh saved: {mesh_path}")

        # update scene_dict
        scene_dict["recon"]["bg_mesh_path"] = str(mesh_path)
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
    from utils.compose_config import compose_configs
    key_cfgs = {key: compose_configs(key, cfg) for key in keys}
    key_scene_dicts = {}
    for key in keys:
        scene_pkl = base_dir / f'outputs/{key}/scene/scene.pkl'
        with open(scene_pkl, 'rb') as f:
            scene_dict = pickle.load(f)
        key_scene_dicts[key] = scene_dict

    background_mesh_generation(keys, key_scene_dicts, key_cfgs)
