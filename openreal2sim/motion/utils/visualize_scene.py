#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aggregate (all frames) export for object trajectories in the world frame.

Outputs:
- One PLY containing:
  * Background point cloud (downsampled once, light gray)
  * For each frame (with stride) and each object:
      - Object point cloud transformed by Δ_w[frame]
      - Object center point at that frame (brighter color)
  Optionally encodes time in alpha channel (RGBA): early=dark, late=bright.

- (Optional) One GLB containing all frames at once (can be large):
  * Background mesh once
  * For each frame (with stride), one duplicated and transformed mesh per object.
"""

import json
from pathlib import Path
import argparse
import numpy as np
import trimesh
import yaml


def _create_arrow_mesh(origin: np.ndarray, direction: np.ndarray, length: float, radius: float = 0.01) -> trimesh.Trimesh:
    """Construct an arrow mesh even if `trimesh.creation.arrow` is missing."""
    direction = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        raise ValueError("Grasp direction must be non-zero.")
    direction = direction / norm
    shaft_length = length * 0.75
    tip_length = length - shaft_length

    shaft = trimesh.creation.cylinder(radius=radius, height=shaft_length, sections=18)
    shaft.apply_translation([0.0, 0.0, shaft_length / 2.0])

    tip = trimesh.creation.cone(radius=radius * 1.6, height=tip_length, sections=18)
    tip.apply_translation([0.0, 0.0, shaft_length + tip_length])

    arrow = trimesh.util.concatenate([shaft, tip])
  
    align_tf = trimesh.geometry.align_vectors(np.array([0.0, 0.0, 1.0]), direction)
    arrow.apply_transform(align_tf)
    arrow.apply_translation(origin)
    return arrow


# ---------------- utils ----------------
def load_scene(scene_json: Path):
    scene = json.loads(Path(scene_json).read_text())
    bg_path = Path(scene["background"]["registered"])
    objs = scene["objects"]
    return scene, bg_path, objs

def sample_surface_points(mesh: trimesh.Trimesh, n_pts: int) -> np.ndarray:
    """
    Return (N,3) uniformly sampled surface points.
    Prefer sample_surface_even; fallback to sample_surface.
    """
    if n_pts <= 0:
        return mesh.vertices.view(np.ndarray).copy()
    try:
        from trimesh.sample import sample_surface_even
        pts, _ = sample_surface_even(mesh, n_pts)
    except Exception:
        from trimesh.sample import sample_surface
        pts, _ = sample_surface(mesh, n_pts)
    return pts.astype(np.float64, copy=False)

def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    """
    Simple voxel downsampling by snapping to a voxel grid and
    keeping the first sample per voxel.
    """
    if voxel is None or voxel <= 0 or len(points) == 0:
        return points
    grid = np.floor(points / float(voxel)).astype(np.int64)
    _, keep_idx = np.unique(grid, axis=0, return_index=True)
    keep_idx.sort()
    return points[keep_idx]

def transform_points(D: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Row-vector transform: p' = p * R^T + t"""
    R = D[:3, :3]
    t = D[:3, 3]
    return pts @ R.T + t

def transform_point(D: np.ndarray, p: np.ndarray) -> np.ndarray:
    R = D[:3, :3]
    t = D[:3, 3]
    return p @ R.T + t

def color_palette(k: int) -> np.ndarray:
    base = np.array([
        [230, 57,  70],
        [29,  161, 242],
        [39,  174, 96],
        [155, 89,  182],
        [241, 196, 15],
        [230, 126, 34],
        [52,  152, 219],
        [46,  204, 113],
        [243, 156, 18],
        [231, 76,  60],
    ], dtype=np.uint8)
    if k <= base.shape[0]:
        return base[:k]
    reps = int(np.ceil(k / base.shape[0]))
    pal = np.vstack([base for _ in range(reps)])[:k]
    return pal

def save_scene_glb_all(bg_mesh, per_frame_meshes, out_path: Path):
    """
    Save a single GLB with background once and all per-frame object mesh copies.
    per_frame_meshes: List[(node_name, trimesh.Trimesh)] for all frames.
    """
    sc = trimesh.Scene()
    sc.add_geometry(bg_mesh, node_name="background")
    for name, m in per_frame_meshes:
        sc.add_geometry(m, node_name=name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sc.export(str(out_path), file_type="glb")

# ---------------- core: single-key visualization (callable) ----------------
def visualize_single_scene(
    key: str,
    out_dir: Path | None = None,          # None -> outputs/<key>/debug
    *,
    bg_samples: int = 60000,              # default background sample count
    obj_samples: int = 12000,             # default object sample count
    bg_voxel: float = 0.01,               # meters
    obj_voxel: float = 0.005,             # meters
    stride: int = 1,
    save_glb_all: bool = False,
    encode_time_in_alpha: bool = False,
    mesh_key: str = "optimized",
    traj_key: str = "final_trajs",
) -> dict:
    """
    Render a single combined PLY (and optional GLB) for one key.

    Args:
        key: scene key under outputs/<key>/simulation/scene.json
        out_dir: if None -> outputs/<key>/debug ; otherwise results write to <out_dir>/<key>/
        bg_samples, obj_samples: sample counts
        bg_voxel, obj_voxel: voxel sizes in meters
        stride: frame stride
        save_glb_all: also export a GLB with all frames (heavy)
        encode_time_in_alpha: encode time as alpha in PLY colors
        mesh_key: which object mesh field to use (default "fdpose"; also "refined")
        traj_key: which trajectory field to use (default "trajs"; e.g., "trajs_hybrid")

    Returns:
        dict: {"ply": <Path>, "glb": <Path or None>, "frames": int, "points": int}
    """
    base = Path.cwd()
    scene_json = base / "outputs" / key / "simulation" / "scene.json"
    out_dir = base / "outputs" / key / "motion" / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)

    scene, bg_path, objs = load_scene(scene_json)
    assert bg_path.exists(), f"[{key}] background_aligned.glb not found: {bg_path}"
    if len(objs.keys()) == 0:
        raise RuntimeError(f"[{key}] scene.objects is empty")

    print(f"[{key}] load background: {bg_path}")
    bg_mesh = trimesh.load(str(bg_path), force='mesh')

    # --- background point cloud (once) ---
    bg_pts = sample_surface_points(bg_mesh, bg_samples)
    bg_pts = voxel_downsample(bg_pts, bg_voxel)
    bg_color = np.array([180, 180, 180], dtype=np.uint8)

    # --- preload all objects ---
    palette = color_palette(len(objs))
    seq_lengths = []
    obj_infos = []
    for oid, obj in objs.items():
        oid = obj["oid"]
        oname = obj["name"]
        name = f"{oid}_{oname}"

        # 1) mesh field (e.g., fdpose or refined)
        mesh_path = Path(obj.get(mesh_key, ""))
        assert mesh_path.exists(), f"[{key}] object mesh '{mesh_key}' not found: {mesh_path}"

        # 2) trajectory field (e.g., trajs or trajs_hybrid)
        trajs_path = Path(obj.get(traj_key, ""))
        if not trajs_path.exists():
            traj_key = "fdpose_trajs"
            trajs_path = Path(obj.get(traj_key, ""))
            rel = np.load(str(trajs_path))
            N = rel.shape[0]
            seq_lengths.append(N)
            rel = np.stack([np.eye(4) for _ in range(N)])
        else:
            rel = np.load(str(trajs_path))  # [N,4,4]
            assert rel.ndim == 3 and rel.shape[1:] == (4,4), f"[{key}] trajs shape invalid: {rel.shape}"
            N = rel.shape[0]
            seq_lengths.append(N)
        

        print(f"[{key}] load object {name}: mesh={mesh_path.name} ({mesh_key}), trajs={trajs_path.name} ({traj_key})")
        base_mesh = trimesh.load(str(mesh_path), force='mesh')  # world-space at frame 0


        obj_pts0 = sample_surface_points(base_mesh, obj_samples)
        obj_pts0 = voxel_downsample(obj_pts0, obj_voxel)
        center0 = base_mesh.centroid.view(np.ndarray)

        obj_infos.append({
            "name": name,
            "base_mesh": base_mesh,
            "rel": rel.astype(np.float64),
            "pts0": obj_pts0.astype(np.float64),
            "center0": center0.astype(np.float64),
            "color": palette[0].astype(np.uint8)
        })
        print(f"[{key}] object {name}: traj_len={N}, pts0={len(obj_pts0)}")

    # unify sequence length by the shortest and apply stride
    N_all = min(seq_lengths)
    idxs = np.arange(0, N_all, max(1, int(stride)))
    n_frames = len(idxs)
    print(f"[{key}] aggregating {n_frames} frames into ONE PLY (and optional ONE GLB)")

    # --- aggregate PLY ---
    pts_all_list = []
    cols_all_list = []

    # add background first
    pts_all_list.append(bg_pts)
    if encode_time_in_alpha:
        alpha_bg = np.full((len(bg_pts), 1), 255, dtype=np.uint8)  # background alpha = 255
        cols_all_list.append(np.hstack([np.repeat(bg_color[None, :], len(bg_pts), axis=0), alpha_bg]))
    else:
        cols_all_list.append(np.repeat(bg_color[None, :], len(bg_pts), axis=0))

    # optional GLB: collect per-frame object mesh copies
    all_mesh_copies = []  # (name, mesh)

    for k, i in enumerate(idxs):
        # time alpha from 64 -> 255
        if encode_time_in_alpha:
            alpha_val = int(np.clip(64 + (191 * (k / max(1, n_frames - 1))), 64, 255))
        for info in obj_infos:
            Di = info["rel"][i]
            # object cloud
            pts_i = transform_points(Di, info["pts0"])
            pts_all_list.append(pts_i)

            if encode_time_in_alpha:
                alpha = np.full((len(pts_i), 1), alpha_val, dtype=np.uint8)
                cols = np.hstack([np.repeat(info["color"][None, :], len(pts_i), axis=0), alpha])
            else:
                cols = np.repeat(info["color"][None, :], len(pts_i), axis=0)
            cols_all_list.append(cols)

            # object center (brighter)
            c_i = transform_point(Di, info["center0"])
            pts_all_list.append(c_i.reshape(1, 3))
            ctr_col = np.clip(info["color"].astype(int) + 50, 0, 255).astype(np.uint8)
            if encode_time_in_alpha:
                cols_all_list.append(np.array([[ctr_col[0], ctr_col[1], ctr_col[2], alpha_val]], dtype=np.uint8))
            else:
                cols_all_list.append(ctr_col.reshape(1, 3).astype(np.uint8))

            # GLB accumulation (optional)
            if save_glb_all:
                m = info["base_mesh"].copy()
                m.apply_transform(Di)
                all_mesh_copies.append((f"{info['name']}_f{i:06d}", m))


    # 处理抓取点（grasp_point），生成球体mesh，并按红色加入到 ply 和 glb 中
    manipulated_oid = scene["manipulated_oid"]
    grasp_point = np.array(scene["objects"][manipulated_oid]["grasp_point"])
    grasp_point_radius = 0.01  # 球体半径可调

    # 创建抓取点球体mesh，并移动到抓取点位置
    grasp_point_mesh = trimesh.creation.icosphere(subdivisions=3, radius=grasp_point_radius)
    grasp_point_mesh.apply_translation(grasp_point)


    grasp_color = np.array([0, 0, 255], dtype=np.uint8)  # 蓝色
    grasp_pts = grasp_point_mesh.vertices
    grasp_cols = np.tile(grasp_color, (grasp_pts.shape[0], 1))
    if encode_time_in_alpha:
        grasp_cols = np.hstack([grasp_cols, np.full((grasp_pts.shape[0], 1), 255, dtype=np.uint8)])
    pts_all_list.append(grasp_pts)
    cols_all_list.append(grasp_cols)

    grasp_direction = np.array(scene["objects"][manipulated_oid]["grasp_direction"], dtype=np.float32)
    if grasp_direction.shape == (3, 3):
        grasp_direction = grasp_direction[:, 0]
    grasp_direction = grasp_direction.reshape(-1)
    if grasp_direction.shape[0] != 3 or np.linalg.norm(grasp_direction) < 1e-6:
        print(f"[WARN][{key}] Invalid grasp_direction recorded; defaulting to +Z.")
        grasp_direction = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        grasp_direction = grasp_direction / np.linalg.norm(grasp_direction)
    grasp_direction_length = 0.1  # 箭头长度
    grasp_direction_mesh = _create_arrow_mesh(grasp_point, grasp_direction, grasp_direction_length, radius=0.01)
    grasp_direction_pts = grasp_direction_mesh.vertices
    grasp_direction_cols = np.tile(grasp_color, (grasp_direction_pts.shape[0], 1))
    if encode_time_in_alpha:
        grasp_direction_cols = np.hstack([grasp_direction_cols, np.full((grasp_direction_pts.shape[0], 1), 255, dtype=np.uint8)])
    pts_all_list.append(grasp_direction_pts)
    cols_all_list.append(grasp_direction_cols)

    # GLB文件也加入mesh
    if save_glb_all:
        all_mesh_copies.append(("grasp_point", grasp_point_mesh))

    # 重新聚合所有点和颜色再写PLY
    pts_all = np.vstack(pts_all_list)
    cols_all = np.vstack(cols_all_list)

    ply_path = out_dir / f"object_{traj_key}.ply"
    point_cloud = trimesh.points.PointCloud(vertices=pts_all, colors=cols_all)
    point_cloud.export(str(ply_path))
    print(f"[{key}][PLY] aggregated: {ply_path}  (pts={pts_all.shape[0]:,})")

    
    # write single GLB (optional)
    glb_path = None
    if save_glb_all:
        glb_path = out_dir / "scene_all_frames.glb"
        save_scene_glb_all(bg_mesh, all_mesh_copies, glb_path)
        print(f"[{key}][GLB] aggregated: {glb_path}  (meshes={len(all_mesh_copies) + 1:,} incl. background)")

    return {"ply": ply_path, "glb": glb_path, "frames": int(n_frames), "points": int(len(pts_all))}

def visualize_scene(keys):
    for key in keys:
        visualize_single_scene(key)
        print(f"[done] key={key}")
    print('[Info] Scene visualization completed.')

# ---------------- main (batch over config keys) ----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=5, help="frame stride (1 = all frames)")
    ap.add_argument("--save_glb_all", action="store_true", help="export a single GLB containing all frames")
    ap.add_argument("--encode_time_in_alpha", action="store_true", help="encode time as alpha in PLY colors")
    ap.add_argument("--mesh_key", type=str, default="optimized", help="object mesh field name (e.g., fdpose or optimized)")
    ap.add_argument("--traj_key", type=str, default="hybrid_trajs", help="trajectory field name (e.g., fdpose_trajs/simple_trajs/hybrid_trajs)")
    return ap.parse_args()

def main():
    args = parse_args()
    base = Path.cwd()
    cfg = yaml.safe_load((base / "config" / "config.yaml").open("r"))
    keys = cfg["keys"]

    for key in keys:
        print(f"========== [visualize object trajectories] key: {key} ==========\n")
        visualize_single_scene(
            key=key,
            out_dir=None,                      # default -> outputs/<key>/debug
            bg_samples=60000,
            obj_samples=12000,
            bg_voxel=0.01,
            obj_voxel=0.005,
            stride=args.stride,
            save_glb_all=args.save_glb_all,
            encode_time_in_alpha=args.encode_time_in_alpha,
            mesh_key=args.mesh_key,
            traj_key=args.traj_key,
        )
        print(f"[done] key={key}")

if __name__ == "__main__":
    main()
