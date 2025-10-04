#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimize object placement so the objects and the background mesh have no inter-penetration (essential for stable simulation). 
Inputs:
    - outputs/{key_name}/scene/scene.pkl
Outputs:
    - outputs/{key_name}/scene/scene.pkl (updated)
Note:
    - Updated keys in "info":{
        "objects": { 
            "oid": {
                ...
                "optimized":      # object placement after collision optimization [glb],
            },
            ...
        },
        "scene_mesh": {
            ...
            "scene_optimized": # entire scene with collision optimized object meshes [glb],
        }
    }
"""
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from tqdm import trange
import yaml
import pickle

# ──────────────── utils ──────────────── #
def trimesh_single(path):
    m = trimesh.load(path, force="mesh")
    if isinstance(m, trimesh.Scene):
        m = list(m.geometry.values())[0]
    return m

def ensure_unit_normal(n):
    n = np.asarray(n, dtype=np.float64)
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        raise ValueError("Plane normal has near-zero length.")
    return n / norm

# ──────────────── Kaolin SDF ──────────────── #
@torch.no_grad()
def build_sdf_kaolin(mesh_path, res=192, margin=0.02, device="cuda"):
    import kaolin as kal
    ext = os.path.splitext(mesh_path)[1].lower()
    if ext in ['.glb', '.gltf']:
        mesh = kal.io.gltf.import_mesh(mesh_path)
    elif ext == '.obj':
        mesh = kal.io.obj.import_mesh(mesh_path)
    elif ext == '.ply':
        mesh = kal.io.ply.import_mesh(mesh_path)
    else:
        m = trimesh.load(mesh_path, force='mesh')
        mesh = kal.rep.SurfaceMesh(
            vertices=torch.as_tensor(np.asarray(m.vertices), device=device),
            faces=torch.as_tensor(np.asarray(m.faces), device=device)
        )
    verts = mesh.vertices.to(device)
    faces = mesh.faces.long().to(device)

    vmin = verts.min(0).values - margin
    vmax = verts.max(0).values + margin
    xs = torch.linspace(vmin[0], vmax[0], res, device=device)
    ys = torch.linspace(vmin[1], vmax[1], res, device=device)
    zs = torch.linspace(vmin[2], vmax[2], res, device=device)
    gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing='ij')
    samples = torch.stack([gx, gy, gz], -1).reshape(-1, 3)

    face_v = mesh.face_vertices.unsqueeze(0).to(device)
    unsigned2, _, _ = kal.metrics.trianglemesh.point_to_mesh_distance(
        samples.unsqueeze(0), face_v)
    unsigned = unsigned2.squeeze(0).sqrt()
    inside = kal.ops.mesh.check_sign(
        verts.unsqueeze(0), faces, samples.unsqueeze(0)
    ).squeeze(0)
    sign = torch.where(inside, -1.0, 1.0)

    sdf = (unsigned * sign).reshape(res, res, res).unsqueeze(0).unsqueeze(0)
    voxel = (vmax - vmin) / (res - 1)
    return sdf, vmin, voxel

# ──────────────── SDF query ──────────────── #
def sdf_query(vol, origin, vox, pts):
    norm = (pts - origin) / (vox * (torch.tensor(vol.shape[-3:], device=pts.device) - 1)) * 2 - 1
    grid = norm.view(1, -1, 1, 1, 3).clamp(-1, 1)
    return F.grid_sample(vol, grid, align_corners=True, padding_mode='border')[0, 0, :, 0, 0]

# ──────────────── Z-only SDF refine ──────────────── #
def optimise_sdf_z(mesh_path, sdf_pack,
                   n_sample=5000, lr=2e-3,
                   n_iter=4000, max_step=1e-3,
                   clearance=0.004):
    vol, origin, vox = sdf_pack
    device = vol.device
    mesh = trimesh_single(mesh_path)
    V = torch.as_tensor(mesh.vertices, dtype=torch.float32, device=device)

    samp, _ = mesh.sample(min(n_sample, len(mesh.vertices)), return_index=True)
    samp = torch.as_tensor(samp, dtype=torch.float32, device=device)

    tz = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.Adam([tz], lr=lr)

    for it in trange(n_iter, leave=False):
        pts = samp + torch.tensor([0.0, 0.0, 1.0], device=device) * tz
        sdf_val = sdf_query(vol, origin, vox, pts)
        pen = F.relu(-sdf_val).pow(2).mean()
        if pen.item() == 0.0:
            print(f"Early stop at iter {it}, penetration loss=0.")
            break
        opt.zero_grad()
        pen.backward()
        with torch.no_grad():
            tz.grad.clamp_(-max_step, max_step)
        opt.step()

    moved = V + torch.tensor([0.0, 0.0, 1.0], device=device) * tz.detach()
    sdf_vals = sdf_query(vol, origin, vox, moved)
    min_dist = sdf_vals.min().item()
    if min_dist < clearance:
        dz_extra = clearance - min_dist
        print(f"Applying extra clearance dz={dz_extra:.4f}")
        moved[:, 2] += dz_extra

    mesh.vertices = moved.cpu().numpy()
    return mesh

# ──────────────── Plane-based lift ──────────────── #
def refine_by_plane_clearance(mesh_path, plane_point, plane_normal, clearance=0.004):
    """
    Move the entire mesh along plane_normal so that all vertices satisfy:
        n · (v - p0) >= clearance
    """
    p0 = np.asarray(plane_point, dtype=np.float64)
    n  = ensure_unit_normal(plane_normal)

    mesh = trimesh_single(mesh_path)
    V = np.asarray(mesh.vertices, dtype=np.float64)

    dists = (V - p0) @ n
    d_min = float(dists.min())
    if d_min < clearance:
        delta = (clearance - d_min) * n
        V = V + delta
        mesh.vertices = V
        print(f"lifted by {np.linalg.norm(delta):.6f} along normal")
    else:
        print("already above plane with required clearance")
    return mesh

# ──────────────── main function ──────────────── #
def scenario_collision_optimization(keys, key_scene_dicts, key_cfgs):

    for key in keys:
        scene_dict = key_scene_dicts[key]
        key_cfg = key_cfgs[key]
        mode = key_cfg["collision_optimization_mode"]
        clearance = key_cfg["collision_clearance"]
        sdf_res = key_cfg["collision_sdf_resolution"]  # default 192

        print(f"[Info] Processing {key} with mode={mode}, clearance={clearance}...\n")
        
        if mode == "sdf":
            print(f"[Info]: [{key}] Building Kaolin SDF...")
            background_mesh_path = scene_dict["info"]["background"]['registered']
            sdf_t, org_t, vox_t = build_sdf_kaolin(background_mesh_path, res=sdf_res)
            sdf_t, org_t, vox_t = sdf_t.cuda(), org_t.cuda(), vox_t.cuda()
        elif mode == "plane":
            plane_sim = scene_dict["info"]["groundplane_in_sim"]
            p0 = plane_sim["point"]
            n  = plane_sim["normal"]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # per-object refine
        for oid, obj in scene_dict["info"]["objects"].items():
            name = obj["name"]
            print(f"[Info] [{key}] Refining {oid}_{name}...")
            in_path = obj["fdpose"]

            if mode == "sdf":
                refined = optimise_sdf_z(in_path, (sdf_t, org_t, vox_t), clearance=clearance)
            elif mode == "plane":
                refined = refine_by_plane_clearance(in_path, p0, n, clearance=clearance)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            out_path = Path(in_path).parent / f"{oid}_{name}_optimized.glb"
            refined.export(str(out_path))
            obj["optimized"] = str(out_path)

            # update AABB stats
            bounds = trimesh_single(str(out_path)).bounds
            obj["object_min"]    = bounds[0].tolist()
            obj["object_max"]    = bounds[1].tolist()
            obj["object_center"] = (0.5 * (bounds[0] + bounds[1])).tolist()
            print(f"[Info] [{key}] collision optimized object is saved to {out_path}")

            # update scene_dict
            scene_dict["info"]["objects"][oid] = obj

        # merge & save
        scene = trimesh.Scene()
        scene.add_geometry(trimesh_single(scene_dict["info"]["background"]["registered"]), 'background')
        for oid, obj in scene_dict["info"]["objects"].items():
            name = obj["name"]
            scene.add_geometry(trimesh_single(obj["optimized"]), f"obj{oid}_{name}")
        merged = base_dir / Path(f"outputs/{key}/reconstruction/scenario") / "scene_optimized.glb"
        scene.export(str(merged))
        scene_dict["info"]["scene_mesh"]["scene_optimized"] = str(merged)
        key_scene_dicts[key] = scene_dict
        with open(base_dir / f'outputs/{key}/scene/scene.pkl', 'wb') as f:
            pickle.dump(scene_dict, f)
    
    return key_scene_dicts

# ──────────────── main ──────────────── #
if __name__ == '__main__':
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
    scenario_collision_optimization(keys, key_scene_dicts, key_cfgs)
