#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Complete point-cloud geometry for inpainted background images.
Since the background pixels are inpainted, we also need to inpaint the geomtry in the masked region. 
Inputs:
    - outputs/{key_name}/scene/scene.pkl (must contain a "recon" key containing inpainting results)
Outputs:
    - outputs/{key_name}/scene/scene.pkl (updated with completed point-cloud geometry)
    - outputs/{key_name}/reconstruction/background_points.ply (background complete point cloud)
    - outputs/{key_name}/reconstruction/foreground_points.ply (foreground point cloud)
Note:
    - added keys in "recon": "bg_depth", "fg_depth", "bg_pts", "fg_pts"
"""

import json
from pathlib import Path
from typing import Tuple
import pickle
import cv2
import numpy as np
import open3d as o3d
import torch
import tqdm
import yaml

from moge.model.v2 import MoGeModel


base_dir = Path.cwd()

def dilate_mask(binary_mask: np.ndarray, pixels: int = 4, shape: str = "ellipse") -> np.ndarray:
    """Dilate a boolean mask outward to stabilize boundary pixels near object edges."""
    if pixels == 0:
        return binary_mask
    shape_map = {
        "ellipse": cv2.MORPH_ELLIPSE,
        "rect":    cv2.MORPH_RECT,
        "cross":   cv2.MORPH_CROSS,
    }
    k = cv2.getStructuringElement(shape_map.get(shape, cv2.MORPH_ELLIPSE),
                                  (2 * pixels + 1, 2 * pixels + 1))
    out = cv2.dilate(binary_mask.astype(np.uint8), k, iterations=1)
    return out.astype(bool)

# ─────────────────────────── Depth prediction  ──────────────────────────
def run_moge_depth(img_rgb: np.ndarray, device: torch.device) -> np.ndarray:
    """MoGe-2 inference (unchanged). Returns float32 depth (H,W)."""
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device).eval()
    with torch.no_grad():
        inp = torch.tensor(img_rgb/255.0, dtype=torch.float32,
                           device=device).permute(2,0,1).unsqueeze(0)
        depth = model.infer(inp[0])["depth"].detach().cpu().numpy()
    return depth

# ─────────────────────────── Depth alignment  ──────────────────────────
def huber_weights(residuals: np.ndarray, delta: float) -> np.ndarray:
    """Huber weights: w = 1 if |r| <= d else d/|r|."""
    abs_r = np.abs(residuals)
    w = np.ones_like(residuals, dtype=np.float64)
    mask = abs_r > delta
    w[mask] = (delta / (abs_r[mask] + 1e-12))
    return w

def robust_scale_shift_align(
    pred_depth: np.ndarray,
    ref_depth: np.ndarray,
    mask: np.ndarray,
    iters: int = 5,
    huber_delta: float = 0.02
) -> Tuple[float, float]:
    """
    Solve for a,b in:  a * pred_depth + b ≈ ref_depth, on the masked region.
    Uses Iteratively Reweighted Least Squares with Huber weights.
    """
    assert pred_depth.shape == ref_depth.shape == mask.shape
    valid = (mask > 0) & np.isfinite(pred_depth) & np.isfinite(ref_depth) & (pred_depth > 0) & (ref_depth > 0)
    if valid.sum() < 100:
        ratio = np.median(ref_depth[valid]) / (np.median(pred_depth[valid]) + 1e-12)
        return float(ratio), 0.0

    x = pred_depth[valid].astype(np.float64)
    y = ref_depth[valid].astype(np.float64)

    A = np.stack([x, np.ones_like(x)], axis=1)
    w = np.ones_like(x, dtype=np.float64)

    for _ in range(iters):
        Aw = A * np.sqrt(w[:, None])
        yw = y * np.sqrt(w)
        params, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
        a, b = params[0], params[1]
        r = (A @ params) - y
        med = np.median(r)
        mad = np.median(np.abs(r - med)) + 1e-12
        sigma = 1.4826 * mad
        delta = huber_delta if sigma < 1e-12 else huber_delta * sigma / max(sigma, 1e-12)
        w = huber_weights(r - med, delta)

    return float(a), float(b)

def depth_to_points(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    H, W = depth.shape
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    ii, jj = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    z = depth.reshape(-1)
    x = (ii.reshape(-1) - cx) * z / fx
    y = (jj.reshape(-1) - cy) * z / fy
    return np.stack([x, y, z], 1)

def plane_fill(depth0, K, ground, obj):
    H,W=depth0.shape; fx,fy,cx,cy=K[0,0],K[1,1],K[0,2],K[1,2]

    # fit plane on ground region
    pts_ground = depth_to_points(depth0, K)[ground.ravel()]
    pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(pts_ground)
    plane,_ = pc.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    a,b,c,d = plane

    # intersect rays with plane inside the object mask
    ys,xs = np.where(obj)
    dx = (xs - cx) / fx; dy = (ys - cy) / fy; dz = np.ones_like(dx)
    denom = a*dx + b*dy + c*dz
    valid = np.abs(denom) > 1e-6
    t = np.full_like(dx, np.nan, dtype=np.float32)
    t[valid] = -d / denom[valid]
    valid &= (t > 0) & np.isfinite(t)
    z_new = t[valid] * dz[valid]

    depth0_filled = depth0.copy()
    depth0_filled[ys[valid], xs[valid]] = np.clip(z_new, 0, depth0.max())
    return depth0_filled


def export_cloud(pts: np.ndarray, colors: np.ndarray, out_path: Path):
    pts = pts.reshape(-1,3)
    colors = colors.reshape(-1,3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(out_path), pcd)

def background_point_inpainting(keys, key_scene_dicts, key_cfgs):

    for key in tqdm.tqdm(keys):
        print(f"[Info] Processing {key}...\n")
        scene_dict = key_scene_dicts[key]
        key_cfg = key_cfgs[key]
        recon_dir = base_dir / "outputs" / key / "reconstruction"
        recon_dir.mkdir(parents=True, exist_ok=True)

        if "recon" not in scene_dict:
            print(f"[Warning] [{key}] No 'recon' key found in scene.pkl; run background_pixel_inpainting first.")
            continue
        if "background" not in scene_dict["recon"] or "foreground" not in scene_dict["recon"]:
            print(f"[Warning] [{key}] 'recon' key missing 'background' or 'foreground'; run background_pixel_inpainting first.")
            continue
        if "ground_mask" not in scene_dict["recon"] or "object_mask" not in scene_dict["recon"]:
            print(f"[Warning] [{key}] 'recon' key missing 'ground_mask' or 'object_mask'; run background_pixel_inpainting first.")
            continue

        ground_mask = scene_dict["recon"]["ground_mask"]  # H x W, bool
        object_mask   = scene_dict["recon"]["object_mask"]  # H x W, bool
         # dilate the masks a bit more to remove boundary outliers
        object_mask = dilate_mask(object_mask, pixels=key_cfg["obj_dilate_pixels"], shape="ellipse") 
        ground_mask = dilate_mask(ground_mask, pixels=key_cfg["ground_dilate_pixels"], shape="ellipse")
        fg_img = scene_dict["recon"]["foreground"]  # H x W x 3, uint8
        bg_img = scene_dict["recon"]["background"]  # H x W x 3, uint8
        depth0 = scene_dict["depths"][0]  # H x W, float32
        K      = scene_dict["intrinsics"].astype(np.float32)  # 3 x 3, float32
        H, W = depth0.shape

        completion_mode = key_cfg["bg_completion_mode"]
        if completion_mode == "plane":
            # assume the masked region is the ground, and using ground plane for geometry completion
            depth_bg = plane_fill(depth0, K, ground_mask, object_mask)
        else:
            # use monocular depth prediction for geometry completion
            print(f"[Info] [{key}] predicting depth with MoGe-2 ...")
            gpu_id = key_cfg["gpu"]
            device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
            depth_pd = run_moge_depth(bg_img, device)

            # align the depth scale of current bg depth prediction and original image depth prediction
            valid_mask = (~object_mask) # align using non-masked region 
            a, b = robust_scale_shift_align(
                pred_depth=depth_pd,
                ref_depth=depth0,
                mask=valid_mask.astype(np.uint8),
                iters=5,
                huber_delta=0.02
            )
            depth_align = (a * depth_pd + b).astype(np.float32)
            depth_bg = depth0.copy()
            depth_bg[object_mask] = depth_align[object_mask]

        scene_dict["recon"]["bg_depth"] = depth_bg.astype(np.float32)
        scene_dict["recon"]["fg_depth"] = depth0.astype(np.float32)
        bg_pts = depth_to_points(depth_bg, K)
        bg_colors = bg_img.reshape(-1,3) / 255.
        bg_color_pts = np.concatenate([bg_pts.reshape(H,W,3), bg_colors.reshape(H,W,3)], -1)
        fg_pts = depth_to_points(depth0, K)
        fg_colors = fg_img.reshape(-1,3) / 255.
        fg_color_pts = np.concatenate([fg_pts.reshape(H,W,3), fg_colors.reshape(H,W,3)], -1)
        scene_dict["recon"]["bg_pts"] = bg_color_pts.astype(np.float32) # (H, W, 6) xyzrgb rgb in [0,1]
        scene_dict["recon"]["fg_pts"] = fg_color_pts.astype(np.float32) # (H, W, 6) xyzrgb rgb in [0,1]
        key_scene_dicts[key] = scene_dict
        with open(base_dir / f'outputs/{key}/scene/scene.pkl', 'wb') as f:
            pickle.dump(scene_dict, f)

        # exports
        export_cloud(fg_pts, fg_colors, recon_dir / "foreground_points.ply")
        export_cloud(bg_pts, bg_colors, recon_dir / "background_points.ply")
        print(f"[Info] [{key}] geometry inpainting done.\n")
    return key_scene_dicts

if __name__ == "__main__":
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
    background_point_inpainting(keys, key_scene_dicts, key_cfgs)
