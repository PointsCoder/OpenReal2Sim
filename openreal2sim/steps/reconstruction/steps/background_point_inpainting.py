#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Complete point-cloud geometry for inpainted background images.
Since the background pixels are inpainted, we also need to inpaint the geomtry in the masked region.

Opt-in: the step runs only when the reconstruction config sets bg_point_inpainting, because
background_mesh_generation fills the object region with the ground plane itself. Turn it on
for bg_completion_mode "moge", the monocular depth prediction, which is the one completion
this step can do and the mesh step cannot.
Inputs:
    - outputs/{key}/scene/scene.pkl (must contain a "recon" key containing inpainting results)
Outputs:
    - outputs/{key}/scene/scene.pkl (updated with completed point-cloud geometry)
    - outputs/{key}/reconstruction/background_points.ply (background complete point cloud)
    - outputs/{key}/reconstruction/foreground_points.ply (foreground point cloud)
Note:
    - added keys in "recon": "bg_depth", "bg_pts", "fg_pts"
"""

from pathlib import Path
from typing import Tuple
import pickle
import numpy as np
import open3d as o3d
import torch
import tqdm
import yaml

from openreal2sim.steps.reconstruction.utils.table_slab import (
    color_pointmap, dilate_mask, plane_fill)


base_dir = Path.cwd()

# ─────────────────────────── Depth prediction  ──────────────────────────
def run_moge_depth(img_rgb: np.ndarray, device: torch.device) -> np.ndarray:
    """MoGe-2 inference (unchanged). Returns float32 depth (H,W)."""
    from moge.model.v2 import MoGeModel
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
        bg_color_pts = color_pointmap(depth_bg, K, bg_img)  # (H, W, 6) xyzrgb, rgb in [0,1]
        fg_color_pts = color_pointmap(depth0, K, fg_img)
        scene_dict["recon"]["bg_pts"] = bg_color_pts
        scene_dict["recon"]["fg_pts"] = fg_color_pts
        key_scene_dicts[key] = scene_dict
        with open(base_dir / f'outputs/{key}/scene/scene.pkl', 'wb') as f:
            pickle.dump(scene_dict, f)

        # exports
        export_cloud(fg_color_pts[..., :3], fg_color_pts[..., 3:6], recon_dir / "foreground_points.ply")
        export_cloud(bg_color_pts[..., :3], bg_color_pts[..., 3:6], recon_dir / "background_points.ply")
        print(f"[Info] [{key}] geometry inpainting done.\n")
    return key_scene_dicts

if __name__ == "__main__":
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
    background_point_inpainting(keys, key_scene_dicts, key_cfgs)
