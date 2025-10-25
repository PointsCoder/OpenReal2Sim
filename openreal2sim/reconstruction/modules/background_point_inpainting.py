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

def fill_mask(mask: np.ndarray) -> np.ndarray:
    """
    Fill the mask with the flood fill algorithm.
    """
    mask = mask.astype(np.uint8)
    filled = mask.copy()
    inv_mask = 1 - mask
    h, w = mask.shape
    floodfilled = inv_mask.copy()
    floodfilled_pad = np.pad(floodfilled, 1, mode='constant', constant_values=0)
    cv2.floodFill(floodfilled_pad, None, (0,0), 255)
    floodfilled_nohole = floodfilled_pad[1:-1,1:-1]
    out_mask = ((floodfilled_nohole == 0) | (mask == 1)).astype(bool)
    return out_mask

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
    """
    Convert a depth map to 3D points and exclude pixels with zero or non-finite depth.
    Returns an (N,3) array containing only valid points (z>0 and finite).
    """
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    ii, jj = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    z = depth.reshape(-1).astype(np.float32)
    x = (ii.reshape(-1) - cx) * z / fx
    y = (jj.reshape(-1) - cy) * z / fy
    pts = np.stack([x, y, z], axis=1)
    valid = (z > 0) & (z < np.inf)
    pts[~valid] = np.nan
    return pts

def handcraft_segment_plane(pts_ground: np.ndarray, distance_threshold: float = 0.02, ransac_n: int = 3, num_iterations: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Handcraft plane segmentation using RANSAC.
    """
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts_ground)
    
    # Fix random number generation issues
    import random
    import os
    try:
        # Try to set multiple random seeds for better entropy
        random.seed(42)
        np.random.seed(42)
        os.environ['PYTHONHASHSEED'] = '42'
        
        # Try RANSAC plane segmentation with error handling
        try:
            plane, _ = pc.segment_plane(distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations)
        except RuntimeError as e:
            if "random_device" in str(e):
                print(f"[Warning] RANSAC failed due to random device issue, using fallback method...")
                # Fallback: Use simple plane fitting without RANSAC
                if len(pts_ground) >= 3:
                    # Use least squares to fit a plane
                    A = np.column_stack([pts_ground[:, 0], pts_ground[:, 1], np.ones(len(pts_ground))])
                    b = pts_ground[:, 2]
                    try:
                        coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
                        a, b, c = coeffs[0], coeffs[1], -1.0
                        d = coeffs[2]
                        plane = [a, b, c, d]
                    except np.linalg.LinAlgError:
                        # Ultimate fallback: use horizontal plane at median depth
                        median_z = np.median(pts_ground[:, 2])
                        plane = [0, 0, 1, -median_z]
                else:
                    # Not enough points, use horizontal plane at median depth
                    median_z = np.median(pts_ground[:, 2])
                    plane = [0, 0, 1, -median_z]
            else:
                raise e
    except Exception as e:
        print(f"[Warning] Plane fitting failed: {e}, using horizontal plane fallback...")
        # Ultimate fallback: horizontal plane at median depth
        median_z = np.median(pts_ground[:, 2])
        plane = [0, 0, 1, -median_z]
    finally:
        # Cleanup point cloud to prevent memory leaks
        del pc
        import gc
        gc.collect()
    
    return plane

def plane_fill(depth0, K, ground, obj):
    fx,fy,cx,cy=K[0,0],K[1,1],K[0,2],K[1,2]

    from scipy.ndimage import distance_transform_edt
    from scipy.interpolate import griddata

    ground = ground & ~obj
    ground_border = (ground > 0).astype(np.uint8)
    dist = distance_transform_edt(ground_border)
    mask_inner = dist >= 1
    ground_filtered = np.zeros_like(ground, dtype=bool)
    ground_filtered[ground & mask_inner] = True
    z = depth0.reshape(-1).astype(np.float32)
    nan_mask = (z <= 0) | np.isnan(z) | np.isinf(z)
    nan_mask = nan_mask.reshape(depth0.shape)
    ground_filtered = ground_filtered & (~nan_mask)
    #import pdb; pdb.set_trace()
    pts_ground = depth_to_points(depth0, K)[ground_filtered.ravel()]
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts_ground)
  
    np.random.seed(42) 
    plane , _= pc.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)
    a, b, c, d = plane

    from scipy.ndimage import binary_dilation

    obj_edge = binary_dilation(obj, iterations=5) 
    candidate_mask = obj_edge & ground

    ys, xs = np.where(candidate_mask)
    #import pdb; pdb.set_trace()
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    dx = (xs - cx) / fx
    dy = (ys - cy) / fy
    dz = np.ones_like(dx)
    denom = a * dx + b * dy + c * dz
    valid = np.abs(denom) > 1e-6
    t = np.full_like(dx, np.nan, dtype=np.float32)
    t[valid] = -d / denom[valid]
    plane_z = t * dz
    dist_to_plane = np.abs(a * (dx * t) + b * (dy * t) + c * (dz * t) + d) / np.linalg.norm([a, b, c])
    plane_dist_th = 0.003
    mask_valid = (t > 0) & np.isfinite(t) & (dist_to_plane < plane_dist_th)
    z_plane = plane_z[mask_valid]
    xs_plane = xs[mask_valid]
    ys_plane = ys[mask_valid]

    fill_mask = obj
    fill_ys, fill_xs = np.where(fill_mask)

    depth0_filled = depth0.copy()
    if len(z_plane) > 0:
        z_fill = griddata(
            (ys_plane, xs_plane),
            z_plane,
            (fill_ys, fill_xs),
            method="linear",
            fill_value=np.mean(z_plane)
        )
        depth0_filled[fill_ys, fill_xs] = np.clip(z_fill, 0, 5 * depth0.max())
    return depth0_filled


def vanilla_plane_fill(depth0, K, ground, obj):
    H,W=depth0.shape; fx,fy,cx,cy=K[0,0],K[1,1],K[0,2],K[1,2]

    pts_ground = depth_to_points(depth0, K)[ground.ravel()]
    
    
    a,b,c,d = handcraft_segment_plane(pts_ground)

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
    
    depth0_filled[ys[valid], xs[valid]] = np.clip(z_new, 0, 5 * depth0.max())
    return depth0_filled


# def get_ground_mask_from_existing_mask(img: np.ndarray, existing_ground_mask: np.ndarray) -> np.ndarray:
#     # Import SAM modules
#     import sys
#     from pathlib import Path
#     ROOT = Path.cwd()
#     THIRD = ROOT / "third_party/Grounded-SAM-2"
#     sys.path.append(str(THIRD))
    
#     from sam2.build_sam import build_sam2
#     from sam2.sam2_image_predictor import SAM2ImagePredictor
#     import torch
    
#     # Initialize Hydra
#     import hydra
#     from omegaconf import DictConfig
    
#     # Initialize SAM model
#     DEV = "cuda" if torch.cuda.is_available() else "cpu"
#     CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
#     CKPT = "third_party/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
    
#     img_predictor = SAM2ImagePredictor(build_sam2(CFG, CKPT))
#     img_predictor.set_image(img)
#     H, W = img.shape[:2]
#     ground_points = []
#     ground_labels = []
#     existing_ground_mask = existing_ground_mask.astype(bool)
#     ground_indices = np.where(existing_ground_mask)
#     if len(ground_indices[0]) > 0:
#         sample_indices = np.random.choice(len(ground_indices[0]), 
#                                         min(20, len(ground_indices[0])), 
#                                         replace=False)
#         for idx in sample_indices:
#             y, x = ground_indices[0][idx], ground_indices[1][idx]
#             ground_points.append([x, y])
#             ground_labels.append(1)
    
#     ground_mask = np.zeros((H, W), dtype=bool)
#     if len(ground_points) > 0:
#         point_coords = np.array(ground_points)
#         point_labels = np.array(ground_labels)
    
#         masks, scores, logits = img_predictor.predict(
#             point_coords=point_coords,
#             point_labels=point_labels,
#             multimask_output=True
#         )
        

#         best_idx = np.argmax(scores)
#         ground_mask = masks[best_idx]
        
#         ground_mask = ground_mask.astype(bool)
    
#     return ground_mask

    

def hybrid_fill(depth0: np.ndarray, fg_img: np.ndarray, bg_img: np.ndarray, K: np.ndarray, 
                           obj_msk: np.ndarray, device: torch.device, 
                           plane_masks: np.ndarray = None) -> np.ndarray:
    """
    Generate depth using hybrid approach: 
    - Ground regions use plane fill
    - Other regions use MoGe depth prediction
    
    Args:
        depth0: Original depth map
        bg_img: Inpainted background image
        K: Camera intrinsic matrix
        obj_msk: Object mask
        device: PyTorch device
        existing_ground_mask: Existing ground mask to sample points from (H, W)
        
    Returns:
        depth_bg: Generated background depth
    """
    print(f"[Info] Starting hybrid depth generation...")
    depth_bg = depth0.copy()
    kernel = np.ones((25, 25), np.uint8)  # prevent boundaries problem.
    obj_msk = cv2.dilate(obj_msk.astype(np.uint8), kernel, iterations=1).astype(bool)
    left_obj_msk = obj_msk.copy()
    for plane_mask in plane_masks:
        if (plane_mask & left_obj_msk).sum() > 0:
            depth_filled = plane_fill(depth0, K, plane_mask, left_obj_msk & plane_mask)
            depth_bg[plane_mask & left_obj_msk] = depth_filled[plane_mask & left_obj_msk]
        left_obj_msk = left_obj_msk & ~plane_mask
    depth_moge = run_moge_depth(bg_img, device)
   
    # # kernel = np.ones((25, 25), np.uint8)  # prevent boundaries problem.
    # # obj_msk = cv2.dilate(obj_msk.astype(np.uint8), kernel, iterations=1).astype(bool)
    # depth_ground = vanilla_plane_fill(depth0, K,existing_ground_mask, obj_msk)

    # depth_bg = depth0.copy()
    # obj_msk =obj_msk.astype(bool)
    # ground_mask = ground_mask.astype(bool)
    
    # obj_ground = obj_msk & ground_mask
    # obj_non_ground = obj_msk & (~ground_mask)

    # if obj_ground.sum() > 0:
    #     depth_bg[obj_ground] = depth_ground[obj_ground]
    
    if left_obj_msk.sum() > 0:
        a, b = robust_scale_shift_align(
                pred_depth=depth_moge,
                ref_depth=depth0,
                mask= ~obj_msk.astype(np.uint8),
                iters=5,
                huber_delta=0.02
            )
        depth_align = (a * depth_moge + b).astype(np.float32)
        depth_bg[left_obj_msk] = depth_align[left_obj_msk]

    return depth_bg


def export_cloud(pts: np.ndarray, colors: np.ndarray, out_path: Path):
    pts = pts.reshape(-1,3)
    colors = colors.reshape(-1,3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(out_path), pcd)

def clear_nan(pts: np.ndarray, colors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    valid_mask = np.isfinite(pts).all(axis=1)
    return pts[valid_mask], colors[valid_mask]

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
        ground_mask = scene_dict["recon"]["ground_mask"]
        plane_masks = scene_dict["recon"]["plane_masks"] if scene_dict["recon"]["plane_masks"] is not None else []
        object_mask   = scene_dict["recon"]["object_mask"]  # H x W, bool
         # dilate the masks a bit more to remove boundary outliers
        object_mask = fill_mask(object_mask)
        ground_mask = dilate_mask(ground_mask, pixels=key_cfg["ground_dilate_pixels"], shape="ellipse")
        fg_img = scene_dict["recon"]["foreground"]  # H x W x 3, uint8
        bg_img = scene_dict["recon"]["background"]  # H x W x 3, uint8
        depth0 = scene_dict["depths"][0]  # H x W, float32
        K      = scene_dict["intrinsics"].astype(np.float32)  # 3 x 3, float32
        H, W = depth0.shape
        gpu_id = key_cfg["gpu"]
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        completion_mode = key_cfg["bg_completion_mode"]
        if completion_mode == "plane":
            # assume the masked region is the ground, and using ground plane for geometry completion
            depth_bg = plane_fill(depth0, K, ground_mask, object_mask)
        elif completion_mode == "hybrid":
            depth_bg = hybrid_fill(depth0, fg_img, bg_img, K, object_mask, device, plane_masks)
        else:
            # use monocular depth prediction for geometry completion
            print(f"[Info] [{key}] predicting depth with MoGe-2 ...")
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
        bg_pts, bg_colors = clear_nan(bg_pts, bg_colors)
        fg_pts, fg_colors = clear_nan(fg_pts, fg_colors)
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
