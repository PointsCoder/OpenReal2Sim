#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert depth.png to a color image using magma colormap.

Steps:
1. Load depth.png as a grayscale depth map.
2. Multiply values by the given depth_scale.
3. Normalize depths to [0,1] for colormap mapping (robust to outliers).
4. Apply matplotlib 'magma' colormap.
5. Save the result as PNG or JPG.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
base_dir = Path.cwd()
import yaml

def depth_to_color_image(
    depth_path: str,
    output_path: str,
    depth_scale: float,
    colormap: str = "magma",
    vmin: float | None = None,
    vmax: float | None = None,
    percentiles: tuple[float, float] = (1.0, 99.0),
):
    """
    Convert depth.png to color image and save, with robust outlier handling.

    Args:
        depth_path: path to raw depth PNG (usually uint16).
        output_path: where to save the colored image (.png or .jpg).
        depth_scale: scale factor to convert raw units to meters.
        colormap: matplotlib colormap name (default 'magma').
        vmin/vmax: optional fixed depth range in meters for visualization.
                   If both provided and vmax>vmin, depths will be clipped to [vmin, vmax].
        percentiles: used only when vmin/vmax are not both provided; e.g., (1.0, 99.0).
                     Depths outside percentile range are clipped before normalization.
    """
    # ---- Load depth (keep as-is if load fails) ----
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"Cannot read {depth_path}")
    depth_raw = depth_raw.astype(np.float32)

    # ---- Scale to meters ----
    depth_m = depth_raw * float(depth_scale)

    # ---- Build valid mask (ignore non-positive / non-finite) ----
    valid_mask = np.isfinite(depth_m) & (depth_m > 0.0)
    if not np.any(valid_mask):
        raise ValueError("No valid depth pixels (>0) found after scaling.")

    # ---- Determine visualization range and clip to it ----
    use_fixed_range = (vmin is not None) and (vmax is not None) and (float(vmax) > float(vmin))
    if use_fixed_range:
        lo, hi = float(vmin), float(vmax)
    else:
        p_lo, p_hi = percentiles
        lo, hi = np.percentile(depth_m[valid_mask], [p_lo, p_hi])
        # Fallback if degenerate
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(depth_m[valid_mask].min()), float(depth_m[valid_mask].max())

    depth_clipped = np.clip(depth_m, lo, hi)

    # ---- Normalize to [0,1] based on [lo, hi] ----
    denom = max(hi - lo, 1e-8)
    depth_norm = (depth_clipped - lo) / denom
    depth_norm = np.clip(depth_norm, 0.0, 1.0)
    # Keep invalid pixels black
    depth_norm[~valid_mask] = 0.0

    # ---- Apply colormap (matplotlib returns RGBA in [0,1]) ----
    try:
        import matplotlib
        cmap = matplotlib.colormaps.get(colormap)
    except Exception:
        cmap = plt.get_cmap(colormap)
    depth_color = (cmap(depth_norm)[..., :3] * 255.0 + 0.5).astype(np.uint8)  # RGB

    # ---- Ensure output directory exists ----
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Save color image ----
    out_ext = out_path.suffix.lower()
    if out_ext in [".jpg", ".jpeg"]:
        cv2.imwrite(str(out_path), cv2.cvtColor(depth_color, cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    else:  # default PNG
        cv2.imwrite(str(out_path), cv2.cvtColor(depth_color, cv2.COLOR_RGB2BGR))

    print(f"Saved colorized depth to {out_path}  (range used: [{lo:.4f}, {hi:.4f}] m)")

if __name__ == "__main__":
    # Read keys and visualization parameters directly from config/config.yaml
    cfg_path = base_dir / "config" / "config.yaml"
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)

    keys = cfg["keys"]  # e.g., ["lab2", "lab3"]

    for key in keys:
        input_path = f"data/{key}_depth.png"
        output_path = f"outputs/{key}/geometry/reconstruction/colored_depth.png"
        depth_scale = float(cfg["specs"][key]["intrinsics"]["depth_scale"])

        # Per-key override if provided
        depth_key = cfg["specs"][key].get("depth_rectify", {})
        dmin = depth_key.get("dmin", None)
        dmax = depth_key.get("dmax", None)
        percentiles = tuple(depth_key.get("percentiles", [1.0, 99.0]))

        depth_to_color_image(
            input_path,
            output_path,
            depth_scale=depth_scale,
            colormap="magma",
            vmin=dmin,
            vmax=dmax,
            percentiles=percentiles,
        )
