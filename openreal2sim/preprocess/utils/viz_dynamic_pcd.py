# tools/vis_pcd.py
# -*- coding: utf-8 -*-
"""
Save an interactive multi-frame colored point cloud HTML after reconstruction.
Additionally: merge all frames into a single point cloud and voxel-downsample (Open3D).
"""

from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm
import yaml
from pathlib import Path
import open3d as o3d  # Default assumption: Open3D is installed
import pickle

base_dir = Path.cwd()


def voxel_downsample_open3d(points_xyz: np.ndarray,
                             colors_rgb01: np.ndarray,
                             voxel_size: float,
                             max_total_points: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Downsample merged points using Open3D voxel grid.

    Args:
        points_xyz: (N,3) float32 world-space points
        colors_rgb01: (N,3) float32 colors in [0,1]
        voxel_size: voxel size in meters (>0)
        max_total_points: optional cap after downsampling

    Returns:
        (ds_points_xyz, ds_colors_rgb01)
    """
    if points_xyz.size == 0:
        return points_xyz, colors_rgb01

    # Build point cloud for voxel downsampling
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors_rgb01.astype(np.float64))
    pcd_ds = pcd.voxel_down_sample(voxel_size=float(voxel_size))

    ds_points = np.asarray(pcd_ds.points, dtype=np.float32)
    ds_colors = np.asarray(pcd_ds.colors, dtype=np.float32)

    # Optional cap to avoid huge files
    if max_total_points is not None and len(ds_points) > max_total_points:
        keep = np.random.choice(len(ds_points), size=max_total_points, replace=False)
        ds_points = ds_points[keep]
        ds_colors = ds_colors[keep]

    return ds_points, ds_colors


def save_merged_cloud_ply(key: str,
                           points_xyz: np.ndarray,
                           colors_rgb01: np.ndarray,
                           voxel_size: float,
                           binary_compressed: bool = True) -> str:
    """
    Save merged cloud to PLY (binary + compressed by default).

    Returns:
        output_path: file path string to the saved file.
    """
    recon_dir = Path(f"outputs/{key}/geometry")
    recon_dir.mkdir(parents=True, exist_ok=True)
    out_path = recon_dir / f"dynamic_pcd.ply"

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors_rgb01.astype(np.float64))

    # write_ascii=False -> binary; compressed=True -> smaller file size
    o3d.io.write_point_cloud(
        str(out_path),
        pcd,
        write_ascii=not binary_compressed,
        compressed=binary_compressed
    )
    return str(out_path)


def save_dynamic_pcd(
    key: str,
    max_points: int = 50000,
    point_size: int = 5,
    title: str = "Dynamic Point Cloud (drag the slider or press Play)",
    # —— Merged cloud options —— #
    save_merged: bool = True,
    voxel_size: float = 0.01,
    max_total_points: int = 500_000,
    merge_use_frame_downsample: bool = True,
) -> tuple[str, str | None]:
    """
    Build a time-varying point cloud visualization and save as HTML.
    Optionally: merge ALL frames into a single point cloud and voxel-downsample (Open3D).

    Args:
        key: scene key, e.g., "lab1"
        max_points: per-frame random downsampling count for HTML interactivity
        point_size: marker size in the 3D scatter
        title: plot title
        save_merged: if True, save a single merged point cloud of all frames
        voxel_size: voxel size in meters for merged downsample (Open3D)
        max_total_points: cap after downsampling to avoid huge files
        merge_use_frame_downsample:
            - True  (default): merge the *same* per-frame downsampled points used for HTML
            - False: merge *all pixels of each frame* (much heavier), then downsample by voxel grid

    Returns:
        (html_out, merged_out)
          - html_out: output HTML path string for the animation
          - merged_out: saved merged PLY path if save_merged=True, else None
    """
    # I/O paths
    scene_path = f"outputs/{key}/scene/scene.pkl"
    geometry_dir = Path(f"outputs/{key}/geometry")
    geometry_dir.mkdir(parents=True, exist_ok=True)
    html_out = f"outputs/{key}/geometry/dynamic_pcd.html"

    # Load data directly, keep your original naming
    with open(scene_path, "rb") as f:
        data = pickle.load(f)
    imgs, depths = data["images"], data["depths"]        # imgs: (T,H,W,3) uint8; depths: (T,H,W) float16
    K = data["intrinsics"]                                # (3,3)
    c2ws = data["extrinsics"]                               # (T,4,4)

    # Camera intrinsics
    H, W = depths.shape[1], depths.shape[2]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # Prepare a pixel grid (1D form for fast vectorization)
    ii, jj = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    ii, jj = ii.reshape(-1), jj.reshape(-1)
    num_pixels = ii.shape[0]

    frames = []
    merged_points_world_chunks = []
    merged_colors_rgb01_chunks = []

    for t in tqdm(range(len(imgs)), desc="frames"):
        # -------- depth to camera coordinates --------
        z_all = depths[t].reshape(-1).astype(np.float32)    # (H*W,)
        x_all = (ii - cx) * z_all / fx
        y_all = (jj - cy) * z_all / fy
        pts_cam_all = np.stack([x_all, y_all, z_all], axis=1)  # (Npix, 3)

        # -------- camera-to-world transform --------
        R, T = c2ws[t][:3, :3], c2ws[t][:3, 3]
        pts_world_all = (R @ pts_cam_all.T + T[:, None]).T  # (Npix, 3)

        # -------- colors --------
        cols01_all = (imgs[t].reshape(-1, 3) / 255.0).astype(np.float32)

        # -------- choose indices for HTML frame & (optionally) for merging --------
        if max_points is not None and max_points < num_pixels:
            # Random subset for interactive speed
            keep_idx = np.random.choice(num_pixels, size=max_points, replace=False)
        else:
            keep_idx = np.arange(num_pixels)

        pts_for_html = pts_world_all[keep_idx]
        cols_for_html = cols01_all[keep_idx]

        # Plotly frame
        frames.append(
            go.Frame(
                data=[
                    go.Scatter3d(
                        x=pts_for_html[:, 0], y=pts_for_html[:, 1], z=pts_for_html[:, 2],
                        mode="markers",
                        marker=dict(
                            size=point_size,
                            color=[
                                'rgb(%d,%d,%d)' % (int(r*255), int(g*255), int(b*255))
                                for r, g, b in cols_for_html
                            ]
                        )
                    )
                ],
                name=f"{t:03d}"
            )
        )

        # Accumulate merged data
        if save_merged:
            if merge_use_frame_downsample:
                # Merge the same downsampled points used in the HTML
                merged_points_world_chunks.append(pts_for_html)
                merged_colors_rgb01_chunks.append(cols_for_html)
            else:
                # Merge *all* pixels for higher fidelity (memory heavier)
                merged_points_world_chunks.append(pts_world_all)
                merged_colors_rgb01_chunks.append(cols01_all)

    # Use first frame as initial content
    init_data = frames[0].data

    fig = go.Figure(
        data=init_data,
        frames=frames,
        layout=go.Layout(
            scene=dict(aspectmode="data"),
            updatemenus=[dict(
                type="buttons",
                buttons=[
                    dict(label="▶️ Play", method="animate", args=[None]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}])
                ],
                showactive=False,
                x=0.05, y=0
            )],
            sliders=[dict(
                steps=[dict(method="animate", args=[[f.name],
                       dict(mode="immediate", frame={"duration": 0})], label=f.name)
                       for f in frames],
                active=0, x=0.05, y=-0.05, len=0.9
            )],
            margin=dict(l=0, r=0, b=0, t=30),
            title=title
        )
    )

    # Save HTML
    Path(html_out).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(html_out, include_plotlyjs="cdn")
    print("✅ HTML saved:", html_out)

    merged_out = None
    if save_merged:
        if len(merged_points_world_chunks) > 0:
            all_pts = np.concatenate(merged_points_world_chunks, axis=0).astype(np.float32)
            all_cols = np.concatenate(merged_colors_rgb01_chunks, axis=0).astype(np.float32)
        else:
            all_pts = np.empty((0, 3), dtype=np.float32)
            all_cols = np.empty((0, 3), dtype=np.float32)

        # Apply voxel downsampling to the merged set
        if voxel_size is None or voxel_size <= 0:
            raise ValueError("voxel_size must be > 0 when Open3D is used for downsampling.")
        ds_pts, ds_cols = voxel_downsample_open3d(
            all_pts, all_cols, voxel_size=float(voxel_size), max_total_points=max_total_points
        )

        # Save merged PLY (binary + compressed)
        merged_out = save_merged_cloud_ply(
            key=key,
            points_xyz=ds_pts,
            colors_rgb01=ds_cols,
            voxel_size=float(voxel_size),
            binary_compressed=True
        )
        print("✅ Merged point cloud saved:", merged_out)

    return html_out, merged_out


if __name__ == "__main__":
    # Read keys and visualization parameters directly from config/config.yaml
    cfg_path = base_dir / "config" / "config.yaml"
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)

    keys = cfg["keys"]  # e.g., ["lab2", "lab3"]

    for scene_key in keys:
        print(f"[Visualization] Saving dynamic PCD HTML to: outputs/{scene_key}/geometry/dynamic_pcd.html")
        html_path, merged_path = save_dynamic_pcd(
            key=scene_key,
            max_points=5000,                 # per-frame random downsample for HTML speed
            point_size=5,
            title="Dynamic Point Cloud (drag the slider or press Play)",
            save_merged=True,                # enable merged cloud output
            voxel_size=0.001,                 # 1 mm voxel
            max_total_points=500_000,        # safety cap after voxel DS
            merge_use_frame_downsample=True  # set False for denser merging (heavier)
        )
        print(f"[Done] HTML:   {html_path}")
        if merged_path is not None:
            print(f"[Done] Merged: {merged_path}")
