# grasp_det_full_all.py (batch keys)
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import trimesh
import open3d as o3d
import yaml

from grasp_utils import MyGraspNet


def sample_surface(glb_path: str, n_points: int = 5000) -> np.ndarray:
    """Load mesh and sample surface points."""
    mesh = trimesh.load_mesh(glb_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = list(mesh.geometry.values())[0]
    pts, _ = trimesh.sample.sample_surface(mesh, n_points)
    return np.array(pts, dtype=np.float32)


def crop_pointcloud_quadrant(
    points,
    quadrant=1,
    axes="xy",
):
    # Map axis letters to indices
    axis_index = {"x": 0, "y": 1, "z": 2}
    a1, a2 = axes[0], axes[1]
    idx1, idx2 = axis_index[a1], axis_index[a2]

    center_full = points.mean(axis=0)

    c1, c2 = center_full[idx1], center_full[idx2]

    ge = np.greater  # >

    # Build boolean masks for the four quadrants
    # Right/Left split along axis1, Top/Bottom split along axis2
    right = ge(points[:, idx1], c1)  # axis1 >= c1  (or > c1)
    top = ge(points[:, idx2], c2)  # axis2 >= c2  (or > c2)

    # Quadrant mapping per convention documented above
    if quadrant == 1:
        mask = right & top
    elif quadrant == 2:
        mask = (~right) & top
    elif quadrant == 3:
        mask = (~right) & (~top)
    elif quadrant == 4:
        mask = right & (~top)
    else:
        raise ValueError("`quadrant` must be an integer in {1,2,3,4}.")

    cropped_points = points[mask]
    return cropped_points, mask


def read_score_safe(gg, i: int, fallback: float) -> float:
    """Try to read candidate score; fall back to a synthetic rank-based value."""
    try:
        return float(getattr(gg[i], "score"))
    except Exception:
        pass
    try:
        return float(gg.grasp_group_array[i][0])
    except Exception:
        return fallback


def grasps_to_pointcloud(
    gg, pts_per_gripper: int = 400, color=(1.0, 0.0, 0.0)
) -> o3d.geometry.PointCloud:
    """Sample each gripper mesh to points and tint with color."""
    geoms = gg.to_open3d_geometry_list()  # list of TriangleMesh
    out = o3d.geometry.PointCloud()
    for g in geoms:
        out += g.sample_points_uniformly(pts_per_gripper)
    if len(out.points) > 0:
        out.colors = o3d.utility.Vector3dVector(
            np.tile(np.array(color, dtype=np.float32), (len(out.points), 1))
        )
    return out


def save_vis_ply(
    points_xyz: np.ndarray,
    gg,
    save_path: Path,
    pts_per_gripper: int = 400,
    cloud_color=(0.0, 1.0, 0.0),
):
    """Write a PLY: green object cloud + red ALL grasp candidates."""
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(
        np.tile(np.array(cloud_color, dtype=np.float32), (len(cloud.points), 1))
    )
    grasp_pcd = grasps_to_pointcloud(
        gg, pts_per_gripper=pts_per_gripper, color=(1.0, 0.0, 0.0)
    )
    merged = cloud + grasp_pcd
    save_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(save_path), merged)
    print(f"[VIS] saved {save_path}")


def process_one_object(
    gg_net: MyGraspNet,
    obj_idx: int,
    obj_dict: dict,
    proposals_dir: Path,
    keep: int | None,
    nms: bool,
    n_points: int,
    overwrite: bool,
    vis_pts_per_gripper: int = 400,
) -> str | None:
    """
    Build full point cloud for one object, run GraspNet, export all candidates to NPZ,
    save a PLY visualization, and return the NPZ path (string).
    """
    glb_path = obj_dict["optimized"]
    if glb_path is None or (not os.path.exists(glb_path)):
        print(
            f"[WARN][{obj_dict['oid']}_{obj_dict['name']}] mesh not found -> {glb_path}"
        )
        return None

    npy_path = proposals_dir / f"{obj_dict['oid']}_{obj_dict['name']}_grasp.npy"
    vis_path = proposals_dir / f"{obj_dict['oid']}_{obj_dict['name']}_grasp_viz.ply"

    # Build cloud & inference
    pcs = sample_surface(glb_path, n_points=n_points)
    # pcs = crop_pointcloud_quadrant(pcs, quadrant=3, axes="xy")[0]
    pcs[..., 2] *= -1.0  # keep identical convention
    gg = gg_net.inference(pcs)
    if nms:
        gg = gg.nms()
    gg = gg.sort_by_score()
    if keep is not None and len(gg) > keep:
        gg = gg[:keep]
    if len(gg) == 0:
        print(f"[WARN][{obj_dict['oid']}_{obj_dict['name']}] zero proposals")
        return None

    # Save visualization
    save_vis_ply(pcs, gg, vis_path, pts_per_gripper=vis_pts_per_gripper)

    for g_i in range(len(gg)):
        translation = gg[g_i].translation
        rotation = gg[g_i].rotation_matrix
        translation = np.array([translation[0], translation[1], -translation[2]])
        rotation[2, :] = -rotation[2, :]
        rotation[:, 2] = -rotation[:, 2]
        gg.grasp_group_array[g_i][13:16] = translation
        gg.grasp_group_array[g_i][4:13] = rotation.reshape(-1)

    gg.save_npy(npy_path)
    print(
        f"[OK][{obj_dict['oid']}_{obj_dict['name']}] saved {len(gg.grasp_group_array)} -> {npy_path.name}"
    )

    return str(npy_path.resolve())


def run_for_key(
    key: str,
    n_points: int,
    keep: int | None,
    nms: bool,
    overwrite: bool,
    vis_pts_per_gripper: int,
):
    base_dir = Path.cwd()
    out_dir = base_dir / "outputs"
    scene_json = out_dir / key / "simulation" / "scene.json"
    if not scene_json.exists():
        raise FileNotFoundError(scene_json)

    # Load scene.json once
    scene_dict = json.load(open(scene_json))
    objects = scene_dict.get("objects", {})
    if not isinstance(objects, dict) or len(objects) == 0:
        print(f"[WARN][{key}] scene_dict['objects'] is empty.")
        return

    # Prepare save dir
    key_dir = out_dir / key
    proposals_dir = key_dir / "grasps"
    proposals_dir.mkdir(parents=True, exist_ok=True)

    # Single GraspNet instance reused for all objects
    net = MyGraspNet()

    # Loop all objects
    for i, obj in objects.items():
        npy_path = process_one_object(
            gg_net=net,
            obj_idx=i,
            obj_dict=obj,
            proposals_dir=proposals_dir,
            keep=keep,
            nms=nms,
            n_points=n_points,
            overwrite=overwrite,
            vis_pts_per_gripper=vis_pts_per_gripper,
        )
        if npy_path is not None:
            scene_dict["objects"][i]["grasps"] = npy_path
            with open(scene_json, "w") as f:
                json.dump(scene_dict, f, indent=2)
            print(
                f"[OK][{key}] scene.json updated with 'grasps' for {obj['oid']}_{obj['name']}."
            )


def main():
    parser = argparse.ArgumentParser(
        "Export grasp proposals for ALL objects (batch over keys) and write paths into scene.json"
    )
    parser.add_argument("--n_points", type=int, default=100000)
    parser.add_argument("--keep", type=int, default=None)  # set None to keep all
    parser.add_argument("--nms", type=bool, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--vis_pts_per_gripper",
        type=int,
        default=400,
        help="points sampled per gripper mesh for PLY",
    )
    args = parser.parse_args()

    # load keys from YAML
    cfg_path = Path.cwd() / "config" / "config.yaml"
    cfg = yaml.safe_load(cfg_path.open("r"))
    keys = cfg["keys"]

    for key in keys:
        print(f"\n========== [GraspDet] Processing key: {key} ==========")
        run_for_key(
            key=key,
            n_points=args.n_points,
            keep=args.keep,
            nms=args.nms,
            overwrite=args.overwrite,
            vis_pts_per_gripper=args.vis_pts_per_gripper,
        )


if __name__ == "__main__":
    main()
