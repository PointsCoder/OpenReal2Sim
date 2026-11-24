"""
Assemble object meshes and background mesh into a physical scene by aligning them with point clouds.
Inputs:
    - outputs/{key}/scene/scene.pkl
Outputs:
    - outputs/{key}/scene/scene.pkl (updated with "info" key)
    - outputs/{key}/reconstruction/objects/{oid}_{name}_registered.glb (object mesh that is registered to the scene)
    - outputs/{key}/reconstruction/background_registered.glb (background mesh that is registered to the scene)
    - outputs/{key}/reconstruction/scene_registered.glb (the entire registered scene as a glb)
Note:
    - Here we extract all necessary information about the scene in "info":
        "info" : {
            "camera": {
                "camera_heading_wxyz": # camera heading as a quaternion [w,x,y,z],
                "camera_position":     # camera position in world frame [x,y,z],
                "camera_opencv_to_world": # camera extrinsics (opencv camera convention to world) as a flattened 4x4 matrix,
                "width":  # image width,
                "height": # image height,
                "fx":     # focal length x,
                "fy":     # focal length y,
                "cx":     # principal point x,
                "cy":     # principal point y,
            },
            "objects": {  # a list of objects in the scene
                "oid":   {
                        "oid":   # object id,
                        "name": # object name,
                        "object_center": # object center [x,y,z],
                        "object_min":    # object aabb min [x,y,z],
                        "object_max":    # object aabb max [x,y,z],
                        "original":      # original object mesh path,
                        "registered":    # registered object mesh path,
                        "object_to_world": # 4x4 transform from the original object glb to the world frame,
                    },
                ...
            },
            "background": {
                "original":   # original background mesh path,
                "registered": # registered background mesh path,
            },
            "aabb": {
                "scene_min": # scene aabb min [x,y,z],
                "scene_max": # scene aabb max [x,y,z],
            },
            "scene_mesh": {
                "registered": # registered scene mesh path,
            },
            "groundplane_in_cam": {
                "point":  # a point on the ground plane [x,y,z],
                "normal": # the normal of the ground plane [x,y,z],
            },
            "groundplane_in_sim": {
                "point":  # a point on the ground plane [x,y,z],
                "normal": # the normal of the ground plane [x,y,z],
            }
        }
"""

import yaml
from pathlib import Path
import sys
import json
import pickle
import numpy as np
import open3d as o3d
import trimesh
import copy
import transforms3d
# from scipy.spatial import ConvexHull, distance_matrix

from openreal2sim.steps.reconstruction.utils.table_slab import color_pointmap


# ─────────────────────────── object registration utilities ──────────────────────────

def coarse_align_by_centroid_and_bbox(source_pts: np.ndarray,
                                      target_pts: np.ndarray,
                                      do_uniform_scale: bool = True):
    """
    coarse align source_pts to target_pts by centroid and bounding box
    return T: only translation and scaling
    """
    center_s = source_pts.mean(axis=0)
    center_t = target_pts.mean(axis=0)

    s_shifted = source_pts - center_s

    scale_factor = 1.0
    if do_uniform_scale:
        def bbox_diag(pts):
            min_xyz = pts.min(axis=0)
            max_xyz = pts.max(axis=0)
            return np.linalg.norm(max_xyz - max_xyz * 0 + min_xyz)  # keep original behavior but corrected logically

        # keep original magnitude behavior
        min_s, max_s = source_pts.min(axis=0), source_pts.max(axis=0)
        min_t, max_t = target_pts.min(axis=0), target_pts.max(axis=0)
        diag_s = np.linalg.norm(max_s - min_s)
        diag_t = np.linalg.norm(max_t - min_t)
        if diag_s > 1e-9:
            scale_factor = diag_t / diag_s
        s_shifted *= scale_factor

    s_shifted += center_t

    T_coarse = np.eye(4)
    T_coarse[:3, :3] = scale_factor * np.eye(3)
    T_coarse[:3, 3] = center_t - scale_factor * center_s

    n = s_shifted.shape[0]
    ones = np.ones((n, 1))
    homo_source = np.hstack([source_pts, ones])  # Nx4
    source_pts_coarse = (T_coarse @ homo_source.T).T[:, :3]

    return T_coarse, source_pts_coarse

def trimesh_registration(source_pcd_o3d, target_pcd_o3d, max_iter=50):
    print(f"Running trimesh registration with {len(source_pcd_o3d.points)} source points and {len(target_pcd_o3d.points)} target points...")
    src_pts = np.asarray(source_pcd_o3d.points)
    tgt_pts = np.asarray(target_pcd_o3d.points)
    # trimesh registration
    T_fine, _transformed, cost = trimesh.registration.icp(
        src_pts, tgt_pts, max_iterations=max_iter, scale=True, reflection=False
    )
    return T_fine


# SAM-3D exports a canonical Y-up GLB while the point cloud is in the OpenCV
# camera frame, which is Y-down. Used only when the predicted layout pose is
# missing: a 180 degree rotation about x, so determinant +1, where the old code
# used the mirror diag(1, -1, 1).
FALLBACK_YUP_TO_OPENCV = np.diag([1.0, -1.0, -1.0])


def rotation_about(R, center):
    """4x4 transform that applies the rotation R about the given center."""
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = center - R @ center
    return transform


def load_object_rotation(obj_info, mesh):
    """
    Rotation that brings the object GLB from its canonical frame into the
    OpenCV camera frame, applied about the mesh centroid so that the metric
    scale and translation fitted afterwards stay valid.

    Returns the 4x4 transform and whether it came from the SAM-3D pose.
    """
    pose_path = obj_info.get("pose")
    R = FALLBACK_YUP_TO_OPENCV
    from_sam3d = False
    if pose_path and Path(pose_path).exists():
        pose = json.loads(Path(pose_path).read_text())
        linear = np.array(pose["glb_to_camera"], dtype=np.float64)[:3, :3]
        scale = np.cbrt(np.linalg.det(linear))
        if scale > 1e-9:
            R = linear / scale
            from_sam3d = True
        else:
            print(f"[Warn] Degenerate SAM-3D layout pose in {pose_path}, using the fallback rotation")
    else:
        print(f"[Warn] No SAM-3D layout pose for object {obj_info.get('oid')}, using the fallback rotation")

    determinant = float(np.linalg.det(R))
    assert determinant > 0.0, f"object rotation must be proper, determinant {determinant}"

    return rotation_about(R, np.asarray(mesh.centroid, dtype=np.float64)), from_sam3d

# How far the ICP refinement may rotate the object away from the orientation
# SAM-3D predicted. Past this it is the registration guessing again, so only
# the placement of the refinement is kept. SAM-3D predicts the pose against the
# pointmap of its own depth model, whose intrinsics need not match ours, so a
# residual tilt of a few tens of degrees is normal and has to stay correctable;
# the limit is there to block the 90 and 180 degree flips.
MAX_REFINE_ROTATION_DEG = 45.0


def limit_refinement_rotation(transform, center, max_deg=MAX_REFINE_ROTATION_DEG):
    """
    Drop the rotation of a refinement that turns the object too far, keeping
    where it moved the centroid. Returns the transform and the angle it asked for.
    """
    R = transform[:3, :3]
    angle = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))
    if angle <= max_deg:
        return transform, angle
    limited = np.eye(4)
    limited[:3, 3] = (R @ center + transform[:3, 3]) - center
    return limited, angle


def center_and_scale_mesh(mesh, scale_factor):
    center = mesh.centroid
    mesh.vertices -= center
    mesh.vertices *= scale_factor
    mesh.vertices += center
    return mesh


def scale_about(scale_factor, center):
    """4x4 transform matching center_and_scale_mesh, for bookkeeping."""
    transform = np.eye(4) * scale_factor
    transform[3, 3] = 1.0
    transform[:3, 3] = center - scale_factor * center
    return transform

# ───────────────────── SAM-3D layout placement ─────────────────────

def load_sam3d_pose(obj_info):
    """The layout pose SAM-3D predicted for an object, or None."""
    pose_path = obj_info.get("pose")
    if not pose_path or not Path(pose_path).exists():
        return None
    pose = json.loads(Path(pose_path).read_text())
    if "glb_to_camera" not in pose:
        return None
    return pose


def mesh_in_sam3d_camera(obj_info, mesh, pose):
    """The object mesh placed by SAM-3D, in the camera frame SAM-3D predicted it in."""
    transform = np.array(pose["glb_to_camera"], dtype=np.float64)
    determinant = float(np.linalg.det(transform[:3, :3]))
    assert determinant > 0.0, (
        f"object {obj_info.get('oid')} layout pose is a reflection, determinant {determinant}"
    )
    vertices = trimesh.transform_points(np.asarray(mesh.vertices, dtype=np.float64), transform)
    return transform, vertices


def camera_scale_factors(key, scene_dict, clouds, poses):
    """
    How SAM-3D's camera relates to ours, as two numbers per scene.

    SAM-3D predicts the layout in the camera of its own depth model, whose
    depth is not our depth and whose focal length is not our focal length.
    Both differences are global to the scene, so they are measured once over
    all the objects instead of being fitted per object:

      depth_factor: our depth over SAM-3D's depth, both medians over the
                    object masks. A uniform scaling about the camera origin
                    by this factor keeps the image unchanged.
      focal_ratio:  SAM-3D's focal over ours. After the depth factor the
                    object sits at the right distance but subtends the wrong
                    angle, and this is exactly the residual.

    The object then needs a uniform scale of depth_factor * focal_ratio, which
    is the one scale that makes it project onto its mask at our depth.
    """
    intrinsics = np.asarray(scene_dict["intrinsics"], dtype=np.float64)
    our_focal = 0.5 * (float(intrinsics[0, 0]) + float(intrinsics[1, 1]))
    depth_ratios, focal_ratios, rows = [], [], []
    for oid, pose in poses.items():
        if pose is None or "focal_px" not in pose:
            continue
        cloud = np.asarray(clouds[oid].points, dtype=np.float64)
        our_depth = float(np.median(cloud[:, 2]))
        sam_depth = float(pose["pointmap_median_depth"])
        sam_focal = 0.5 * (float(pose["focal_px"][0]) + float(pose["focal_px"][1]))
        depth_ratios.append(our_depth / max(sam_depth, 1e-9))
        focal_ratios.append(sam_focal / max(our_focal, 1e-9))
        rows.append((oid, our_depth, sam_depth, sam_focal, depth_ratios[-1], focal_ratios[-1]))

    if not rows:
        print(f"[Warn] [{key}] no SAM-3D camera measurements, falling back to the point cloud fit")
        return None

    factors = {
        "our_focal_px": our_focal,
        "depth_factor": float(np.median(depth_ratios)),
        "focal_ratio": float(np.median(focal_ratios)),
    }
    factors["object_scale"] = factors["depth_factor"] * factors["focal_ratio"]
    print(f"[Info] [{key}] SAM-3D camera against ours, our focal {our_focal:.2f} px")
    for oid, our_depth, sam_depth, sam_focal, depth_ratio, focal_ratio in rows:
        print(f"[Info] [{key}]   object {oid}: our depth {our_depth:.4f} m, SAM-3D depth "
              f"{sam_depth:.4f} m, ratio {depth_ratio:.4f}; SAM-3D focal {sam_focal:.2f} px, "
              f"ratio {focal_ratio:.4f}")
    print(f"[Info] [{key}] global depth factor {factors['depth_factor']:.4f}, focal ratio "
          f"{factors['focal_ratio']:.4f}, object scale {factors['object_scale']:.4f}")
    return factors


def sam3d_placement(obj_info, mesh, pose, factors, scene_dict):
    """
    Place the object exactly where SAM-3D put it, re-expressed in our camera.

    Keeps SAM-3D's rotation, its translation and its scale; the only thing
    added is the per-scene conversion between its camera and ours. No point
    cloud fit, no ICP: the mesh lands on the ray its own layout projects to,
    at our depth, with the one scale that matches its mask.
    """
    transform, vertices = mesh_in_sam3d_camera(obj_info, mesh, pose)
    center = vertices.mean(axis=0)
    depth = float(center[2])
    assert depth > 1e-6, f"object {obj_info.get('oid')} is behind the SAM-3D camera"

    width, height = pose["image_size"]
    sam_focal = 0.5 * (float(pose["focal_px"][0]) + float(pose["focal_px"][1]))
    # where SAM-3D's own camera projects the object centre
    u = sam_focal * center[0] / depth + 0.5 * width - 0.5
    v = sam_focal * center[1] / depth + 0.5 * height - 0.5

    intrinsics = np.asarray(scene_dict["intrinsics"], dtype=np.float64)
    ray = np.array([
        (u - intrinsics[0, 2]) / intrinsics[0, 0],
        (v - intrinsics[1, 2]) / intrinsics[1, 1],
        1.0,
    ])

    scale = factors["object_scale"]
    target = factors["depth_factor"] * depth * ray

    T_fit = np.eye(4)
    T_fit[:3, :3] = scale * np.eye(3)
    T_fit[:3, 3] = target - scale * center
    T_object = T_fit @ transform

    placed = mesh.copy()
    placed.apply_transform(T_object)
    print(f"[Info] Object {obj_info.get('oid')} placed from the SAM-3D layout: scale {scale:.4f}, "
          f"depth {target[2]:.4f} m, centre {np.round(target, 4)}")
    return placed, T_object


def report_projection(key, obj_info, mesh, scene_dict):
    """How the placed mesh projects against the mask it came from."""
    intrinsics = np.asarray(scene_dict["intrinsics"], dtype=np.float64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    front = vertices[vertices[:, 2] > 1e-6]
    if len(front) == 0:
        print(f"[Warn] [{key}] object {obj_info.get('oid')} projects behind the camera")
        return
    u = intrinsics[0, 0] * front[:, 0] / front[:, 2] + intrinsics[0, 2]
    v = intrinsics[1, 1] * front[:, 1] / front[:, 2] + intrinsics[1, 2]
    ys, xs = np.nonzero(obj_info["mask"])
    print(f"[Info] [{key}] object {obj_info.get('oid')} projects to "
          f"{u.max() - u.min():.1f} x {v.max() - v.min():.1f} px centred "
          f"({0.5 * (u.min() + u.max()):.1f}, {0.5 * (v.min() + v.max()):.1f}); mask is "
          f"{xs.max() - xs.min()} x {ys.max() - ys.min()} px centred "
          f"({0.5 * (xs.min() + xs.max()):.1f}, {0.5 * (ys.min() + ys.max()):.1f})")


def slow_registration(target_pcd_o3d, obj_info, object_dir):
    """
    Slow but accurate mesh to point cloud registration
    """

    fg_mesh_orig = trimesh.load_mesh(obj_info["glb"])
    if isinstance(fg_mesh_orig, trimesh.Scene):
        fg_mesh_orig = list(fg_mesh_orig.geometry.values())[0]
    T_rotate, rotation_from_sam3d = load_object_rotation(obj_info, fg_mesh_orig)

    # we use open3d to simplify the mesh first
    fg_mesh_o3d = o3d.io.read_triangle_mesh(str(obj_info["glb"]))  # legacy

    if fg_mesh_o3d.has_triangle_uvs():
        fg_mesh_o3d.triangle_uvs = o3d.utility.Vector2dVector()
    if fg_mesh_o3d.has_textures():
        fg_mesh_o3d.textures = []

    # simplify mesh for speed
    number = np.asarray(fg_mesh_o3d.vertices).shape[0]
    if number > 1200:
        if number > 50000:
            fg_mesh_o3d = fg_mesh_o3d.simplify_quadric_decimation(target_number_of_triangles=int(len(fg_mesh_o3d.triangles) * 0.01))
        else:
            keep = 1000
            target_tris = max(int(len(fg_mesh_o3d.triangles) * (keep / max(number, 1))), 200)
            fg_mesh_o3d = fg_mesh_o3d.simplify_quadric_decimation(target_number_of_triangles=target_tris)

    v_pos = np.asarray(fg_mesh_o3d.vertices, dtype=np.float32)
    t_idx = np.asarray(fg_mesh_o3d.triangles, dtype=np.int64)

    fg_mesh_register = trimesh.Trimesh(vertices=v_pos, faces=t_idx, process=False)

    # orient the mesh in the camera frame
    fg_mesh_register.apply_transform(T_rotate)

    # resize the mesh to point cloud scale
    points_masked = np.asarray(target_pcd_o3d.points, dtype=np.float32)
    min_p = points_masked.min(axis=0)
    max_p = points_masked.max(axis=0)
    distance_points = float(np.linalg.norm(max_p - min_p))

    min_m = v_pos.min(axis=0)
    max_m = v_pos.max(axis=0)
    distance_mesh = float(np.linalg.norm(max_m - min_m))

    scale_factor = distance_points / max(distance_mesh, 1e-12)
    T_scale = scale_about(scale_factor, np.asarray(fg_mesh_register.centroid, dtype=np.float64))
    fg_mesh_register = center_and_scale_mesh(fg_mesh_register, scale_factor)

    if rotation_from_sam3d:
        # The orientation is already known, so only the placement is searched:
        # move the centroid onto the cloud and refine with a rigid ICP.
        print(f"[Info] Refining the SAM-3D orientation with a rigid ICP ...")
        source_pts = np.asarray(fg_mesh_register.vertices, dtype=np.float64)
        T_center = np.eye(4)
        T_center[:3, 3] = points_masked.mean(axis=0) - source_pts.mean(axis=0)
        source_pts = trimesh.transform_points(source_pts, T_center)
        T_icp, _, _ = trimesh.registration.icp(
            source_pts, points_masked, max_iterations=50, scale=False, reflection=False
        )
        T_icp, angle = limit_refinement_rotation(T_icp, source_pts.mean(axis=0))
        print(f"[Info] ICP refinement rotation: {angle:.2f} deg"
              f"{' (dropped, placement only)' if angle > MAX_REFINE_ROTATION_DEG else ''}")
        T = T_icp @ T_center
    else:
        # No predicted orientation, so fall back to the principal axis search.
        # reflection=False keeps it from returning a mirrored solution.
        print(f"[Info] Running trimesh registration ...")
        T, _ = trimesh.registration.mesh_other(
            fg_mesh_register,
            points_masked,
            samples=points_masked.shape[0],
            scale=False,
            reflection=False,
        )
    print(f"[Info] Registration results: {T}")

    fg_mesh = fg_mesh_orig.copy()
    fg_mesh.apply_transform(T_rotate)                      # orientation
    fg_mesh = center_and_scale_mesh(fg_mesh, scale_factor)  # scale
    fg_mesh.apply_transform(T)                             # rigid placement

    return fg_mesh, T @ T_scale @ T_rotate


def fast_registration(target_pcd_o3d, obj_info, object_dir):
    fg_mesh = trimesh.load(obj_info["glb"])
    if isinstance(fg_mesh, trimesh.Scene):
        geo_list = list(fg_mesh.geometry.values())
        fg_mesh = geo_list[0]
    T_rotate, _ = load_object_rotation(obj_info, fg_mesh)
    sample_mesh_npts = 5000
    sampled_points, _ = fg_mesh.sample(sample_mesh_npts, return_index=True)
    source_pcd_o3d = o3d.geometry.PointCloud()
    source_pcd_o3d.points = o3d.utility.Vector3dVector(sampled_points)

    src_pts = np.asarray(source_pcd_o3d.points)
    tgt_pts = np.asarray(target_pcd_o3d.points)

    print("[Info] [Coarse Alignment] Using centroid + bounding box scaling...")
    T_coarse, src_pts_coarse = coarse_align_by_centroid_and_bbox(src_pts, tgt_pts, do_uniform_scale=True)

    T_coarse = T_coarse @ T_rotate

    source_pcd_o3d_coarse = copy.deepcopy(source_pcd_o3d)
    source_pcd_o3d_coarse.points = o3d.utility.Vector3dVector(src_pts_coarse)

    T_fine = trimesh_registration(source_pcd_o3d_coarse, target_pcd_o3d, max_iter=50)
    print(f"[Info] [Fine Alignment] T_fine: {T_fine}")

    fg_mesh.apply_transform(T_coarse)
    fg_mesh.apply_transform(T_fine)

    return fg_mesh, T_fine @ T_coarse

def foreground_pointmap(scene_dict: dict) -> np.ndarray:
    """The (H,W,6) xyzrgb map of the first frame, rgb in [0,1].

    The opt-in point completion step stores it in the scene; the default path does not run
    it, so build it from the same depth map and frame.
    """
    recon = scene_dict["recon"]
    if "fg_pts" in recon:
        return recon["fg_pts"]
    K = np.asarray(scene_dict["intrinsics"], np.float32)
    return color_pointmap(scene_dict["depths"][0], K, recon["foreground"])

def object_point_cloud(key: str, fg_pmap: np.ndarray, obj_info: dict):
    """The masked, outlier trimmed object point cloud in the camera frame."""
    base_dir = Path.cwd()
    object_dir = base_dir / f"outputs/{key}/reconstruction/objects"

    # extract object points and colors using the mask
    obj_mask = obj_info['mask']
    color_map = fg_pmap[..., 3:6].reshape(-1, 3)  # [H*W, 3]
    xyz_map = fg_pmap[..., 0:3].reshape(-1, 3)
    obj_mask = obj_mask.reshape(-1)
    object_points = xyz_map[obj_mask]
    object_colors = color_map[obj_mask]
    valid_idx = np.all(np.isfinite(object_points), axis=1)
    object_points = object_points[valid_idx]
    object_colors = object_colors[valid_idx]

    if len(object_points) < 10:
        raise ValueError("Not enough valid object points extracted from the mask.")

    # preprocess object points
    target_pcd_o3d = o3d.geometry.PointCloud()
    target_pcd_o3d.points = o3d.utility.Vector3dVector(object_points)
    target_pcd_o3d.colors = o3d.utility.Vector3dVector(object_colors)

    _, ind = target_pcd_o3d.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.5)
    target_pcd_o3d = target_pcd_o3d.select_by_index(ind)

    target_points_path = object_dir / Path(f"{obj_info['oid']}_{obj_info['name']}_points.ply")
    o3d.io.write_point_cloud(str(target_points_path), target_pcd_o3d)
    print(f"[Info] target object points saved to: {target_points_path}")
    return target_pcd_o3d


def register_object_mesh(
        key: str,
        scene_dict: dict,
        obj_info: dict,
        target_pcd_o3d,
        pose=None,
        factors=None,
        fast_alignment: bool = True,
        cfg=None,
    ):
    base_dir = Path.cwd()
    object_dir = base_dir / f"outputs/{key}/reconstruction/objects"

    # place the object mesh
    if pose is not None and factors is not None:
        mesh = trimesh.load_mesh(obj_info["glb"])
        if isinstance(mesh, trimesh.Scene):
            mesh = list(mesh.geometry.values())[0]
        fg_mesh, T_object = sam3d_placement(obj_info, mesh, pose, factors, scene_dict)
        report_projection(key, obj_info, fg_mesh, scene_dict)
        cloud = np.asarray(target_pcd_o3d.points, dtype=np.float64)
        extent_mesh = np.asarray(fg_mesh.bounds[1] - fg_mesh.bounds[0], dtype=np.float64)
        extent_cloud = cloud.max(axis=0) - cloud.min(axis=0)
        print(f"[Info] [{key}] object {obj_info['oid']} extent {np.round(extent_mesh, 4)} m "
              f"against the point cloud {np.round(extent_cloud, 4)} m, diagonal ratio "
              f"{np.linalg.norm(extent_mesh) / max(np.linalg.norm(extent_cloud), 1e-9):.4f}, "
              f"centroid distance "
              f"{np.linalg.norm(np.asarray(fg_mesh.vertices).mean(axis=0) - cloud.mean(axis=0)):.4f} m")
    elif fast_alignment:
        fg_mesh, T_object = fast_registration(target_pcd_o3d, obj_info, object_dir)
    else:
        fg_mesh, T_object = slow_registration(target_pcd_o3d, obj_info, object_dir)
    # Decide the orientation by render-and-compare instead of trusting the
    # seed: SAM-3D gets symmetries wrong, and neither a point cloud fit nor a
    # projection check can tell a remote turned 180 degrees from the right one.
    search_record = None
    if cfg is not None and cfg["pose_candidates"]:
        from openreal2sim.steps.reconstruction.steps.pose_candidate_search import search_object_pose
        fg_mesh, T_search, search_record = search_object_pose(
            key, scene_dict, obj_info, fg_mesh, cfg)
        T_object = T_search @ T_object
        report_projection(key, obj_info, fg_mesh, scene_dict)

    determinant = float(np.linalg.det(T_object[:3, :3]))
    print(f"[Info] Object {obj_info['oid']} registered, transform determinant {determinant:.6f}.")
    assert determinant > 0.0, (
        f"object {obj_info['oid']} registration transform is a reflection, determinant {determinant}"
    )

    return fg_mesh, obj_info, T_object, search_record

# ─────────────────────────── scene registration utilities ──────────────────────────

def gravity_alignment(normal):
    """
    recover camera pose from the ground normal
    """
    # compute the rotation R applied to the plane pts, so that the plane normal is aligned with the +z-axis in the world frame
    z_axis = np.array([0,0,1], dtype=np.float64)
    dot_val = np.dot(normal, z_axis)              # = cos(θ)
    angle = np.arccos(np.clip(dot_val, -1.0, 1.0))
    rot_axis = np.cross(normal, z_axis)
    axis_len = np.linalg.norm(rot_axis)

    if axis_len < 1e-8:
        # if the normal is already aligned with the z-axis
        R = np.eye(3, dtype=np.float64)
    else:
        rot_axis = rot_axis / axis_len
        R = transforms3d.axangles.axangle2mat(rot_axis, angle)

    cam_quat = transforms3d.quaternions.mat2quat(R)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R  # without translation

    return T, cam_quat.tolist()

def snap_to_plane(mesh, point, normal, max_height_fraction):
    """
    Drop the object onto the plane: translate it along the plane normal until
    its lowest vertices touch, but only when it is already close to it, so a
    stacked or hanging object is not pulled down. Returns the 4x4 translation
    and the gap it closed.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    along = vertices @ normal
    gap = float(along.min() - point @ normal)
    height = float(along.max() - along.min())
    transform = np.eye(4)
    if abs(gap) > max_height_fraction * max(height, 1e-9):
        print(f"[Info] Object sits {gap:.4f} m off the plane, more than "
              f"{max_height_fraction:.0%} of its {height:.4f} m height, so it is left alone")
        return transform, gap, False
    transform[:3, 3] = -gap * normal
    mesh.apply_transform(transform)
    print(f"[Info] Object snapped onto the plane by {-gap:.4f} m "
          f"(height {height:.4f} m)")
    return transform, gap, True


def register_scene_meshes(key, fg_meshes, obj_infos, object_transforms, scene_dict, cfg=None,
                          search_records=None):
    """
    Register the foreground and background meshes to the physical scene.
    Recover the camera pose and save the aligned meshes.
    """
    background_mesh_path = scene_dict["info"]["background"]["original"]
    bg_mesh = trimesh.load(background_mesh_path)

    normal = scene_dict["recon"]["normal"]
    T_gravity, cam_pose = gravity_alignment(normal)
    print(f"Gravity transform: {T_gravity}")
    assert np.linalg.det(T_gravity[:3, :3]) > 0.0, "gravity alignment must be a proper rotation"

    # rotate the background mesh to align with gravity
    bg_mesh.apply_transform(T_gravity)

    # move the background mesh to the scene origin
    aabb = bg_mesh.bounds
    aabb_center = 0.5 * (aabb[0] + aabb[1])
    T_to_origin = np.eye(4)
    T_to_origin[:3, 3] = -aabb_center
    bg_mesh.apply_transform(T_to_origin)
    aabb_info = {
        "scene_min": (aabb[0] - aabb_center).tolist(),
        "scene_max": (aabb[1] - aabb_center).tolist(),
    }

    R_c2w_new = T_gravity[:3, :3]
    C_new = -aabb_center

    # update plane info
    p0 = np.array(scene_dict["info"]["groundplane_in_cam"]["point"], dtype=np.float64)
    normal = np.array(scene_dict["info"]["groundplane_in_cam"]["normal"], dtype=np.float64)
    p0 = (T_to_origin @ T_gravity @ np.hstack([p0, 1.0]))[:3]
    normal = (T_gravity @ np.hstack([normal, 0.0]))[:3]
    normal = normal / np.linalg.norm(normal)
    assert abs(np.dot(normal, np.array([0,0,1], dtype=np.float64)) - 1.0) < 1e-6, "Plane normal not aligned with z-axis after gravity alignment"
    plane_info = {
        "point": p0.tolist(),
        "normal": normal.tolist(),
    }

    # add camera info
    cam_info = {}
    cam_info["camera_heading_wxyz"] = list(transforms3d.quaternions.mat2quat(R_c2w_new))
    cam_info["camera_position"]     = C_new.tolist()
    T_c2w = T_to_origin @ T_gravity
    cam_info["camera_opencv_to_world"] = T_c2w.tolist()
    cam_info["width"] = float(scene_dict["width"])
    cam_info["height"] = float(scene_dict["height"])
    cam_info["fx"] = float(scene_dict["intrinsics"][0,0])
    cam_info["fy"] = float(scene_dict["intrinsics"][1,1])
    cam_info["cx"] = float(scene_dict["intrinsics"][0,2])
    cam_info["cy"] = float(scene_dict["intrinsics"][1,2])

    # transform object meshes
    base_dir = Path.cwd()
    save_obj_infos = {}
    search_records = search_records or {}
    snap = True if cfg is None else cfg["table_snap"]
    rest = True if cfg is None else cfg["table_rest"]
    max_height_fraction = 0.2 if cfg is None else cfg["table_snap_max_height_fraction"]
    max_rest_rotation = 45.0 if cfg is None else cfg["table_rest_max_rotation_deg"]
    lift_below_plane = True if cfg is None else cfg["table_rest_lift_below_plane"]
    for obj_info, fg_mesh, T_object in zip(obj_infos, fg_meshes, object_transforms):
        # apply gravity and translation to each object mesh
        fg_mesh.apply_transform(T_gravity)
        fg_mesh.apply_transform(T_to_origin)

        # rest the object on the table, which is the plane the gravity
        # alignment was estimated from: turn its support face flat, then drop
        # it along the normal until it touches
        T_snap = np.eye(4)
        rest_report = None
        if rest:
            from openreal2sim.steps.reconstruction.steps.pose_candidate_search import rest_on_plane
            print(f"[Info] Resting object {obj_info['oid']}_{obj_info['name']} on the table")
            T_snap, rest_report = rest_on_plane(
                fg_mesh, p0, normal,
                max_rotation_deg=max_rest_rotation,
                max_height_fraction=max_height_fraction,
                lift_below_plane=lift_below_plane,
            )
        elif snap:
            T_snap, _, _ = snap_to_plane(fg_mesh, p0, normal, max_height_fraction)

        # the object GLB to world transform, the whole chain in one matrix
        T_object_to_world = T_snap @ T_to_origin @ T_gravity @ T_object
        determinant = float(np.linalg.det(T_object_to_world[:3, :3]))
        print(f"[Info] Object {obj_info['oid']}_{obj_info['name']} object to world determinant: {determinant:.6f}")
        assert determinant > 0.0, (
            f"object {obj_info['oid']} to world transform is a reflection, determinant {determinant}"
        )

        obj_aabb = fg_mesh.bounds
        obj_aabb_center = 0.5 * (obj_aabb[0] + obj_aabb[1])
        out_path = base_dir / Path(f"outputs/{key}/reconstruction/objects") / Path(f"{obj_info['oid']}_{obj_info['name']}_registered.glb")
        fg_mesh.export(out_path)

        save_obj_info = {
            "oid": obj_info['oid'],
            "name": obj_info['name'],
            "object_center": obj_aabb_center.tolist(),
            "object_min": obj_aabb[0].tolist(),
            "object_max": obj_aabb[1].tolist(),
            "original": str(obj_info['glb']),
            "registered": str(out_path),
            "object_to_world": T_object_to_world.tolist(),
        }
        if rest_report is not None:
            save_obj_info["table_rest"] = rest_report
        if obj_info['oid'] in search_records and search_records[obj_info['oid']] is not None:
            save_obj_info["pose_search"] = search_records[obj_info['oid']]
        save_obj_infos[obj_info['oid']] = save_obj_info
        print(f"[Info] Object {obj_info['oid']}_{obj_info['name']} aligned and saved to: {out_path}")

    # save transformed background mesh
    bg_mesh_save_path = base_dir / Path(f"outputs/{key}/reconstruction/background_registered.glb")
    bg_mesh.export(bg_mesh_save_path)
    print(f"[Info] Background mesh aligned and saved to: {bg_mesh_save_path}")

    # save the entire scene as a glb
    scene = trimesh.Scene()
    scene.add_geometry(bg_mesh, node_name="background")
    for obj_info in save_obj_infos.values():
        fg_mesh = trimesh.load(obj_info['registered'])
        scene.add_geometry(fg_mesh, node_name=f"{obj_info['oid']}_{obj_info['name']}")
    scene_mesh_path = base_dir / Path(f"outputs/{key}/reconstruction/scenario/scene_registered.glb")
    scene_mesh_path.parent.mkdir(parents=True, exist_ok=True)
    scene_mesh_path_str = str(scene_mesh_path)
    scene.export(
        file_obj=scene_mesh_path_str,
        file_type='glb'
    )
    print(f"[Info] Scene mesh saved to: {scene_mesh_path}")

    # save scene info
    scene_dict["info"]["groundplane_in_sim"] = plane_info
    scene_dict["info"]["camera"] = cam_info
    scene_dict["info"]["objects"] = save_obj_infos
    scene_dict["info"]["background"] = {
        "original": str(background_mesh_path),
        "registered": str(bg_mesh_save_path),
    }
    scene_dict["info"]["aabb"] = aabb_info
    scene_dict["info"]["scene_mesh"] = {
        "registered": str(scene_mesh_path)
    }
    return scene_dict


def scenario_construction(keys, key_scene_dicts, key_cfgs):
    base_dir = Path.cwd()
    for key in keys:
        print(f"[Info] Scenario construction for key: {key}")
        scene_dict = key_scene_dicts[key]
        cfg = key_cfgs[key]
        object_metas = scene_dict.get("objects", {})
        if not object_metas:
            print(f"[Warning] No objects found in scene_dict for key: {key}. Skipping.")
            continue
        fg_pmap = foreground_pointmap(scene_dict)
        clouds = {oid: object_point_cloud(key, fg_pmap, obj)
                  for oid, obj in object_metas.items()}

        # Placement from the layout SAM-3D predicted, which needs the two
        # per-scene numbers that relate its camera to ours.
        factors = None
        poses = {oid: None for oid in object_metas}
        if cfg["object_placement_mode"] == "sam3d":
            poses = {oid: load_sam3d_pose(obj) for oid, obj in object_metas.items()}
            missing = [oid for oid, pose in poses.items() if pose is None]
            if missing:
                print(f"[Warn] [{key}] no SAM-3D layout pose for objects {missing}")
            factors = camera_scale_factors(key, scene_dict, clouds, poses)
            if factors is None:
                poses = {oid: None for oid in object_metas}

        fg_meshes, obj_infos, object_transforms = [], [], []
        search_records = {}
        for oid, obj in object_metas.items():
            print(f"[Info] Adding object {obj['name']} (oid={obj['oid']}) to scene [key={key}]")
            fg_mesh, obj_info, T_object, search_record = register_object_mesh(
                key            = key,
                scene_dict      = scene_dict,
                obj_info       = obj,
                target_pcd_o3d = clouds[oid],
                pose           = poses[oid],
                factors        = factors,
                fast_alignment = cfg["fast_scene_construction"],
                cfg            = cfg,
            )
            fg_meshes.append(fg_mesh)
            obj_infos.append(obj_info)
            object_transforms.append(T_object)
            search_records[obj_info["oid"]] = search_record
        print(f"[Info] align the scene meshes [key={key}] with gravity ===")
        scene_dict = register_scene_meshes(key, fg_meshes, obj_infos, object_transforms, scene_dict, cfg,
                                           search_records=search_records)
        key_scene_dicts[key] = scene_dict
        with open(base_dir / f'outputs/{key}/scene/scene.pkl', 'wb') as f:
            pickle.dump(scene_dict, f)

    return key_scene_dicts

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

    scenario_construction(keys, key_scene_dicts, key_cfgs)
