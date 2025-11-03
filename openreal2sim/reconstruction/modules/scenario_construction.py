"""
Assemble object meshes and background mesh into a physical scene by aligning them with point clouds.
Inputs:
    - outputs/{key_name}/scene/scene.pkl
Outputs:
    - outputs/{key_name}/scene/scene.pkl (updated with "info" key)
    - outputs/{key_name}/reconstruction/objects/{oid}_{name}_registered.glb (object mesh that is registered to the scene)
    - outputs/{key_name}/reconstruction/background_registered.glb (background mesh that is registered to the scene)
    - outputs/{key_name}/reconstruction/scene_registered.glb (the entire registered scene as a glb)
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
import pickle
import numpy as np
import open3d as o3d
import trimesh
import copy
import transforms3d
# from scipy.spatial import ConvexHull, distance_matrix


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
    T_fine, _transformed, cost = trimesh.registration.icp(src_pts, tgt_pts, max_iterations=max_iter, scale=True)
    return T_fine


def get_flip(mesh):
    center = mesh.centroid  # 或 fg_mesh.bounds.mean(axis=0)

    T_to_origin = np.eye(4)
    T_to_origin[:3, 3] = -center

    T_back = np.eye(4)
    T_back[:3, 3] = center

    flip_y = np.diag([1, -1, 1, 1])

    transform = T_back @ flip_y @ T_to_origin

    return transform

def center_and_scale_mesh(mesh, scale_factor):
    center = mesh.centroid
    mesh.vertices -= center
    mesh.vertices *= scale_factor
    mesh.vertices += center
    return mesh

def slow_registration(target_pcd_o3d, obj_info, object_dir):
    """
    Slow but accurate mesh to point cloud registration
    """

    fg_mesh_orig = trimesh.load_mesh(obj_info["glb"])
    if isinstance(fg_mesh_orig, trimesh.Scene):
        fg_mesh_orig = list(fg_mesh_orig.geometry.values())[0]
    T_flip = get_flip(fg_mesh_orig)

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

    # apply the flip transform
    fg_mesh_register.apply_transform(T_flip)

    # resize the mesh to point cloud scale
    points_masked = np.asarray(target_pcd_o3d.points, dtype=np.float32)
    min_p = points_masked.min(axis=0)
    max_p = points_masked.max(axis=0)
    distance_points = float(np.linalg.norm(max_p - min_p))

    min_m = v_pos.min(axis=0)
    max_m = v_pos.max(axis=0)
    distance_mesh = float(np.linalg.norm(max_m - min_m))

    scale_factor = distance_points / max(distance_mesh, 1e-12)
    fg_mesh_register = center_and_scale_mesh(fg_mesh_register, scale_factor)

    # using trimesh registration to place the object mesh in correct position
    print(f"[Info] Running trimesh registration ...")
    T, _ = trimesh.registration.mesh_other(
        fg_mesh_register,
        points_masked,
        samples=points_masked.shape[0],
        scale=False
    )
    print(f"[Info] Registration results: {T}")

    fg_mesh = fg_mesh_orig.copy()
    fg_mesh.apply_transform(T_flip)                # flip
    fg_mesh = center_and_scale_mesh(fg_mesh, scale_factor)  # scale
    fg_mesh.apply_transform(T)                     # rigid placement from mesh_other
    
    return fg_mesh


def fast_registration(target_pcd_o3d, obj_info, object_dir):
    fg_mesh = trimesh.load(obj_info["glb"])
    T_flip = get_flip(fg_mesh)

    if isinstance(fg_mesh, trimesh.Scene):
        geo_list = list(fg_mesh.geometry.values())
        fg_mesh = geo_list[0]
    sample_mesh_npts = 5000
    sampled_points, _ = fg_mesh.sample(sample_mesh_npts, return_index=True)
    source_pcd_o3d = o3d.geometry.PointCloud()
    source_pcd_o3d.points = o3d.utility.Vector3dVector(sampled_points)

    src_pts = np.asarray(source_pcd_o3d.points)
    tgt_pts = np.asarray(target_pcd_o3d.points)

    print("[Info] [Coarse Alignment] Using centroid + bounding box scaling...")
    T_coarse, src_pts_coarse = coarse_align_by_centroid_and_bbox(src_pts, tgt_pts, do_uniform_scale=True)

    T_coarse = T_coarse @ T_flip

    source_pcd_o3d_coarse = copy.deepcopy(source_pcd_o3d)
    source_pcd_o3d_coarse.points = o3d.utility.Vector3dVector(src_pts_coarse)

    T_fine = trimesh_registration(source_pcd_o3d_coarse, target_pcd_o3d, max_iter=50)
    print(f"[Info] [Fine Alignment] T_fine: {T_fine}")

    fg_mesh.apply_transform(T_coarse)
    fg_mesh.apply_transform(T_fine)

    return fg_mesh

def register_object_mesh(
        key: str,
        scene_dict: dict,
        obj_info: dict,
        fast_alignment: bool = True,
    ):
    base_dir = Path.cwd()
    object_dir = base_dir / f"outputs/{key}/reconstruction/objects"

    # extract object points and colors using the mask
    obj_mask = obj_info['mask']
    color_map = scene_dict["recon"]["fg_pts"][..., 3:6]  # use RGB from fg_pts
    color_map = color_map.reshape(-1, 3)  # [H*W, 3]
    xyz_map = scene_dict["recon"]["fg_pts"][..., 0:3]  # use XYZ from fg_pts
    xyz_map = xyz_map.reshape(-1, 3)
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

    # register the object mesh to the target points
    if fast_alignment:
        fg_mesh = fast_registration(target_pcd_o3d, obj_info, object_dir)
    else:
        fg_mesh = slow_registration(target_pcd_o3d, obj_info, object_dir)
    print(f"[Info] Object {obj_info['oid']} registered.")

    return fg_mesh, obj_info

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

def register_scene_meshes(key, fg_meshes, obj_infos, scene_dict):
    """
    Register the foreground and background meshes to the physical scene.
    Recover the camera pose and save the aligned meshes.
    """
    background_mesh_path = scene_dict["info"]["background"]["original"]
    bg_mesh = trimesh.load(background_mesh_path)

    normal = scene_dict["recon"]["normal"]
    T_gravity, cam_pose = gravity_alignment(normal)
    print(f"Gravity transform: {T_gravity}")

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
    for obj_info, fg_mesh in zip(obj_infos, fg_meshes):
        # apply gravity and translation to each object mesh
        fg_mesh.apply_transform(T_gravity)
        fg_mesh.apply_transform(T_to_origin)
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
        }
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
        fg_meshes, obj_infos = [], []
        for obj in object_metas.values():
            print(f"[Info] Adding object {obj['name']} (oid={obj['oid']}) to scene [key={key}]")
            fg_mesh, obj_info = register_object_mesh(
                key            = key,
                scene_dict      = scene_dict,
                obj_info       = obj,
                fast_alignment = cfg["fast_scene_construction"],
            )
            fg_meshes.append(fg_mesh)
            obj_infos.append(obj_info)
        print(f"[Info] align the scene meshes [key={key}] with gravity ===")
        scene_dict = register_scene_meshes(key, fg_meshes, obj_infos, scene_dict)
        key_scene_dicts[key] = scene_dict
        with open(base_dir / f'outputs/{key}/scene/scene.pkl', 'wb') as f:
            pickle.dump(scene_dict, f)

    return key_scene_dicts

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

    scenario_construction(keys, key_scene_dicts, key_cfgs)
