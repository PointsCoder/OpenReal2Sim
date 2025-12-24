# Source Generated with Decompyle++
# File: grasp_generation.cpython-310.pyc (Python 3.10)


import os
import sys
import numpy as np
import time
import copy
from typing import Optional, Tuple
import open3d as o3d
import trimesh
from pathlib import Path
from scipy.spatial import cKDTree
base_dir = Path.cwd()
print(f'''Base dir: {base_dir}''')
sys.path.append(str(base_dir / 'openreal2sim' / 'motion' / 'modules'))
sys.path.append(str(base_dir / 'openreal2sim' / 'motion' / 'utils'))
sys.path.append(str(base_dir / 'openreal2sim' / 'motion'))
sys.path.append(str(base_dir / 'third_party'))
from grasp_utils import MyGraspGen, filter_grasps_by_score,  save_grasps_npz, load_grasps_npz
from compose_config import compose_configs
import argparse
import json
import yaml

# Fixed gripper config - always use franka_panda
# Note: This should be the full GraspGen model config, not just the gripper config
GRIPPER_CONFIG_PATH = str(base_dir / 'third_party' / 'GraspGen' / 'GraspGenModels' / 'checkpoints' / 'graspgen_franka_panda.yml')
GRIPPER_NAME = 'franka_panda'

def sample_surface(glb_path, n_points):
    '''Load mesh and sample surface points.'''
    mesh = trimesh.load_mesh(glb_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = list(mesh.geometry.values())[0]
    (pts, _) = trimesh.sample.sample_surface(mesh, n_points)
    return np.array(pts, dtype=np.float32)


def grasps_to_pointcloud_vis(grasps, gripper_name, pts_per_gripper=400, color=(1.0, 0.0, 0.0), background_mesh=None):
    """
    Convert grasps to point cloud for visualization using GraspGen gripper mesh.
    This shows the actual gripper shape (e.g., fingers/claws) instead of simple spheres.
    
    Args:
        grasps: Array of grasp poses, shape (N, 4, 4)
        gripper_name: Name of the gripper (e.g., 'franka_panda', 'robotiq_2f_140')
        pts_per_gripper: Number of points to sample from each gripper mesh
        color: RGB color tuple for visualization
        background_mesh: Optional background mesh (not used currently)
    
    Returns:
        o3d.geometry.PointCloud: Point cloud containing gripper meshes at grasp poses
    """
    sys.path.append(str(base_dir / 'third_party' / 'GraspGen'))
    from grasp_gen.robot import get_gripper_info
    
    if len(grasps) == 0:
        return o3d.geometry.PointCloud()
    
    gripper_info = get_gripper_info(gripper_name)
    gripper_mesh = gripper_info.visual_mesh
    
    # Sample points from gripper mesh
    gripper_pts, _ = trimesh.sample.sample_surface(gripper_mesh, pts_per_gripper)
    gripper_pts = np.array(gripper_pts, dtype=np.float32)
    
    # Transform gripper points for each grasp
    all_points = []
    all_colors = []
    color_array = np.array(color, dtype=np.float32)
    
    for grasp in grasps:
        grasp_transform = np.asarray(grasp, dtype=np.float32)
        # Transform points: [N, 3] -> [N, 4] (homogeneous) -> transform -> [N, 3]
        gripper_pts_homo = np.hstack([gripper_pts, np.ones((len(gripper_pts), 1))])
        transformed_pts = (grasp_transform @ gripper_pts_homo.T).T[:, :3]
        all_points.append(transformed_pts)
        all_colors.append(np.tile(color_array, (len(transformed_pts), 1)))
    
    # Combine all points
    combined_points = np.vstack(all_points)
    combined_colors = np.vstack(all_colors)
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(combined_points)
    pcd.colors = o3d.utility.Vector3dVector(combined_colors)
    
    return pcd


def save_vis_ply(points_xyz, grasps, scores, save_path, gripper_name=None, pts_per_gripper=400, cloud_color=(0.0, 1.0, 0.0), bite_points=None, grasp_color=(1.0, 0.0, 0.0), filtered_grasps=None, filtered_bite_points=None):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(np.tile(np.array(cloud_color, dtype=np.float32), (len(cloud.points), 1)))
    grasp_pcd = grasps_to_pointcloud_vis(grasps, gripper_name, pts_per_gripper=pts_per_gripper, color=grasp_color)
    merged = cloud + grasp_pcd
    
    # Add filtered grasps in yellow color for visualization
    if filtered_grasps is not None and len(filtered_grasps) > 0:
        filtered_grasp_pcd = grasps_to_pointcloud_vis(filtered_grasps, gripper_name, pts_per_gripper=pts_per_gripper, color=(1.0, 1.0, 0.0))  # Yellow
        merged = merged + filtered_grasp_pcd
    
    if bite_points is not None and len(bite_points) > 0:
        bite_pcd = o3d.geometry.PointCloud()
        bite_points_flat = bite_points.reshape(-1, 3)
        bite_pcd.points = o3d.utility.Vector3dVector(bite_points_flat.astype(np.float32))
        bite_pcd.colors = o3d.utility.Vector3dVector(np.tile(np.array([0, 0, 1], dtype=np.float32), (len(bite_points_flat), 1)))
        merged = merged + bite_pcd
    
    # Add filtered bite points in orange color
    if filtered_bite_points is not None and len(filtered_bite_points) > 0:
        filtered_bite_pcd = o3d.geometry.PointCloud()
        filtered_bite_points_flat = filtered_bite_points.reshape(-1, 3)
        filtered_bite_pcd.points = o3d.utility.Vector3dVector(filtered_bite_points_flat.astype(np.float32))
        filtered_bite_pcd.colors = o3d.utility.Vector3dVector(np.tile(np.array([1.0, 0.5, 0.0], dtype=np.float32), (len(filtered_bite_points_flat), 1)))  # Orange
        merged = merged + filtered_bite_pcd
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(save_path), merged)
    print(f'''[VIS] saved {save_path}''')


def compute_bite_point_to_grasp_point_distances(bite_points, grasp_point):
    if len(bite_points) == 0 or grasp_point is None:
        return np.zeros((0, 2), dtype=np.float32)
    grasp_point = np.asarray(grasp_point, dtype=np.float32).reshape(3)
    mid_points = (bite_points[:, 0] + bite_points[:, 1]) / 2
    distances = np.linalg.norm(mid_points - grasp_point, axis=1)
    return np.stack([
        distances,
        distances], axis=1).astype(np.float32)


def check_bite_points_cross_bbox(bite_points, object_bbox):
    if len(bite_points) == 0:
        return np.zeros(0, dtype=bool)
    bbox_min = np.asarray(object_bbox[0], dtype=np.float32)
    bbox_max = np.asarray(object_bbox[1], dtype=np.float32)
    N = len(bite_points)
    crosses = np.zeros(N, dtype=bool)
    for i in range(N):
        (p1, p2, p3, p4) = (bite_points[i, 0], bite_points[i, 1], bite_points[i, 2], bite_points[i, 3])
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-09:
            crosses[i] = False
            continue
        normal = normal / norm
        d = -np.dot(normal, p1)
        corners = np.array([
            [
                bbox_min[0],
                bbox_min[1],
                bbox_min[2]],
            [
                bbox_max[0],
                bbox_min[1],
                bbox_min[2]],
            [
                bbox_min[0],
                bbox_max[1],
                bbox_min[2]],
            [
                bbox_max[0],
                bbox_max[1],
                bbox_min[2]],
            [
                bbox_min[0],
                bbox_min[1],
                bbox_max[2]],
            [
                bbox_max[0],
                bbox_min[1],
                bbox_max[2]],
            [
                bbox_min[0],
                bbox_max[1],
                bbox_max[2]],
            [
                bbox_max[0],
                bbox_max[1],
                bbox_max[2]]])
        dists = np.dot(corners, normal) + d
        if np.any(dists >= 0):
            pass
        crosses[i] = np.any(dists <= 0)
    return crosses


def compute_bite_points_from_gripper(grasps, gripper_name):
    if len(grasps) == 0:
        return (np.zeros((0, 4, 3), dtype=np.float32), {
            'gripper_name': gripper_name })
    from grasp_gen.robot import get_gripper_info
    gripper_info = get_gripper_info(gripper_name)
    # Get control points (may be numpy array or torch tensor)
    control_points = gripper_info.control_points
    if hasattr(control_points, 'cpu'):
        control_points = control_points.cpu().numpy()
    control_points = np.asarray(control_points)
    control_points_transposed = control_points.T
    bite_points_local = control_points_transposed[:4, :3].astype(np.float32)
    bite_points_homo = np.hstack([
        bite_points_local,
        np.ones((4, 1))])
    N = len(grasps)
    bite_points_world = np.zeros((N, 4, 3), dtype=np.float32)
    for i in range(N):
        grasp_transform = np.asarray(grasps[i])
        grasp_transform_4x4 = grasp_transform.astype(np.float32)
        bite_points_transformed = (grasp_transform_4x4 @ bite_points_homo.T).T
        bite_points_world[i] = bite_points_transformed[:, :3]
    metadata = {
        'gripper_name': gripper_name }
    return (bite_points_world, metadata)


def process_one_object(gg_net, obj_idx, obj_dict, proposals_dir, n_points, overwrite, vis_pts_per_gripper=400, min_grasp_score=None, scene_mesh_path=None, object_bbox=None, grasp_point=None, num_grasps=200):
    '''
    Build full point cloud for one object, run GraspGen, export all candidates to NPZ,
    save a PLY visualization, and return the NPZ path.
    
    Returns:
        npz_path: Path to saved NPZ file, or None if failed
        grasps: Array of grasps (N, 4, 4), or None if failed
        scores: Array of scores (N,), or None if failed
        bite_points: Array of bite points (N, 4, 3), or None if failed
        bite_distance: Array of bite distances (N, 2), or None if failed
    '''
    glb_path = obj_dict['optimized']
    if glb_path is None or not os.path.exists(glb_path):
        print(f'''[WARN][{obj_dict['oid']}_{obj_dict['name']}] mesh not found -> {glb_path}''')
        return (None, None, None, None, None)
    
    npz_path = proposals_dir / f'''{obj_dict['oid']}_{obj_dict['name']}_grasp.npz'''
    vis_path = proposals_dir / f'''{obj_dict['oid']}_{obj_dict['name']}_grasp_viz.ply'''
    
    # Check if already exists and overwrite is False
    if npz_path.exists() and not overwrite:
        print(f'''[SKIP][{obj_dict['oid']}_{obj_dict['name']}] grasp file already exists -> {npz_path}''')
        return 
    
    # Sample point cloud from mesh
    pcs = sample_surface(glb_path, n_points=n_points)
    
    # Run grasp generation inference
    grasp_threshold = -1.0  # Use top grasps
    grasps, scores = gg_net.inference(pcs, num_grasps=num_grasps, grasp_threshold=grasp_threshold, scene_mesh_path=scene_mesh_path)
    
    if len(grasps) == 0:
        print(f'''[WARN][{obj_dict['oid']}_{obj_dict['name']}] zero grasps generated''')
        return (None, None, None, None, None)
    
    # Compute bite points for all grasps first (needed for filtering and saving filtered grasps)
    all_bite_points, _ = compute_bite_points_from_gripper(grasps, GRIPPER_NAME)
    
    # Filter by min score if specified
    score_filtered_grasps = None
    score_filtered_scores = None
    score_filtered_bite_points = None
    if min_grasp_score is not None:
        score_mask = scores >= min_grasp_score
        if np.any(~score_mask):
            score_filtered_grasps = grasps[~score_mask]
            score_filtered_scores = scores[~score_mask]
            score_filtered_bite_points = all_bite_points[~score_mask]
        grasps = grasps[score_mask]
        scores = scores[score_mask]
        bite_points = all_bite_points[score_mask]
        if len(grasps) == 0:
            print(f'''[WARN][{obj_dict['oid']}_{obj_dict['name']}] all grasps filtered out by min_score={min_grasp_score}''')
            # Save filtered grasps even if all are filtered
            if score_filtered_grasps is not None and len(score_filtered_grasps) > 0:
                filtered_npz_path = proposals_dir / f'''{obj_dict['oid']}_{obj_dict['name']}_grasp_filtered_out.npz'''
                save_grasps_npz(score_filtered_grasps, score_filtered_scores, filtered_npz_path, bite_points=score_filtered_bite_points)
            return (None, None, None, None, None)
    else:
        bite_points = all_bite_points
    
    # Filter by bbox if specified
    bbox_filtered_grasps = None
    bbox_filtered_scores = None
    bbox_filtered_bite_points = None
    if object_bbox is not None:
        crosses_bbox = check_bite_points_cross_bbox(bite_points, object_bbox)
        valid_mask = crosses_bbox  
        invalid_mask = ~crosses_bbox  
        
        # Save filtered out grasps
        if np.any(invalid_mask):
            bbox_filtered_grasps = grasps[invalid_mask]
            bbox_filtered_scores = scores[invalid_mask]
            bbox_filtered_bite_points = bite_points[invalid_mask]
        
        grasps = grasps[valid_mask]
        scores = scores[valid_mask]
        bite_points = bite_points[valid_mask]
        print(f"[BBOX] {len(grasps)}/{len(crosses_bbox)} grasps remaining")
        if len(grasps) == 0:
            print(f'''[WARN][{obj_dict['oid']}_{obj_dict['name']}] all grasps filtered out by bbox filter''')
            # Save filtered grasps even if all are filtered
            if bbox_filtered_grasps is not None and len(bbox_filtered_grasps) > 0:
                filtered_npz_path = proposals_dir / f'''{obj_dict['oid']}_{obj_dict['name']}_grasp_filtered_out.npz'''
                save_grasps_npz(bbox_filtered_grasps, bbox_filtered_scores, filtered_npz_path, bite_points=bbox_filtered_bite_points)
            return (None, None, None, None, None)
        print(f'''[INFO][{obj_dict['oid']}_{obj_dict['name']}] bbox filter: {len(grasps)}/{len(crosses_bbox)} grasps remaining''')
    
    # Compute bite distances if grasp_point is provided
    bite_distance = None
    if grasp_point is not None:
        bite_distance = compute_bite_point_to_grasp_point_distances(bite_points, grasp_point)

    # Save valid grasps to NPZ (with bite_points and bite_distance)
    save_grasps_npz(grasps, scores, npz_path, bite_points=bite_points, bite_distance=bite_distance)
    
    # Save filtered out grasps (combine score-filtered and bbox-filtered)
    filtered_grasps = None
    filtered_scores = None
    filtered_bite_points = None
    if score_filtered_grasps is not None or bbox_filtered_grasps is not None:
        filtered_grasps_list = []
        filtered_scores_list = []
        filtered_bite_points_list = []
        # if score_filtered_grasps is not None and len(score_filtered_grasps) > 0:
        #     filtered_grasps_list.append(score_filtered_grasps)
        #     filtered_scores_list.append(score_filtered_scores)
        #     filtered_bite_points_list.append(score_filtered_bite_points)
        if bbox_filtered_grasps is not None and len(bbox_filtered_grasps) > 0:
            filtered_grasps_list.append(bbox_filtered_grasps)
            filtered_scores_list.append(bbox_filtered_scores)
            filtered_bite_points_list.append(bbox_filtered_bite_points)
        
        if filtered_grasps_list:
            filtered_grasps = np.vstack(filtered_grasps_list)
            filtered_scores = np.concatenate(filtered_scores_list)
            filtered_bite_points = np.vstack(filtered_bite_points_list)
            
            filtered_npz_path = proposals_dir / f'''{obj_dict['oid']}_{obj_dict['name']}_grasp_filtered_out.npz'''
            save_grasps_npz(filtered_grasps, filtered_scores, filtered_npz_path, bite_points=filtered_bite_points)
            print(f'''[OK][{obj_dict['oid']}_{obj_dict['name']}] saved {len(filtered_grasps)} filtered grasps -> {filtered_npz_path.name}''')
    
    # Save visualization (include filtered grasps if available)
    # Use bbox_filtered for visualization (they are the ones shown in yellow)
    if bbox_filtered_grasps is not None and len(bbox_filtered_grasps) > 0:
        save_vis_ply(pcs, grasps[:10], scores[:10], vis_path, gripper_name=GRIPPER_NAME, pts_per_gripper=vis_pts_per_gripper, 
                 bite_points=bite_points[:10], filtered_grasps=bbox_filtered_grasps[:10], filtered_bite_points=bbox_filtered_bite_points[:10])
    else:
        print(f"[INFO] No bbox filtered grasps.")
        save_vis_ply(pcs, grasps[:10], scores[:10], vis_path, gripper_name=GRIPPER_NAME, pts_per_gripper=vis_pts_per_gripper, 
                 bite_points=bite_points[:10], filtered_grasps=None, filtered_bite_points=None)

    print(f'''[OK][{obj_dict['oid']}_{obj_dict['name']}] saved {len(grasps)} grasps -> {npz_path.name}''')
    
    return (str(npz_path), grasps, scores, bite_points, bite_distance)


def run_for_key(key, n_points, overwrite, vis_pts_per_gripper, filter_collisions=False, collision_threshold=0.002, min_grasp_score=None, num_grasps=200):
    base_dir = Path.cwd()
    out_dir = base_dir / 'outputs'
    scene_json = out_dir / key / 'simulation' / 'scene.json'
    if not scene_json.exists():
        raise FileNotFoundError(scene_json)
    scene_dict = json.load(open(scene_json))
    objects = scene_dict.get('objects', {})
    if not isinstance(objects, dict) or len(objects) == 0:
        print(f'''[WARN][{key}] scene_dict[\'objects\'] is empty.''')
        return None
    scene_mesh_path = None
    if filter_collisions:
        if 'scene_mesh' in scene_dict and 'optimized' in scene_dict['scene_mesh']:
            scene_mesh_path = scene_dict['scene_mesh']['optimized']
        elif 'background' in scene_dict:
            bg_data = scene_dict['background']
            if not bg_data.get('registered'):
                pass
            scene_mesh_path = bg_data.get('original')
        if scene_mesh_path:
            if not os.path.isabs(scene_mesh_path):
                scene_mesh_path = str(scene_json.parent.parent / scene_mesh_path)
            if os.path.exists(scene_mesh_path):
                print(f'''[INFO] Using scene mesh for collision detection: {scene_mesh_path}''')
            else:
                print(f'''[WARN] Scene mesh path not found: {scene_mesh_path}, will use point cloud fallback''')
                scene_mesh_path = None
    key_dir = out_dir / key
    proposals_dir = key_dir / 'grasps'
    proposals_dir.mkdir(parents=True, exist_ok=True)
    # Initialize network with fixed franka_panda gripper config
    # Pass num_grasps as num_grasps_per_object to configure the model
    if num_grasps is None:
        num_grasps = 10000
    net = MyGraspGen(gripper_config=GRIPPER_CONFIG_PATH, filter_collisions=filter_collisions, collision_threshold=collision_threshold, num_grasps_per_object=num_grasps)
    manipulated_oid = scene_dict['manipulated_oid']
    
    # Store gripper_config_path in net for later use
    net.gripper_config = GRIPPER_CONFIG_PATH
    
    # Process each object
    for obj_idx, obj_dict in objects.items():
        # Get object bbox and grasp point if available
        object_bbox = obj_dict.get('bbox')
        grasp_point = obj_dict.get('grasp_point')
        
        npz_path, grasps, scores, bite_points, bite_distances = process_one_object(
            gg_net=net,
            obj_idx=obj_idx,
            obj_dict=obj_dict,
            proposals_dir=proposals_dir,
            n_points=n_points,
            overwrite=overwrite,
            vis_pts_per_gripper=vis_pts_per_gripper,
            min_grasp_score=min_grasp_score,
            scene_mesh_path=scene_mesh_path,
            object_bbox=object_bbox,
            grasp_point=grasp_point,
            num_grasps=num_grasps
        )
        
        if npz_path is not None:
            # Update scene.json with grasp path
            scene_dict['objects'][obj_idx]['grasps'] = npz_path
            with open(scene_json, 'w') as f:
                json.dump(scene_dict, f, indent=2)
            print(f'''[OK][{key}] scene.json updated with 'grasps' for {obj_dict['oid']}_{obj_dict['name']}.''')


def grasp_generation(key=None):
    '''Process a single key or list of keys for grasp generation.'''
    cfg_path = Path.cwd() / 'config' / 'config.yaml'
    cfg = yaml.safe_load(cfg_path.open('r'))
    
    # Handle both single key and list of keys
    if isinstance(key, list):
        keys = key
    elif key is not None:
        keys = [key]
    else:
        # If key is None, get keys from config
        keys = cfg.get("keys", [])
    
    for single_key in keys:
        key_cfg = compose_configs(single_key, cfg)
        print(f'''\n========== [GraspGen] Processing key: {single_key} ==========''')
        run_for_key(single_key, key_cfg['n_points'], key_cfg['overwrite'], key_cfg['vis_pts_per_gripper'], key_cfg.get('filter_collisions', True), key_cfg.get('collision_threshold', 0.002), key_cfg.get('min_grasp_score', 0), key_cfg.get('keep', 200))
        print(f'''[Info] Grasp generation completed for key: {single_key}.''')


def main():
    parser = argparse.ArgumentParser('Export grasp proposals for a single key using GraspGen and write paths into scene.json')
    parser.add_argument('--key', type=str, required=True, help='Key to process (required)')
    parser.add_argument('--n_points', type=int, default=None, help='Override n_points from config')
    parser.add_argument('--overwrite', action='store_true', help='Override overwrite from config')
    parser.add_argument('--vis_pts_per_gripper', type=int, default=None, help='Override vis_pts_per_gripper from config')
    parser.add_argument('--filter_collisions', action='store_true', help='Enable collision filtering for grasps')
    parser.add_argument('--collision_threshold', type=float, default=0.002, help='Distance threshold for collision detection (in meters)')
    parser.add_argument('--min_grasp_score', type=float, default=None, help='Minimum grasp score threshold (filter out low-score grasps)')
    parser.add_argument('--num_grasps', type=int, default=None, help='Number of grasps to generate (override config keep)')
    
    args = parser.parse_args()
    cfg_path = Path.cwd() / 'config' / 'config.yaml'
    cfg = yaml.safe_load(cfg_path.open('r'))
    key_cfg = compose_configs(args.key, cfg)
    n_points = args.n_points if args.n_points is not None else key_cfg['n_points']
    overwrite = args.overwrite if args.overwrite else key_cfg['overwrite']
    vis_pts_per_gripper = args.vis_pts_per_gripper if args.vis_pts_per_gripper is not None else key_cfg['vis_pts_per_gripper']
    
    filter_collisions = args.filter_collisions if args.filter_collisions else key_cfg.get('filter_collisions', False)
    collision_threshold = args.collision_threshold if args.collision_threshold is not None else key_cfg.get('collision_threshold', 0.002)
    min_grasp_score = args.min_grasp_score if args.min_grasp_score is not None else key_cfg.get('min_grasp_score')
    num_grasps = args.num_grasps if args.num_grasps is not None else key_cfg.get('keep', 200)
    print(f'''\n========== [GraspGen] Processing key: {args.key} ==========''')
    print(f'''[Config] filter_collisions={filter_collisions}, collision_threshold={collision_threshold}''')
    print(f'''[Config] min_grasp_score={min_grasp_score}, num_grasps={num_grasps}, gripper_config={GRIPPER_CONFIG_PATH} (fixed: franka_panda)''')
    run_for_key(args.key, n_points, overwrite, vis_pts_per_gripper, filter_collisions, collision_threshold, min_grasp_score, num_grasps)

if __name__ == '__main__':
    main()
