#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize the whole trajectory in one PLY file.
Inputs:
    - outputs/{key_name}/scene/scene.pkl
Outputs:
    - outputs/{key_name}/visualization/trajectory.ply
Note:
    - Combines background static mesh with all frames of dynamic/manipulated objects
    - Each frame and object type has different colors for visualization
"""
import os
from pathlib import Path
import numpy as np
import trimesh
import pickle
import yaml
from tqdm import trange
import argparse
from typing import Dict, List, Tuple, Optional

# ──────────────── utils ──────────────── #
def trimesh_single(path):
    """Load single mesh from path"""
    m = trimesh.load(path, force="mesh")
    if isinstance(m, trimesh.Scene):
        m = list(m.geometry.values())[0]
    return m

def ensure_unit_normal(n):
    """Ensure normal vector is unit length"""
    n = np.asarray(n, dtype=np.float64)
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        raise ValueError("Plane normal has near-zero length.")
    return n / norm

def create_colored_mesh(mesh: trimesh.Trimesh, color: Tuple[float, float, float]) -> trimesh.Trimesh:
    colored_mesh = mesh.copy()
    colored_mesh.visual.face_colors = [int(c * 255) for c in color] + [255]  # RGB + Alpha
    return colored_mesh

def transform_mesh(mesh: trimesh.Trimesh, pose: np.ndarray) -> trimesh.Trimesh:
    transformed_mesh = mesh.copy()
    transformed_mesh.apply_transform(pose)
    return transformed_mesh

# ──────────────── trajectory visualization ──────────────── #
def extract_trajectory_data(scene_dict: Dict) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Extract trajectory data from scene dictionary
    
    Args:
        scene_dict: Scene data dictionary
        
    Returns:
        dynamic_meshes: List of dynamic object mesh paths
        manipulated_meshes: List of manipulated object mesh paths  
        poses: Array of poses [T, N, 4, 4] where T is frames, N is objects
    """
    objects = scene_dict["info"]["objects"]
    dynamic_meshes = []
    manipulated_meshes = []
    poses = []
    
    # Extract mesh paths and poses
    for oid, obj in objects.items():
        if obj["type"] == "dynamic":
            dynamic_meshes.append(obj["registered"])
            poses.append(obj["abs_trajs"])  # [T, 4, 4]
        elif obj["type"] == "manipulated":
            manipulated_meshes.append(obj["registered"])
            poses.append(obj["abs_trajs"])  # [T, 4, 4]
    
    # Convert poses to numpy array [T, N, 4, 4]
    if poses:
        poses = np.array(poses).transpose(1, 0, 2, 3)  # [T, N, 4, 4]
    else:
        poses = np.array([])
    
    return dynamic_meshes, manipulated_meshes, poses

def visualize_trajectory_single_scene(scene_dict: Dict, output_path: str, 
                                    background_color: Tuple[float, float, float] = (0.8, 0.8, 0.8),
                                    dynamic_colors: List[Tuple[float, float, float]] = None,
                                    manipulated_colors: List[Tuple[float, float, float]] = None):
    """
    Visualize trajectory for a single scene
    
    Args:
        scene_dict: Scene data dictionary
        output_path: Output PLY file path
        background_color: Background color (R, G, B)
        dynamic_colors: Dynamic object colors
        manipulated_colors: Manipulated object colors
    """
    # Default colors
    if dynamic_colors is None:
        dynamic_colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]  # 红绿蓝
    if manipulated_colors is None:
        manipulated_colors = [(1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0)]  # 黄紫青
    
    # 1. Load background mesh
    print("[Info] Loading background mesh...")
    background_mesh_path = scene_dict["info"]["background"]["registered"]
    background_mesh = trimesh_single(background_mesh_path)
    background_mesh = create_colored_mesh(background_mesh, background_color)
    
    # 2. Extract trajectory data
    print("[Info] Extracting trajectory data...")
    dynamic_meshes, manipulated_meshes, poses = extract_trajectory_data(scene_dict)
    
    if len(poses) == 0:
        print("[Warning] No trajectory data found, only background will be exported")
        combined_mesh = background_mesh
    else:
        # 3. Load object meshes
        print("[Info] Loading object meshes...")
        dynamic_mesh_objects = [trimesh_single(path) for path in dynamic_meshes]
        manipulated_mesh_objects = [trimesh_single(path) for path in manipulated_meshes]
        
        # 4. Process each frame
        all_meshes = [background_mesh]  # Start with background
        num_frames, num_objects = poses.shape[:2]
        
        print(f"[Info] Processing {num_frames} frames with {num_objects} objects...")
        for frame_idx in trange(num_frames, desc="Processing frames"):
            # Process dynamic objects
            for obj_idx, mesh in enumerate(dynamic_mesh_objects):
                if obj_idx < num_objects:
                    pose = poses[frame_idx, obj_idx]
                    transformed_mesh = transform_mesh(mesh, pose)
                    color = dynamic_colors[obj_idx % len(dynamic_colors)]
                    colored_mesh = create_colored_mesh(transformed_mesh, color)
                    all_meshes.append(colored_mesh)
            
            # Process manipulated objects
            for obj_idx, mesh in enumerate(manipulated_mesh_objects):
                pose_idx = len(dynamic_mesh_objects) + obj_idx
                if pose_idx < num_objects:
                    pose = poses[frame_idx, pose_idx]
                    transformed_mesh = transform_mesh(mesh, pose)
                    color = manipulated_colors[obj_idx % len(manipulated_colors)]
                    colored_mesh = create_colored_mesh(transformed_mesh, color)
                    all_meshes.append(colored_mesh)
        
        # 5. Merge all meshes
        print("[Info] Merging all meshes...")
        combined_mesh = trimesh.util.concatenate(all_meshes)
    
    # 6. Export PLY file
    print(f"[Info] Exporting to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined_mesh.export(output_path)
    
    print(f"[Info] Trajectory visualization saved to: {output_path}")
    print(f"[Info] Total vertices: {len(combined_mesh.vertices)}")
    print(f"[Info] Total faces: {len(combined_mesh.faces)}")

# ──────────────── main function ──────────────── #
def visualize_trajectory(keys, key_scene_dicts, key_cfgs):
    """
    Main function to visualize trajectories for multiple scenes
    
    Args:
        keys: List of scene keys
        key_scene_dicts: Dictionary mapping keys to scene data
        key_cfgs: Dictionary mapping keys to configurations
    """
    base_dir = Path.cwd()
    
    for key in keys:
        scene_dict = key_scene_dicts[key]
        key_cfg = key_cfgs[key]
        
        print(f"[Info] Processing {key}...")
        
        # Get output path from config or use default
        output_dir = base_dir / f"outputs/{key}/visualization"
        output_path = output_dir / "trajectory.ply"
        
        # Get color settings from config
        background_color = key_cfg.get("background_color", (0.8, 0.8, 0.8))
        dynamic_colors = key_cfg.get("dynamic_colors", None)
        manipulated_colors = key_cfg.get("manipulated_colors", None)
        
        # Visualize trajectory
        visualize_trajectory_single_scene(
            scene_dict, 
            str(output_path),
            background_color=background_color,
            dynamic_colors=dynamic_colors,
            manipulated_colors=manipulated_colors
        )
        
        print(f"[Info] [{key}] Trajectory visualization completed")

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
    
    visualize_trajectory(keys, key_scene_dicts, key_cfgs)