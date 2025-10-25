#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trajectory preprocessing to determine trajectory properties.
Inputs:
    - outputs/{key_name}/scene/scene.pkl
Outputs:
    - Updated scene.pkl with trajectory properties
Note:
    - Determines if trajectory is object-centric
    - Checks for objects underneath manipulated objects at start
    - Updates trajectory metadata
"""
import os
from pathlib import Path
import numpy as np
import trimesh
import pickle
import yaml
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import cdist

# ──────────────── utils ──────────────── #
def trimesh_single(path):
    """Load single mesh from path"""
    m = trimesh.load(path, force="mesh")
    if isinstance(m, trimesh.Scene):
        m = list(m.geometry.values())[0]
    return m

def get_mesh_center(mesh_path: str) -> np.ndarray:
    """Get center point of mesh"""
    mesh = trimesh_single(mesh_path)
    return mesh.centroid

def get_mesh_bounds(mesh_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Get mesh bounding box"""
    mesh = trimesh_single(mesh_path)
    return mesh.bounds[0], mesh.bounds[1]  # min, max

def check_objects_underneath(manipulated_obj: Dict, other_objects: Dict, 
                           threshold: float = 0.1) -> List[str]:
    """
    Check if there are objects underneath the manipulated object at start
    
    Args:
        manipulated_obj: Manipulated object info
        other_objects: Other objects dict (static + dynamic)
        threshold: Distance threshold for "underneath" detection
        
    Returns:
        List of object IDs that are underneath
    """
    underneath_objects = []
    
    # Get manipulated object start position and bounds
    start_pose = manipulated_obj["abs_trajs"][0]  # [4, 4]
    start_pos = start_pose[:3, 3]
    
    # Get manipulated object mesh bounds
    mesh_path = manipulated_obj["registered"]
    mesh = trimesh_single(mesh_path)
    mesh_bounds = mesh.bounds
    
    # Transform bounds to world coordinates
    mesh_min = start_pose @ np.append(mesh_bounds[0], 1)
    mesh_max = start_pose @ np.append(mesh_bounds[1], 1)
    
    # Check all other objects
    for oid, obj in other_objects.items():
        if obj["type"] == "static":
            # For static objects, get object bounds
            static_mesh = trimesh_single(obj["registered"])
            static_bounds = static_mesh.bounds
            obj_min = np.array(obj["object_center"]) + static_bounds[0] - static_mesh.centroid
            obj_max = np.array(obj["object_center"]) + static_bounds[1] - static_mesh.centroid
        elif obj["type"] == "dynamic":
            # For dynamic objects, get bounds at start position
            obj_start_pose = obj["abs_trajs"][0]
            obj_mesh = trimesh_single(obj["registered"])
            obj_bounds = obj_mesh.bounds
            # Transform bounds to world coordinates
            obj_min = obj_start_pose @ np.append(obj_bounds[0], 1)
            obj_max = obj_start_pose @ np.append(obj_bounds[1], 1)
        else:
            continue
            
        # Check if object is underneath (lower Z and X-Y overlap)
        # X-Y overlap: check if bounding boxes intersect
        x_overlap = not (obj_max[0] < mesh_min[0] or obj_min[0] > mesh_max[0])
        y_overlap = not (obj_max[1] < mesh_min[1] or obj_min[1] > mesh_max[1])
        z_underneath = obj_max[2] < start_pos[2]  # Object is below manipulated object
        
        if z_underneath and x_overlap and y_overlap:
            underneath_objects.append(oid)
    
    return underneath_objects

def find_nearest_object_to_manipulated_end(manipulated_obj: Dict, other_objects: Dict, 
                                          threshold: float = 0.2) -> Optional[str]:
    """
    Find the nearest object to the manipulated object's end position
    
    Args:
        manipulated_obj: Manipulated object info
        other_objects: Other objects dict (static + dynamic)
        threshold: Distance threshold for "near" detection
        
    Returns:
        ID of the nearest object
    """
    # Get manipulated object end position
    end_pose = manipulated_obj["abs_trajs"][-1]  # [4, 4]
    end_pos = end_pose[:3, 3]
    
    nearest_obj = None
    min_dist = float('inf')
    
    # Check all other objects
    for oid, obj in other_objects.items():
        if obj["type"] == "static":
            # For static objects, use object center
            obj_pos = np.array(obj["object_center"])
        elif obj["type"] == "dynamic":
            # For dynamic objects, use end position
            obj_end_pose = obj["abs_trajs"][-1]
            obj_pos = obj_end_pose[:3, 3]
        else:
            continue
            
        dist = np.linalg.norm(end_pos - obj_pos)
        if dist < min_dist and dist < threshold:
            min_dist = dist
            nearest_obj = oid
    
    return nearest_obj

def is_object_centric(manipulated_obj: Dict, other_objects: Dict, 
                     threshold: float = 0.2) -> bool:
    """
    Determine if trajectory is object-centric
    
    Args:
        manipulated_obj: Manipulated object info
        other_objects: Other objects dict (static + dynamic)
        threshold: Distance threshold for "near" detection
        
    Returns:
        True if trajectory is object-centric
    """
    # Find nearest object to manipulated object's end position
    nearest_obj = find_nearest_object_to_manipulated_end(
        manipulated_obj, other_objects, threshold
    )
    
    # If manipulated object ends near any other object, it's object-centric
    return nearest_obj is not None

# ──────────────── main function ──────────────── #
def traj_preprocess(keys, key_scene_dicts, key_cfgs):
    """
    Main function to preprocess trajectories and determine their properties
    
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
        
        # Get configuration parameters
        distance_threshold = key_cfg.get("traj_distance_threshold", 0.2)
        underneath_threshold = key_cfg.get("traj_underneath_threshold", 0.1)
        
        # Get objects
        objects = scene_dict["info"]["objects"]
        other_objects = {oid: obj for oid, obj in objects.items() if obj["type"] in ["static", "dynamic"]}
        manipulated_objects = {oid: obj for oid, obj in objects.items() if obj["type"] == "manipulated"}
        
        # Process each manipulated object
        for oid, manipulated_obj in manipulated_objects.items():
            print(f"[Info] [{key}] Processing manipulated object {oid}...")
            
            # 1. Check if trajectory is object-centric
            is_centric = is_object_centric(manipulated_obj, other_objects, distance_threshold)
            
            # 2. Check for objects underneath at start
            underneath_objects = check_objects_underneath(manipulated_obj, other_objects, underneath_threshold)
            
            # 3. Find nearest object to end position
            nearest_obj = find_nearest_object_to_manipulated_end(
                manipulated_obj, other_objects, distance_threshold
            )
            
            # 4. Update object metadata
            manipulated_obj["traj_properties"] = {
                "is_object_centric": is_centric,
                "underneath_objects": underneath_objects,
                "nearest_object_end": nearest_obj,
                "distance_threshold": distance_threshold,
                "underneath_threshold": underneath_threshold
            }
            
            print(f"[Info] [{key}] Object {oid} properties:")
            print(f"  - Object-centric: {is_centric}")
            print(f"  - Objects underneath: {underneath_objects}")
            print(f"  - Nearest object at end: {nearest_obj}")
        
        # Update scene dictionary
        scene_dict["info"]["objects"] = objects
        
        # Save updated scene data
        scene_pkl_path = base_dir / f'outputs/{key}/scene/scene.pkl'
        with open(scene_pkl_path, 'wb') as f:
            pickle.dump(scene_dict, f)
        
        print(f"[Info] [{key}] Trajectory preprocessing completed")

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
    
    traj_preprocess(keys, key_scene_dicts, key_cfgs)