# Source Generated with Decompyle++
# File: grasp_utils.cpython-310.pyc (Python 3.10)

import os
import sys
import numpy as np
import time
import torch
from typing import List, Tuple, Optional
from pathlib import Path
base_dir = Path.cwd()
sys.path.append(str(base_dir / 'third_party' / 'GraspGen'))

class MyGraspGen:
    '''
    Wrapper class for GraspGen that provides a simple interface for grasp generation.
    Uses GraspGen for grasp generation, returning native GraspGen format (4x4 transforms + confidence).
    '''
    
    def __init__(self, gripper_config=None, filter_collisions=False, collision_threshold=0.002, num_grasps_per_object=None):
        '''
        Initialize GraspGen sampler.
        
        Args:
            gripper_config: Path to gripper configuration YAML file. If None, will try to find default.
            filter_collisions: Whether to filter grasps based on collision detection
            collision_threshold: Distance threshold for collision detection (in meters)
            num_grasps_per_object: Number of grasps to generate per object. If None, uses value from config.
        '''
        from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
        from grasp_gen.robot import get_gripper_info
        
        # Load grasp configuration
        # Note: load_grasp_cfg expects a full GraspGen model config file, not just a gripper config
        if gripper_config is None:
            # Try to find default full model config (not just gripper config)
            default_config_path = base_dir / 'third_party' / 'GraspGen' / 'GraspGenModels' / 'checkpoints' / 'graspgen_franka_panda.yml'
            if default_config_path.exists():
                gripper_config = str(default_config_path)
            else:
                raise ValueError("gripper_config must be provided if no default config found")
        
        grasp_cfg = load_grasp_cfg(gripper_config)
        
        # Override num_grasps_per_object if provided
        if num_grasps_per_object is not None:
            grasp_cfg.diffusion.num_grasps_per_object = num_grasps_per_object
        
        # Initialize GraspGen sampler
        self.grasp_sampler = GraspGenSampler(grasp_cfg)
        
        # Get gripper info for collision detection
        gripper_name = grasp_cfg.data.gripper_name
        gripper_info = get_gripper_info(gripper_name)
        
        # Store collision detection settings
        self.filter_collisions = filter_collisions
        self.collision_threshold = collision_threshold
        # Always load collision mesh, but only use it if filter_collisions is True
        self.gripper_collision_mesh = gripper_info.collision_mesh

    
    def inference(self, pcs, num_grasps=10000, grasp_threshold=-1, scene_mesh_path=None):
        '''
        Run grasp generation inference on point cloud.
        
        Args:
            pcs: Point cloud array of shape (N, 3) - object point cloud
            num_grasps: Number of grasps to generate
            grasp_threshold: Threshold for valid grasps. If -1.0, returns top grasps.
            scene_mesh_path: Optional path to scene mesh file for collision detection (if filter_collisions=True)
        
        Returns:
            grasps: Array of shape (M, 4, 4) containing grasp poses as 4x4 transformation matrices
            scores: Array of shape (M,) containing grasp confidence scores
        '''
        from grasp_gen.grasp_server import GraspGenSampler
        topk_num_grasps = 10000 if grasp_threshold == -1 else -1
        (grasps_tensor, grasp_conf_tensor) = GraspGenSampler.run_inference(pcs, self.grasp_sampler, grasp_threshold=grasp_threshold, num_grasps=num_grasps, topk_num_grasps=topk_num_grasps, remove_outliers=True)
        if len(grasps_tensor) == 0:
            return (np.array([]).reshape(0, 4, 4), np.array([]))
        grasps_np = grasps_tensor.cpu().numpy()
        grasp_conf_np = grasp_conf_tensor.cpu().numpy()
        grasps_np[:, 3, 3] = 1
        if self.filter_collisions and self.gripper_collision_mesh is not None:
            if scene_mesh_path is not None and os.path.exists(scene_mesh_path):
                import trimesh
                from grasp_gen.dataset.eval_utils import check_collision
                scene_mesh = trimesh.load(scene_mesh_path, force='mesh')
                if isinstance(scene_mesh, trimesh.Scene):
                    scene_mesh = scene_mesh.dump(concatenate=True)
                collision_mask = check_collision(scene_mesh=scene_mesh, object_mesh=self.gripper_collision_mesh, transforms=grasps_np)
                collision_free_mask = ~collision_mask
                grasps_np = grasps_np[collision_free_mask]
                grasp_conf_np = grasp_conf_np[collision_free_mask]
                print(f'''[Collision] Using scene mesh: {scene_mesh_path}''')
                print(f'''[Collision] Filtered to {len(grasps_np)}/{len(collision_mask)} collision-free grasps''')
                return (grasps_np, grasp_conf_np)
            if scene_mesh_path is not None:
                print(f'''[WARN] Scene mesh not found: {scene_mesh_path}, falling back to point cloud collision detection''')
            from grasp_gen.utils.point_cloud_utils import filter_colliding_grasps
            collision_free_mask = filter_colliding_grasps(scene_pc=pcs, grasp_poses=grasps_np, gripper_collision_mesh=self.gripper_collision_mesh, collision_threshold=self.collision_threshold)
            grasps_np = grasps_np[collision_free_mask]
            grasp_conf_np = grasp_conf_np[collision_free_mask]
            print(f'''[Collision] Using point cloud, filtered to {len(grasps_np)}/{len(collision_free_mask)} collision-free grasps''')
        return (grasps_np, grasp_conf_np)



def get_best_grasp_with_hints(grasps = None, scores = None, point = None, direction = (None, None)):
    '''
    Rescore all grasps using optional spatial and directional hints, then return sorted grasps.
    
    Args:
        grasps: Array of shape (N, 4, 4) containing grasp poses
        scores: Array of shape (N,) containing grasp confidence scores
        point: (3,) world point. If provided, grasps closer to this point are preferred.
        direction: (3,) world direction. If provided, grasps whose approach (+X) aligns
                   with this direction are preferred.
    
    Returns:
        grasps_sorted: Sorted grasps array (best first)
        scores_sorted: Sorted scores array
    '''
    if len(grasps) == 0:
        return (grasps, scores)
    N = len(grasps)
    w_dir = 0.3 if direction is not None else 0
    w_pt = 0.3 if point is not None else 0
    w_net = 1
    if scores.max() > scores.min():
        net_term = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        net_term = np.ones_like(scores)
    translations = grasps[:, :3, 3]
    rotations = grasps[:, :3, :3]
    approach_axes = rotations[:, :, 0]
    approach_axes = approach_axes / (np.linalg.norm(approach_axes, axis=1, keepdims=True) + 1e-12)
    if w_dir > 0:
        d = np.asarray(direction, dtype=np.float64)
        d = d / (np.linalg.norm(d) + 1e-12)
        cosv = np.clip(np.sum(approach_axes * d[None, :], axis=1), -1, 1)
        dir_term = 0.5 * (cosv + 1)
    else:
        dir_term = np.zeros(N, dtype=np.float64)
    if w_pt > 0:
        p = np.asarray(point, dtype=np.float64).reshape(1, 3)
        dists = np.linalg.norm(translations - p, axis=1)
        sigma = 0.05
        sigma2 = max(sigma * sigma, 1e-12)
        pt_term = np.exp(-0.5 * dists * dists / sigma2)
    else:
        pt_term = np.zeros(N, dtype=np.float64)
    total_scores = w_dir * dir_term + w_pt * pt_term + w_net * net_term
    sorted_indices = np.argsort(total_scores)[::-1]
    return (grasps[sorted_indices], total_scores[sorted_indices])


def filter_grasps_by_score(grasps = None, scores = None, min_score = None):
    '''Filter grasps by minimum score threshold.'''
    mask = scores >= min_score
    return (grasps[mask], scores[mask])


def get_top_k_grasps(grasps = None, scores = None, k = None):
    '''Get top k grasps by score.'''
    if len(grasps) == 0:
        return (grasps, scores)
    sorted_indices = np.argsort(scores)[::-1]
    top_k = min(k, len(grasps))
    return (grasps[sorted_indices[:top_k]], scores[sorted_indices[:top_k]])


def save_grasps_npz(grasps = None, scores = None, save_path = None, bite_points = None, bite_distance = None):
    '''
    Save grasps in NPZ format compatible with GraspGen.
    
    Args:
        grasps: Array of shape (N, 4, 4) containing grasp poses
        scores: Array of shape (N,) containing grasp confidence scores
        save_path: Path to save the NPZ file
        bite_points: Optional array of shape (N, 4, 3) containing bite points
        bite_distances: Optional array of shape (N, 2) containing bite distances
    '''
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        'grasps': grasps.astype(np.float32),
        'scores': scores.astype(np.float32)
    }
    if bite_distance is not None:
        save_dict['bite_distance'] = bite_distance.astype(np.float32)
    np.savez(str(save_path), **save_dict)
    print(f'''[OK] Saved {len(grasps)} grasps to {save_path}''')


def load_grasps_npz(load_path = None):
    '''
    Load grasps from NPZ file.
    
    Args:
        load_path: Path to the NPZ file
    
    Returns:
        grasps: Array of shape (N, 4, 4) containing grasp poses
        scores: Array of shape (N,) containing grasp confidence scores
        bite_points: Array of shape (N, 4, 3) containing bite points (if available)
        bite_distances: Array of shape (N, 2) containing bite distances (if available)
    '''
    data = np.load(str(load_path))
    grasps = data['grasps']
    scores = data['scores']
    bite_distances = data.get('bite_distances', None)
    return (grasps, scores, bite_distances)

