"""
Light-weight GraspGroup implementation.

Provides a minimal GraspGroup class that can load .npy/.npz files with grasps,
scores, and expose rotation/translation information for manipulation scripts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union, Optional

import numpy as np
import transforms3d


@dataclass
class SimpleGrasp:
    rotation_matrix: np.ndarray  # (3, 3) - in world frame (already converted)
    translation: np.ndarray  # (3,) - in world frame
    score: float = 0.0
    bite_distances: Optional[np.ndarray] = None  # (2,) - distance to grasp point for each fingertip
 
    def transform(self, new_transform: np.ndarray):
        """Transform a grasp in place."""
        total_matrix = np.eye(4, dtype=np.float32)
        total_matrix[:3, :3] = self.rotation_matrix
        total_matrix[:3, 3] = self.translation
        total_matrix = new_transform @ total_matrix
        self.rotation_matrix = total_matrix[:3, :3]
        self.translation = total_matrix[:3, 3]
        return self
    
    def to_world_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert grasp to world pose (position, quaternion).
        
        Returns:
            tuple: (position, quaternion) in world frame
                position: (3,) np.ndarray
                quaternion: (4,) np.ndarray (wxyz format)
        """
        q = transforms3d.quaternions.mat2quat(self.rotation_matrix).astype(np.float32)
        return self.translation.copy(), q


class GraspGroup:
    """GraspGroup implementation with world frame conversion on load."""

    def __init__(
        self, 
        rotations: Optional[np.ndarray] = None,
        translations: Optional[np.ndarray] = None,
        scores: Optional[np.ndarray] = None,
        bite_distances: Optional[np.ndarray] = None
    ):
        """
        Initialize GraspGroup.
        
        Args:
            rotations: (N, 3, 3) rotation matrices in world frame
            translations: (N, 3) translations in world frame
            scores: (N,) grasp scores
            bite_distances: (N, 2) distances to grasp point
        """
        if rotations is None:
            self._rotations = np.zeros((0, 3, 3), dtype=np.float32)
            self._translations = np.zeros((0, 3), dtype=np.float32)
            self._scores = np.zeros(0, dtype=np.float32)
        else:
            self._rotations = np.asarray(rotations, dtype=np.float32)
            self._translations = np.asarray(translations, dtype=np.float32)
            self._scores = np.asarray(scores, dtype=np.float32) if scores is not None else np.zeros(len(rotations), dtype=np.float32)
        
        self._bite_distances = bite_distances
        
        if len(self._rotations) > 0:
            if self._bite_distances is not None and len(self._bite_distances) != len(self._rotations):
                print(f"[WARN] Mismatch: {len(self._rotations)} grasps but {len(self._bite_distances)} bite_distances. Ignoring bite_distances.")
                self._bite_distances = None

    def __len__(self) -> int:
        return len(self._rotations)

    def __getitem__(
        self, index: Union[int, slice, Sequence[int], np.ndarray]
    ) -> Union[SimpleGrasp, "GraspGroup"]:
        if isinstance(index, int):
            bite_dists = None
            if self._bite_distances is not None and len(self._bite_distances) > index:
                bite_dists = self._bite_distances[index]
            return SimpleGrasp(
                rotation_matrix=self._rotations[index].copy(),
                translation=self._translations[index].copy(),
                score=float(self._scores[index]),
                bite_distances=bite_dists,
            )
        if isinstance(index, slice):
            bite_dists = self._bite_distances[index] if self._bite_distances is not None else None
            return GraspGroup(
                rotations=self._rotations[index],
                translations=self._translations[index],
                scores=self._scores[index],
                bite_distances=bite_dists
            )
        if isinstance(index, (list, tuple, np.ndarray)):
            bite_dists = self._bite_distances[index] if self._bite_distances is not None else None
            return GraspGroup(
                rotations=self._rotations[index],
                translations=self._translations[index],
                scores=self._scores[index],
                bite_distances=bite_dists
            )
        raise TypeError(f"Unsupported index type {type(index)!r} for GraspGroup")

    def from_npy(self, npy_file_path: str) -> "GraspGroup":
        """
        Load grasps from .npy or .npz file and convert to world frame.
        
        Supports both formats:
        - .npy: Single array file (legacy format)
        - .npz: Compressed file with 'grasps' key (new format from save_grasps_npz)
               Can also contain 'scores' and 'bite_distances' arrays
        """
        data = np.load(npy_file_path, allow_pickle=True)
        
        scores = None
        bite_distances = None
        
        if isinstance(data, np.lib.npyio.NpzFile):
            if 'grasps' in data:
                grasps_data = data['grasps']
                if 'scores' in data:
                    scores = data['scores']
                if 'bite_distances' in data:
                    bite_distances = data['bite_distances']
            elif 'grasp_group_array' in data:
                grasps_data = data['grasp_group_array']
            else:
                keys = list(data.keys())
                if len(keys) > 0:
                    grasps_data = data[keys[0]]
                else:
                    raise ValueError(f"No valid data found in npz file: {npy_file_path}")
        else:
            grasps_data = data
        
        if grasps_data.ndim == 1:
            grasps_data = grasps_data.reshape(1, -1)
        
        if grasps_data.ndim == 3 and grasps_data.shape[1] == 4 and grasps_data.shape[2] == 4:
            rotations_graspgen = grasps_data[:, :3, :3]
            translations_graspgen = grasps_data[:, :3, 3]
        elif grasps_data.ndim == 2 and grasps_data.shape[1] == 17:
            rotations_graspgen = grasps_data[:, 4:13].reshape(-1, 3, 3)
            translations_graspgen = grasps_data[:, 13:16]
            if scores is None:
                scores = grasps_data[:, 0]
        else:
            raise ValueError(f"Unsupported grasp data shape: {grasps_data.shape}")
       
        if bite_distances is not None:
            bite_distances = np.asarray(bite_distances, dtype=np.float32)
          
        self._rotations = rotations_graspgen
        self._translations = translations_graspgen
        self._scores = scores.astype(np.float32) if scores is not None else np.zeros(len(rotations_world), dtype=np.float32)
        self._bite_distances = bite_distances
        
        return self
    

    def transform(self, new_transform: np.ndarray):
        """
        Transform all grasps in place.
        
        Args:
            new_transform: (4, 4) transformation matrix
        """
        if len(self._rotations) == 0:
            return self
        
        new_transform = np.asarray(new_transform, dtype=np.float32)
        if new_transform.shape != (4, 4):
            raise ValueError(f"Expected (4, 4) transformation matrix, got {new_transform.shape}")
        
        for i in range(len(self._rotations)):
            total_matrix = np.eye(4, dtype=np.float32)
            total_matrix[:3, :3] = self._rotations[i]
            total_matrix[:3, 3] = self._translations[i]
            total_matrix = new_transform @ total_matrix
            self._rotations[i] = total_matrix[:3, :3]
            self._translations[i] = total_matrix[:3, 3]
        
        return self
    
    def to_world_transform(self) -> np.ndarray:
        """
        Convert the grasp group to a world transformation matrix.
        """
        
        delta = np.array([
            [0, 1, 0],  # X_world = -X_grasp
            [-1, 0, 0],  # Y_world = -Y_grasp
            [0, 0, 1],   # Z_world = Z_grasp    
        ], dtype=np.float32)
        for i in range(len(self._rotations)):
            self._rotations[i] = self._rotations[i] @ delta
            self._translations[i] = self._translations[i]
        return self

    def retrieve_grasp_group(self, index: int) -> "GraspGroup":
        """
        Retrieve a grasp group from the original grasp group.
        """
        p = self._translations[index]
        q = transforms3d.quaternions.mat2quat(self._rotations[index])
        return p, q

    def rescore(
        self,
        direction_hint: Optional[np.ndarray] = None,
        use_point_hint: bool = False,
        reorder_num: int = 50
    ) -> "GraspGroup":
        """
        Rescore grasps based on bite_distances and direction hint.
        
        Args:
            direction_hint: Optional (3,) direction vector for approach alignment
            use_point_hint: If True, use bite_distances (distance to grasp_point) for scoring
            reorder_num: Number of top grasps to reorder (default: 50)
        
        Returns:
            self (modified in place)
        """
        if len(self._rotations) == 0:
            return self
        
        N = min(reorder_num, len(self._rotations))
        
        distance_scores = None
        if use_point_hint and self._bite_distances is not None:
            avg_distances = np.mean(self._bite_distances[:N], axis=1)
            max_dist = avg_distances.max()
            min_dist = avg_distances.min()
            
            if max_dist > min_dist:
                distance_scores = 1.0 - (avg_distances - min_dist) / (max_dist - min_dist)
            else:
                distance_scores = np.ones_like(avg_distances)
        
        dir_term = None
        if direction_hint is not None:
            direction_hint = np.asarray(direction_hint, dtype=np.float64)
            direction_hint = direction_hint / (np.linalg.norm(direction_hint) + 1e-12)
            
            approach_axes = self._rotations[:N, :, 2]
            approach_axes = approach_axes / (np.linalg.norm(approach_axes, axis=1, keepdims=True) + 1e-12)
            
            cosv = np.clip(np.sum(approach_axes * direction_hint[None, :], axis=1), -1.0, 1.0)
            dir_term = 0.5 * (cosv + 1.0)
        
        original_scores = self._scores[:N].astype(np.float64)
        if original_scores.max() > original_scores.min():
            net_term = (original_scores - original_scores.min()) / (original_scores.max() - original_scores.min())
        else:
            net_term = np.ones_like(original_scores)
        
        w_dist = 0.5 if (use_point_hint and distance_scores is not None) else 0.0
        w_dir = 0.2 if (direction_hint is not None and dir_term is not None) else 0.0
        w_net = 1.0 - w_dist - w_dir
        
        new_scores = np.zeros(N, dtype=np.float64)
        if w_dist > 0:
            new_scores += w_dist * distance_scores
        if w_dir > 0:
            new_scores += w_dir * dir_term
        if w_net > 0:
            new_scores += w_net * net_term
        
        self._scores[:N] = new_scores.astype(np.float32)
        
        sorted_indices = np.argsort(new_scores)[::-1]
        self._rotations[:N] = self._rotations[:N][sorted_indices]
        self._translations[:N] = self._translations[:N][sorted_indices]
        self._scores[:N] = self._scores[:N][sorted_indices]
        if self._bite_distances is not None:
            self._bite_distances[:N] = self._bite_distances[:N][sorted_indices]
        
        return self


def grasp_to_world(grasp: Union[int, SimpleGrasp, GraspGroup]) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a grasp to world pose (position, quaternion).
    
    Supports multiple input types:
    - int: index into a GraspGroup (requires grasp to be a GraspGroup instance)
    - SimpleGrasp: a single grasp object
    - GraspGroup: returns the first grasp
    
    Args:
        grasp: Grasp index, SimpleGrasp, or GraspGroup
    
    Returns:
        tuple: (position, quaternion) in world frame
            position: (3,) np.ndarray
            quaternion: (4,) np.ndarray (wxyz format)
    """
    if isinstance(grasp, GraspGroup):
        return grasp.grasp_to_world(0)
    elif isinstance(grasp, SimpleGrasp):
        return grasp.to_world_pose()
    else:
        raise TypeError(f"Expected SimpleGrasp or GraspGroup, got {type(grasp)}")


__all__ = ["GraspGroup", "SimpleGrasp", "grasp_to_world"]
