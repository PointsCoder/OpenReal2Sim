"""
Randomizer configuration for scene and trajectory randomization.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import yaml


@dataclass
class RandomizerConfig:
    """
    Configuration for scene and trajectory randomization.
    
    This config controls how randomized scenes and trajectories are generated
    for data collection and training.
    """
    # Grid-based position randomization
    grid_dist: float = 0.05  # Distance between grid points (meters)
    grid_num: int = 2  # Number of grid points in each direction (-grid_num to +grid_num)
    
    # Rotation randomization
    angle_random_range: float = 0.5  # Random angle range in radians
    angle_random_num: int = 3  # Number of random angles per grid point
    
    # Trajectory randomization
    traj_randomize_num: int = 5  # Number of different trajectory start/end pose combinations
    
    # Scene randomization (other objects)
    scene_randomize_num: int = 3  # Number of different scene configurations for other objects
    
    # Robot pose randomization
    robot_pose_randomize_range: float = 0.1  # Random translation range for robot base (meters)
    robot_pose_randomize_angle: float = 0.2  # Random rotation angle for robot base (radians)
    robot_pose_randomize_num: int = 2  # Number of different robot poses
    
    # Trajectory options
    fix_end_pose: bool = False  # Whether to fix the end pose (useful for simple pick tasks)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RandomizerConfig':
        """Create RandomizerConfig from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_yaml(cls, yaml_path: Path, task_name: Optional[str] = None) -> 'RandomizerConfig':
        """Load RandomizerConfig from YAML file."""
        if not yaml_path.exists():
            print(f"Warning: Randomizer config file not found at {yaml_path}, using defaults")
            return cls()
        
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        if task_name and task_name in data:
            return cls.from_dict(data[task_name])
        elif 'default' in data:
            return cls.from_dict(data['default'])
        else:
            return cls.from_dict(data)
    
    def get_total_trajectories(self) -> int:
        """
        Calculate total number of generated trajectories.
        
        Returns:
            Total number of trajectories = traj_randomize_num * scene_randomize_num * robot_pose_randomize_num
        """
        return self.traj_randomize_num * self.scene_randomize_num * self.robot_pose_randomize_num
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'grid_dist': self.grid_dist,
            'grid_num': self.grid_num,
            'angle_random_range': self.angle_random_range,
            'angle_random_num': self.angle_random_num,
            'traj_randomize_num': self.traj_randomize_num,
            'scene_randomize_num': self.scene_randomize_num,
            'robot_pose_randomize_range': self.robot_pose_randomize_range,
            'robot_pose_randomize_angle': self.robot_pose_randomize_angle,
            'robot_pose_randomize_num': self.robot_pose_randomize_num,
            'fix_end_pose': self.fix_end_pose,
        }

