"""
Evaluation configuration for policy evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List
import yaml


@dataclass
class EvaluationConfig:
    """Configuration for policy evaluation."""
    max_steps: int = 1000  # Maximum number of steps per episode
    success_threshold: Optional[float] = None  # Optional success threshold
    record_video: bool = False  # Whether to record video
    video_save_dir: Optional[Path] = None  # Directory to save videos
    use_verified_randomization: bool = False  # Whether to use verified randomization
    success_keys: List[str] = field(default_factory=lambda: ["grasping", "strict", "metric"])
    pose_dist_threshold: float = 0.05  # Threshold for pose distance
    angle_dist_threshold: float = 0.1  # Threshold for angle distance
    lift_height_threshold: float = 0.02  # Threshold for lift height
    num_trials: int = 10  # Number of environments to evaluate
    physics_freq: int = 100  # Frequency to update the physics
    decimation: int = 1
    save_interval: int = 1

   

    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EvaluationConfig':
        """Create EvaluationConfig from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_yaml(cls, yaml_path: Path, task_name: Optional[str] = None) -> 'EvaluationConfig':
        """Load EvaluationConfig from YAML file."""
        if not yaml_path.exists():
            print(f"Warning: Evaluation config file not found at {yaml_path}, using defaults")
            return cls()
        
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        if task_name and task_name in data:
            return cls.from_dict(data[task_name])
        elif 'default' in data:
            return cls.from_dict(data['default'])
        else:
            return cls.from_dict(data)

