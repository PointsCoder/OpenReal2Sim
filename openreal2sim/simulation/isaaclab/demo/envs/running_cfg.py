from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class RandomizerConfig:
    """Configuration bundle for `Randomizer.generate_randomized_scene_cfg`."""
    grid_dist: float = 0.015
    grid_num: int = 5
    angle_random_range: float = math.pi / 10.0
    angle_random_num: int = 10
    traj_randomize_num: int = 100
    scene_randomize_num: int = 20
    robot_pose_randomize_range: float = 0.03
    robot_pose_randomize_angle: float = math.pi / 180.0
    robot_pose_randomize_num: int = 3
    fix_end_pose: bool = True
    fix_robot_pose: bool = True
    fix_other_objects: bool = True    
    use_no_interpolation: bool = False

    def to_kwargs(self) -> Dict[str, float | int]:
        """Return a shallow dict representation that can be splatted into the randomizer."""
        return asdict(self)


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for the simulation."""

    save_interval: int = 1
    decimation: int = 1
    physics_freq: int = 100
   

@dataclass(frozen=True)
class HeuristicConfig:
    """Configuration for the heuristic manipulation stage."""

    num_envs: int = 1
    num_trials: int = 10
    grasp_num: int = 1
    robot: str = "franka"
    goal_offset: float = 0.03



@dataclass(frozen=True)
class RandomizeRolloutConfig:
    """Rollout-related knobs (e.g., required number of successful trajectories)."""

    total_num: int = 50
    num_envs: int = 1
    goal_offset: float = 0.03


@dataclass(frozen=True)
class RunningConfig:
    """Top-level container bundling all per-scene tunables."""

    randomizer: RandomizerConfig = RandomizerConfig()
    rollout: RandomizeRolloutConfig = RandomizeRolloutConfig()
    heuristic: HeuristicConfig = HeuristicConfig()
    simulation: SimulationConfig = SimulationConfig()

DEFAULT_RUNNING_CONFIG = RunningConfig()

# Users can override the defaults per-scene key by adding entries here.
RUNNING_CONFIGS: Dict[str, RunningConfig] = {
    "default": DEFAULT_RUNNING_CONFIG,
    "demo_video": DEFAULT_RUNNING_CONFIG,
}


def get_randomizer_config(key: str) -> RandomizerConfig:
    """Return the config associated with `key`, falling back to the default."""
    return RUNNING_CONFIGS.get(key, DEFAULT_RUNNING_CONFIG).randomizer


def get_rollout_config(key: str) -> RandomizeRolloutConfig:
    """Return rollout config for a scene key."""
    return RUNNING_CONFIGS.get(key, DEFAULT_RUNNING_CONFIG).rollout


def get_heuristic_config(key: str) -> HeuristicConfig:
    """Return heuristic manipulation config for a scene key."""
    return RUNNING_CONFIGS.get(key, DEFAULT_RUNNING_CONFIG).heuristic

def get_simulation_config(key: str) -> SimulationConfig:
    """Return simulation config for a scene key."""
    return RUNNING_CONFIGS.get(key, DEFAULT_RUNNING_CONFIG).simulation