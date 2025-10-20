# ManiSkill Integration for OpenReal2Sim

This module provides a ManiSkill environment that loads reconstructed scenes from the OpenReal2Sim pipeline, enabling robotic manipulation simulation on real-world reconstructed scenes.

## Overview

The `OpenReal2SimEnv` is a custom ManiSkill environment that:

- Loads reconstructed 3D scenes (background and objects) from `scene.json`
- Configures cameras to match the original scene geometry
- Provides a Franka Panda robot for manipulation
- Supports various observation modes (state, RGB-D, etc.)
- Handles coordinate transformations between OpenCV and SAPIEN/ROS conventions

## Directory Structure

```
maniskill/
├── __init__.py                  # Package initialization
├── env.py                       # Main environment class
├── test_env.py                  # Test script
├── README.md                    # This file
└── utils/
    ├── __init__.py
    ├── scene_loader.py          # Scene configuration loader
    └── transform_utils.py       # Coordinate transformation utilities
```

## Installation

### Prerequisites

1. **ManiSkill 3**: Install ManiSkill following the [official documentation](https://maniskill.readthedocs.io/)

```bash
pip install mani-skill
```

2. **Additional Dependencies**:

```bash
pip install scipy transforms3d
```

### Setup

The ManiSkill integration is already part of the OpenReal2Sim repository. Simply ensure you're in the project root:

```bash
cd /path/to/OpenReal2Sim
```

## Usage

### Basic Usage

```python
import gymnasium as gym
from openreal2sim.simulation.maniskill import OpenReal2SimEnv

# Create environment
env = gym.make(
    "OpenReal2Sim-v0",
    scene_json_path="outputs/demo_genvideo/scene/scene.json",
    num_envs=1,
    obs_mode="rgbd",
    render_mode="rgb_array"
)

# Reset environment
obs, info = env.reset()

# Step through environment
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### Running Tests

Test the environment with your reconstructed scene:

```bash
# Test with default scene
python openreal2sim/simulation/maniskill/test_env.py

# Test with specific scene
python openreal2sim/simulation/maniskill/test_env.py --scene outputs/demo_image/scene/scene.json

# Quick test (fewer steps)
python openreal2sim/simulation/maniskill/test_env.py --quick
```

The test script will:

- Create the environment
- Test reset and step functions
- Test rendering
- Validate observation modes
- Save a test render image

## Key Features

### 1. Scene Loading

The environment automatically loads:

- **Background mesh**: Loaded as a static (kinematic) actor
- **Object meshes**: Loaded as dynamic actors with physics
- **Camera configuration**: Matches the original scene's camera parameters

### 2. Coordinate System Transformation

The environment handles the transformation between:

- **OpenCV** (used by OpenReal2Sim): Z forward, X right, Y down
- **SAPIEN/ROS** (used by ManiSkill): X forward, Y left, Z up

This transformation is applied to:

- Camera extrinsic matrices
- Object poses (if needed)

### 3. Observation Modes

Supported observation modes:

- `state`: Proprioceptive state + ground truth object poses
- `rgbd`: RGB-D images from cameras
- `state_dict`: Dictionary of state observations
- `pointcloud`: Point cloud observations (if configured)

### 4. Robot Configuration

Default robot: **Franka Panda**

- 7-DOF arm
- Parallel gripper
- Configurable initial position

## Configuration

### Environment Parameters

```python
env = gym.make(
    "OpenReal2Sim-v0",
    scene_json_path="path/to/scene.json",  # Required: path to scene.json
    robot_uids="panda",                     # Robot model
    num_envs=1,                             # Number of parallel environments
    obs_mode="rgbd",                        # Observation mode
    render_mode="rgb_array",                # Render mode
    robot_init_qpos_noise=0.02,            # Initial joint position noise
)
```

### Scene JSON Format

The environment expects a `scene.json` file with the following structure:

```json
{
  "background": {
    "registered": "path/to/background.glb"
  },
  "objects": {
    "1": {
      "name": "object_name",
      "registered": "path/to/object.glb",
      "object_center": [x, y, z],
      "grasps": "path/to/grasps.npy"
    }
  },
  "camera": {
    "width": 848,
    "height": 480,
    "fx": 611.23,
    "fy": 610.76,
    "cx": 429.54,
    "cy": 241.20,
    "camera_opencv_to_world": [[...], [...], [...], [...]]
  },
  "groundplane_in_sim": {
    "point": [x, y, z],
    "normal": [nx, ny, nz]
  }
}
```

## Implementation Details

### Key Classes

#### `OpenReal2SimEnv`

Main environment class inheriting from `mani_skill.envs.sapien_env.BaseEnv`.

**Key Methods:**

- `_load_scene()`: Load all assets (called once)
- `_initialize_episode()`: Reset episode state (called every reset)
- `evaluate()`: Evaluate success/failure conditions
- `_get_obs_extra()`: Get additional observations
- `_default_sensor_configs`: Configure cameras

#### `SceneConfig`

Data class holding parsed scene configuration.

#### Utility Functions

- `opencv_to_sapien_pose()`: Transform camera pose
- `intrinsic_to_fov()`: Convert intrinsics to FOV
- `load_scene_config()`: Parse scene.json

### Collision Handling

- **Background**: Uses mesh directly (no decomposition)
- **Objects**: Uses COACD convex decomposition for stable physics

### Camera Configuration

The environment creates cameras that match the original scene:

1. Parse camera intrinsics (fx, fy, cx, cy) and extrinsics
2. Transform extrinsic matrix from OpenCV to SAPIEN coordinates
3. Convert intrinsics to FOV
4. Create `CameraConfig` with matched parameters

## Troubleshooting

### Common Issues

1. **Module not found**: Ensure you're running from the project root

   ```bash
   cd /path/to/OpenReal2Sim
   python -m openreal2sim.simulation.maniskill.test_env
   ```

2. **Mesh files not found**: Check that paths in `scene.json` are correct

   - The loader automatically converts `/app/` paths to local paths
   - Verify files exist: `ls outputs/demo_genvideo/reconstruction/`

3. **Collision decomposition fails**:

   - This is expected for complex meshes
   - The code falls back to using the mesh without decomposition

4. **Camera view is wrong**:
   - Verify coordinate transformation is correct
   - Check that extrinsic matrix in `scene.json` is valid

## Next Steps

### Motion Planning Integration

After validating the environment, the next steps are:

1. **Grasp Loading**: Load pre-computed grasps from `.npy` files
2. **Motion Planning**: Integrate a motion planner (mplib, OMPL, or custom)
3. **Trajectory Execution**: Execute planned trajectories
4. **Data Collection**: Record robot demonstrations

### Example Task Implementation

You can extend `OpenReal2SimEnv` to create specific tasks:

```python
@register_env("OpenReal2Sim-PickPlace-v0", max_episode_steps=200)
class PickPlaceEnv(OpenReal2SimEnv):
    def evaluate(self):
        # Check if object is at target location
        obj_pos = self.object_actors["1"].pose.p
        target_pos = self.target_position
        dist = np.linalg.norm(obj_pos - target_pos)

        return {
            "success": dist < 0.05,
            "distance": dist,
        }

    def compute_dense_reward(self, obs, action, info):
        # Reward based on distance to target
        obj_pos = self.object_actors["1"].pose.p
        dist = np.linalg.norm(obj_pos - self.target_position)
        return -dist
```

## References

- [ManiSkill Documentation](https://maniskill.readthedocs.io/)
- [SAPIEN Documentation](https://sapien.ucsd.edu/)
- [OpenReal2Sim Repository](https://github.com/PointsCoder/OpenReal2Sim)

## License

This code is part of the OpenReal2Sim project and follows the same license.
