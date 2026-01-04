#!/usr/bin/env python3
"""
Convert OpenReal2Sim HDF5 data to LeRobot format.

This script converts simulation trajectories from OpenReal2Sim's HDF5 format
to LeRobot's standardized dataset format.

Usage:
    python openreal2sim_to_lerobot.py --hdf5_path /path/to/episode.hdf5 --output_repo my-dataset
"""

import os
import sys
import json
import argparse
import numpy as np
import h5py
from pathlib import Path
import torch
import torchvision

# Add paths for imports
base_dir = Path(__file__).parent.parent.parent.parent
sys.path.append(str(base_dir / 'third_party' / 'GraspGen'))
sys.path.append(str(base_dir / 'openreal2sim' / 'motion' / 'utils'))

def load_hdf5_episode(hdf5_path):
    """Load OpenReal2Sim HDF5 episode data."""
    with h5py.File(hdf5_path, 'r') as f:
        # Load image data
        rgb_frames = f['observation']['head_camera']['rgb'][:]  # (T, H, W, 3)

        # Load EEF pose in camera frame (only this for state)
        ee_pose_cam = f['ee_pose']['ee_pose_cam'][:]  # (T, 7) - position + quaternion in camera frame

        # Load action data (if available)
        actions = None
        if 'action' in f and 'actions' in f['action']:
            actions = f['action']['actions'][:]  # (T, action_dim)

        # Load metadata
        meta = dict(f['meta'].attrs)
        task_desc = meta.get('task_desc', 'Grasp and manipulate object')

        # Load camera intrinsics/extrinsics if available
        camera_intrinsics = None
        camera_extrinsics = None
        if 'head_camera' in f['observation']:
            cam_grp = f['observation']['head_camera']
            if 'intrinsics' in cam_grp:
                camera_intrinsics = cam_grp['intrinsics'][:]
            if 'extrinsics' in cam_grp:
                camera_extrinsics = cam_grp['extrinsics'][:]

    return {
        'rgb_frames': rgb_frames,
        'ee_pose_cam': ee_pose_cam,  # Only using this for state
        'actions': actions,
        'task_desc': task_desc,
        'camera_intrinsics': camera_intrinsics,
        'camera_extrinsics': camera_extrinsics,
        'meta': meta
    }

def create_lerobot_features(image_height, image_width, state_dim=7, action_dim=7):
    """Create LeRobot features definition."""
    features = {
        # Main camera observation
        "observation.images.camera": {
            "dtype": "video",
            "shape": (image_height, image_width, 3),
            "names": ["height", "width", "channel"],
        },

        # State observation (EEF pose in camera frame)
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),  # position + quaternion in camera frame
        },

        # Action (joint position targets or eef control)
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),  # delta joint positions or eef control
        },

        # Task description
        "task": {
            "dtype": "string",
        },
    }

    return features

def convert_episode_to_lerobot(episode_data, output_repo, episode_idx=0):
    """Convert single episode to LeRobot format."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    rgb_frames = episode_data['rgb_frames']  # (T, H, W, 3)
    ee_pose_cam = episode_data['ee_pose_cam']  # (T, 7) - position + quaternion in camera frame
    actions = episode_data['actions']
    task_desc = episode_data['task_desc']

    T, H, W, C = rgb_frames.shape

    # Use ee_pose_cam as state vector (7 dims: position + quaternion)
    states = ee_pose_cam.astype(np.float32)  # (T, 7)

    # If no actions provided, use delta EEF poses as actions
    if actions is None:
        # Use delta EEF poses as actions (7 dims)
        actions = np.zeros((T, 7), dtype=np.float32)
        actions[1:] = ee_pose_cam[1:] - ee_pose_cam[:-1]  # Delta EEF poses

    # Create features definition (state is now 7 dims, action is 7 dims)
    features = create_lerobot_features(H, W, state_dim=7, action_dim=7)

    # Set up LeRobot dataset
    os.environ["HF_LEROBOT_HOME"] = str(Path(output_repo).parent)
    os.environ["HF_HOME"] = str(Path(output_repo).parent.parent)

    repo_name = f"{Path(output_repo).name}_episode_{episode_idx}"

    # Create LeRobot dataset
    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="panda",  # Franka Panda robot
        fps=10,              # Default fps
        features=features,
        image_writer_threads=8,
        image_writer_processes=0,
        video_backend="pyav",
    )

    # Add frames to dataset
    for t in range(T):
        frame = {
            "observation.images.camera": rgb_frames[t],  # (H, W, 3)
            "observation.state": states[t],              # (15,)
            "action": actions[t],                        # (7,)
            "task": task_desc,
        }
        dataset.add_frame(frame)

    # Save episode
    dataset.save_episode()

    # Consolidate and compute stats
    dataset.consolidate(run_compute_stats=True)

    print(f"Converted episode {episode_idx} to LeRobot format: {repo_name}")
    print(f"Frames: {T}, Image size: {H}x{W}, State dim: {states.shape[1]}, Action dim: {actions.shape[1]}")

    return dataset

def convert_hdf5_to_lerobot(hdf5_path, output_repo, episode_idx=0):
    """Convert HDF5 file to LeRobot format."""
    print(f"Loading HDF5 file: {hdf5_path}")

    # Load episode data
    episode_data = load_hdf5_episode(hdf5_path)

    # Convert to LeRobot format
    dataset = convert_episode_to_lerobot(episode_data, output_repo, episode_idx)

    return dataset

def batch_convert_hdf5_directory(hdf5_dir, output_repo):
    """Convert all HDF5 files in a directory to LeRobot format."""
    hdf5_dir = Path(hdf5_dir)
    output_repo = Path(output_repo)

    hdf5_files = list(hdf5_dir.glob("*.hdf5"))
    print(f"Found {len(hdf5_files)} HDF5 files to convert")

    datasets = []
    for i, hdf5_file in enumerate(hdf5_files):
        try:
            dataset = convert_hdf5_to_lerobot(hdf5_file, output_repo, episode_idx=i)
            datasets.append(dataset)
        except Exception as e:
            print(f"Error converting {hdf5_file}: {e}")
            continue

    print(f"Successfully converted {len(datasets)} episodes")
    return datasets

def main():
    parser = argparse.ArgumentParser(description="Convert OpenReal2Sim HDF5 data to LeRobot format")
    parser.add_argument("--hdf5_path", type=str, help="Path to single HDF5 file")
    parser.add_argument("--hdf5_dir", type=str, help="Path to directory containing HDF5 files")
    parser.add_argument("--output_repo", type=str, required=True, help="Output repository name/path")
    parser.add_argument("--episode_idx", type=int, default=0, help="Episode index for single file conversion")

    args = parser.parse_args()

    # Setup video backend
    torchvision.set_video_backend("pyav")

    if args.hdf5_path:
        convert_hdf5_to_lerobot(args.hdf5_path, args.output_repo, args.episode_idx)
    elif args.hdf5_dir:
        batch_convert_hdf5_directory(args.hdf5_dir, args.output_repo)
    else:
        print("Error: Must specify either --hdf5_path or --hdf5_dir")
        sys.exit(1)

if __name__ == "__main__":
    main()
