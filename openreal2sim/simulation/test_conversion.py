#!/usr/bin/env python3
"""
Test script for OpenReal2Sim to LeRobot conversion.
Creates a minimal HDF5 file and tests the conversion.
"""

import os
import numpy as np
import h5py
from pathlib import Path
import sys
import tempfile

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def create_test_hdf5(output_path, num_frames=50, image_size=(256, 256)):
    """Create a minimal test HDF5 file in OpenReal2Sim format."""
    H, W = image_size

    with h5py.File(output_path, 'w') as f:
        # Create groups
        obs_grp = f.create_group("observation")
        cam_grp = obs_grp.create_group("head_camera")

        # Generate synthetic RGB frames (T, H, W, 3)
        rgb_frames = np.random.randint(0, 255, (num_frames, H, W, 3), dtype=np.uint8)
        rgb_dataset = cam_grp.create_dataset("rgb", data=rgb_frames)

        # Add camera intrinsics (example values)
        intrinsics = np.array([[320.0, 0.0, H/2],
                              [0.0, 320.0, W/2],
                              [0.0, 0.0, 1.0]], dtype=np.float32)
        cam_grp.create_dataset("intrinsics", data=intrinsics)

        # Add camera extrinsics (identity for simplicity)
        extrinsics = np.eye(4, dtype=np.float32)
        cam_grp.create_dataset("extrinsics", data=extrinsics)

        # Camera resolution
        cam_grp.attrs["resolution"] = (H, W)

        # EE pose in camera frame (position + quaternion) - this is our state
        ee_pos_cam = np.random.randn(num_frames, 3).astype(np.float32) * 0.1
        ee_quat_cam = np.random.randn(num_frames, 4).astype(np.float32)
        # Normalize quaternion
        ee_quat_cam = ee_quat_cam / np.linalg.norm(ee_quat_cam, axis=1, keepdims=True)
        ee_pose_cam = np.concatenate([ee_pos_cam, ee_quat_cam], axis=1)

        ee_grp = f.create_group("ee_pose")
        ee_grp.create_dataset("ee_pose_cam", data=ee_pose_cam)

        # Actions (optional - delta joint positions)
        actions = np.random.randn(num_frames, 7).astype(np.float32) * 0.01
        action_grp = f.create_group("action")
        action_grp.create_dataset("actions", data=actions)

        # Metadata
        meta_grp = f.create_group("meta")
        meta_grp.attrs["task_desc"] = "Test grasping task with synthetic data"
        meta_grp.attrs["env_index"] = 0
        meta_grp.attrs["frame_count"] = num_frames
        meta_grp.attrs["source"] = "OpenReal2Sim"
        meta_grp.attrs["episode_name"] = "test_episode"

        print(f"Created test HDF5 file: {output_path}")
        print(f"Frames: {num_frames}, Image size: {H}x{W}")
        print(f"EE pose cam shape: {ee_pose_cam.shape}")
        print(f"Actions shape: {actions.shape}")

def test_conversion(hdf5_path, output_repo):
    """Test the conversion script."""
    try:
        from openreal2sim_to_lerobot import convert_hdf5_to_lerobot

        print(f"Testing conversion of: {hdf5_path}")
        dataset = convert_hdf5_to_lerobot(hdf5_path, output_repo, episode_idx=0)

        # Verify the converted dataset
        print(f"Converted dataset info:")
        print(f"  Episodes: {dataset.num_episodes}")
        print(f"  Total frames: {dataset.num_frames}")
        print(f"  FPS: {dataset.meta.fps}")
        print(f"  Robot type: {dataset.meta.robot_type}")

        # Load a sample
        sample = dataset[0]
        print(f"Sample keys: {list(sample.keys())}")
        print(f"Image shape: {sample['observation.images.camera'].shape}")
        print(f"State shape: {sample['observation.state'].shape}")
        print(f"Action shape: {sample['action'].shape}")

        print("✅ Conversion test passed!")
        return True

    except Exception as e:
        print(f"❌ Conversion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test HDF5 file
        hdf5_path = temp_path / "test_episode.hdf5"
        create_test_hdf5(hdf5_path, num_frames=20, image_size=(128, 128))

        # Test conversion
        output_repo = temp_path / "test_lerobot_dataset"
        success = test_conversion(hdf5_path, output_repo)

        if success:
            print("\n🎉 All tests passed! The conversion script is working correctly.")
        else:
            print("\n💥 Tests failed. Please check the conversion script.")
            sys.exit(1)

if __name__ == "__main__":
    main()
