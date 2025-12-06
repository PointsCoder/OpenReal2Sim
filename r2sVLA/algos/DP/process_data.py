import pickle, os
import numpy as np
import pdb
from copy import deepcopy
import zarr
import shutil
import argparse
import yaml
import cv2
import h5py


def load_hdf5(dataset_path, use_franka_format=False):
    """
    Load HDF5 data.
    
    Args:
        dataset_path: Path to HDF5 file
        use_franka_format: If True, load from Franka format (joint_pos + gripper_cmd)
                          If False, load from ViperX format (left/right arm + gripper)
    """
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        if use_franka_format or '/joint_action/joint_pos' in root:
            # Franka format: single arm robot
            joint_pos = root["/joint_action/joint_pos"][()]  # (T, 7)
            gripper_cmd = root["/joint_action/gripper_cmd"][()]  # (T, 2)
            
            # Normalize gripper: 0.0 = close, 1.0 = open
            FRANKA_GRIPPER_OPEN = 0.04
            FRANKA_GRIPPER_CLOSE = 0.00
            gripper_normalized = (gripper_cmd[:, 0] - FRANKA_GRIPPER_CLOSE) / (FRANKA_GRIPPER_OPEN - FRANKA_GRIPPER_CLOSE)
            gripper_normalized = gripper_normalized.reshape(-1, 1).astype(np.float32)
            
            # Combine to 8D: [7 joints, 1 normalized gripper]
            vector = np.concatenate([joint_pos, gripper_normalized], axis=1).astype(np.float32)  # (T, 8)
            
            # For compatibility, return empty arrays for left/right
            left_gripper = np.array([])
            left_arm = np.array([])
            right_gripper = np.array([])
            right_arm = np.array([])
        else:
            # ViperX format: dual arm robot
            left_gripper, left_arm = (
                root["/joint_action/left_gripper"][()],
                root["/joint_action/left_arm"][()],
            )
            right_gripper, right_arm = (
                root["/joint_action/right_gripper"][()],
                root["/joint_action/right_arm"][()],
            )
            vector = root["/joint_action/vector"][()]
        
        image_dict = dict()
        camera_config = dict()  # Store camera intrinsics/extrinsics and computed fovy
        for cam_name in root[f"/observation/"].keys():
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]
            # Store camera config if available
            cam_group = root[f"/observation/{cam_name}"]
            if 'intrinsics' in cam_group:
                intrinsics = cam_group['intrinsics'][:]
                camera_config[f"{cam_name}_intrinsics"] = intrinsics
                # Calculate fovy from intrinsics (will need image resolution)
                # fovy = 2 * arctan(h / (2 * fy)) * 180 / π
                # We'll calculate this after we decode the first image
                camera_config[f"{cam_name}_fy"] = float(intrinsics[1, 1])  # Store fy for later calculation
            if 'extrinsics' in cam_group:
                camera_config[f"{cam_name}_extrinsics"] = cam_group['extrinsics'][:]

    return left_gripper, left_arm, right_gripper, right_arm, vector, image_dict, camera_config


def main():
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument(
        "task_name",
        type=str,
        help="The name of the task (e.g., beat_block_hammer)",
    )
    parser.add_argument("task_config", type=str)
    parser.add_argument(
        "expert_data_num",
        type=int,
        help="Number of episodes to process (e.g., 50)",
    )
    parser.add_argument(
        "--use_franka_format",
        action="store_true",
        help="Use Franka format (single arm, 8D) instead of ViperX format (dual arm, 14D)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Custom data directory path (default: ../../data/{task_name}/{task_config})",
    )
    args = parser.parse_args()

    task_name = args.task_name
    num = args.expert_data_num
    task_config = args.task_config
    use_franka_format = args.use_franka_format

    if args.data_dir:
        load_dir = args.data_dir
    else:
        load_dir = "../../data/" + str(task_name) + "/" + str(task_config)

    total_count = 0

    save_dir = f"./data/{task_name}-{task_config}-{num}.zarr"

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    current_ep = 0

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")
    zarr_camera = zarr_root.create_group("camera")  # Store camera config

    head_camera_arrays, front_camera_arrays, left_camera_arrays, right_camera_arrays = (
        [],
        [],
        [],
        [],
    )
    episode_ends_arrays, action_arrays, state_arrays, joint_action_arrays = (
        [],
        [],
        [],
        [],
    )

    while current_ep < num:
        print(f"processing episode: {current_ep + 1} / {num}", end="\r")

        # Support both episode naming formats
        if use_franka_format:
            # Franka format: episode_000.hdf5, episode_001.hdf5, etc.
            load_path = os.path.join(load_dir, f"episode_{current_ep:03d}.hdf5")
            if not os.path.exists(load_path):
                load_path = os.path.join(load_dir, f"episode_{current_ep}.hdf5")
        else:
            # ViperX format: episode0.hdf5, episode1.hdf5, etc.
            load_path = os.path.join(load_dir, f"data/episode{current_ep}.hdf5")
        
        (
            left_gripper_all,
            left_arm_all,
            right_gripper_all,
            right_arm_all,
            vector_all,
            image_dict_all,
            camera_config_all,
        ) = load_hdf5(load_path, use_franka_format=use_franka_format)
        
        # Store camera config from first episode
        if current_ep == 0 and camera_config_all:
            for key, value in camera_config_all.items():
                zarr_camera.create_dataset(key, data=value, overwrite=True)

        # Determine episode length from vector_all
        episode_len = vector_all.shape[0]

        for j in range(0, episode_len):
            head_img_bit = image_dict_all["head_camera"][j]
            joint_state = vector_all[j]

            if j != episode_len - 1:
                head_img = cv2.imdecode(np.frombuffer(head_img_bit, np.uint8), cv2.IMREAD_COLOR)
                if head_img is None:
                    print(f"\nWarning: Failed to decode image at episode {current_ep}, timestep {j}")
                    # Create black image as fallback - use detected resolution if available
                    if current_ep == 0 and j == 0:
                        # Try to get resolution from first successful decode
                        fallback_h, fallback_w = 480, 640
                    else:
                        fallback_h, fallback_w = 480, 640
                    head_img = np.zeros((fallback_h, fallback_w, 3), dtype=np.uint8)
                # Store resolution and calculate fovy from first image
                if current_ep == 0 and j == 0 and head_img is not None:
                    img_h, img_w = head_img.shape[:2]
                    zarr_meta.attrs['head_camera_h'] = int(img_h)
                    zarr_meta.attrs['head_camera_w'] = int(img_w)
                    
                    # Calculate fovy from intrinsics if available
                    if 'head_camera_fy' in camera_config_all:
                        fy = camera_config_all['head_camera_fy']
                        # fovy = 2 * arctan(h / (2 * fy)) * 180 / π
                        fovy_rad = 2 * np.arctan(img_h / (2 * fy))
                        fovy_deg = np.degrees(fovy_rad)
                        zarr_meta.attrs['head_camera_fovy'] = float(fovy_deg)
                        print(f"\n[Camera Config] Detected image resolution: {img_w}x{img_h}")
                        print(f"[Camera Config] Calculated fovy: {fovy_deg:.2f} degrees (from fy={fy:.2f})")
                    else:
                        print(f"\n[Camera Config] Detected image resolution: {img_w}x{img_h}")
                        print(f"[Camera Config] Warning: No intrinsics found, cannot calculate fovy")
                head_camera_arrays.append(head_img)
                state_arrays.append(joint_state)
            if j != 0:
                joint_action_arrays.append(joint_state)

        current_ep += 1
        total_count += episode_len - 1
        episode_ends_arrays.append(total_count)

    print()
    episode_ends_arrays = np.array(episode_ends_arrays)
    # action_arrays = np.array(action_arrays)
    state_arrays = np.array(state_arrays)
    head_camera_arrays = np.array(head_camera_arrays)
    joint_action_arrays = np.array(joint_action_arrays)

    head_camera_arrays = np.moveaxis(head_camera_arrays, -1, 1)  # NHWC -> NCHW

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    # action_chunk_size = (100, action_arrays.shape[1])
    state_chunk_size = (100, state_arrays.shape[1])
    joint_chunk_size = (100, joint_action_arrays.shape[1])
    head_camera_chunk_size = (100, *head_camera_arrays.shape[1:])
    zarr_data.create_dataset(
        "head_camera",
        data=head_camera_arrays,
        chunks=head_camera_chunk_size,
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "state",
        data=state_arrays,
        chunks=state_chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "action",
        data=joint_action_arrays,
        chunks=joint_chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_meta.create_dataset(
        "episode_ends",
        data=episode_ends_arrays,
        dtype="int64",
        overwrite=True,
        compressor=compressor,
    )


if __name__ == "__main__":
    main()
