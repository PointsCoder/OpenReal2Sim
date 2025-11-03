import gymnasium as gym
import numpy as np
import sapien
import argparse
import time
import sys
import os
from mani_skill.utils.structs.pose import Pose
from transforms3d.quaternions import mat2quat
from mani_skill.utils.structs.actor import Actor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs import OpenReal2SimEnv


def load_trajectory(filepath: str) -> list[Pose] | None:
    """Loads an object trajectory from a .npy file."""
    if not filepath or not filepath.endswith(".npy"):
        print(f"Error: Invalid trajectory file path provided: {filepath}")
        return None
    try:
        trajectory_data = np.load(filepath, allow_pickle=True)
    except FileNotFoundError:
        print(f"Error: Trajectory file not found at {filepath}")
        return None

    poses = []
    # The trajectory can be stored in different formats, so we handle both
    # a list of 4x4 matrices and a dictionary containing the poses.
    if isinstance(trajectory_data, dict):
        trajectory_data = trajectory_data["world_pose"]

    for matrix in trajectory_data:
        # position
        p = matrix[:3, 3]
        # rotation matrix to wxyz quaternion
        q_wxyz = mat2quat(matrix[:3, :3])
        poses.append(Pose.create_from_pq(p=p, q=q_wxyz))
    return poses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        type=str,
        default="/home/haoyang/project/haoyang/OpenReal2Sim/outputs/demo_video/scene/scene.json",
    )
    args = parser.parse_args()

    env = OpenReal2SimEnv(
        scene_json_path=args.scene,
        control_mode="pd_joint_pos",
        render_mode="human",
    )
    env.reset()

    # --- Load Trajectory ---
    target_object_id = next(iter(env.unwrapped.scene_config.objects))
    traj_path = env.unwrapped.scene_config.objects[target_object_id].trajectory_path
    trajectory_poses = load_trajectory(traj_path)

    if not trajectory_poses:
        print(f"No valid trajectory found for object {target_object_id}")
        env.close()
        return

    # --- Setup ---
    target_object: Actor = env.unwrapped.object_actors[target_object_id]

    # --- Calculate Transformation to Align Trajectory ---
    initial_sim_pose = target_object.pose
    initial_traj_pose = trajectory_poses[0]

    T_world_sim_init = initial_sim_pose.to_transformation_matrix()
    T_world_traj_init = initial_traj_pose.to_transformation_matrix()

    # This transform maps points from the trajectory's frame to the sim's frame
    transform_matrix = np.array(T_world_sim_init) @ np.linalg.inv(T_world_traj_init)

    # --- Adjust Trajectory ---
    adjusted_trajectory = []
    for traj_pose in trajectory_poses:
        # turn into numpy array
        T_world_traj_t = np.array(traj_pose.to_transformation_matrix())
        T_world_adjusted_t = transform_matrix @ T_world_traj_t
        # unsqueeze to 4x4 matrix
        T_world_adjusted_t = T_world_adjusted_t.reshape(4, 4)

        p = T_world_adjusted_t[:3, 3]
        q = mat2quat(T_world_adjusted_t[:3, :3])

        adjusted_pose = Pose.create_from_pq(p=p, q=q)
        adjusted_trajectory.append(adjusted_pose)

    print(
        f"Visualizing adjusted trajectory with {len(adjusted_trajectory)} points. Press Ctrl+C to exit."
    )

    # --- Visualization Loop ---
    viewer = env.render_human()
    viewer.paused = True
    try:
        while True:
            for i, pose in enumerate(adjusted_trajectory):
                print(f"Displaying pose {i + 1}/{len(adjusted_trajectory)}")
                target_object.set_pose(pose)

                # Render the scene for a short duration to create motion
                start_time = time.time()
                while time.time() - start_time < 0.05:  # ~20 FPS
                    env.render()

    except KeyboardInterrupt:
        print("\nExiting.")

    env.close()


if __name__ == "__main__":
    main()
