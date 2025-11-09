import gymnasium as gym
import numpy as np
import sapien
import argparse
import time
import sys
import os
from mani_skill.utils.structs.actor import Actor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs import OpenReal2SimEnv
from motion.grasp import (
    load_grasps_from_path,
    get_top_n_grasp_poses,
    transform_grasp_pose,
)
from motion.planner import build_two_finger_gripper_grasp_pose_visual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        type=str,
        default="/home/haoyang/project/haoyang/OpenReal2Sim/outputs/demo_video/scene/scene.json",
    )
    parser.add_argument(
        "--num-grasps",
        type=int,
        default=5,
        help="Number of top grasp poses to visualize.",
    )
    parser.add_argument(
        "--direction-hint",
        type=float,
        nargs=3,
        default=[0, 0, -1],
        metavar=("X", "Y", "Z"),
        help="e.g. 0 0 -1 for top-down grasps",
    )
    parser.add_argument(
        "--position-hint",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="e.g. 0.1 0 0.2 for a point in world space",
    )
    args = parser.parse_args()

    env = OpenReal2SimEnv(
        scene_json_path=args.scene,
        control_mode="pd_joint_pos",
        render_mode="human",
    )
    env.reset()

    # --- Load Grasps ---
    target_object_id = next(iter(env.unwrapped.scene_config.objects))
    grasp_path = env.unwrapped.scene_config.objects[target_object_id].grasps
    grasps_data = load_grasps_from_path(grasp_path)

    direction_hint = np.array(args.direction_hint) if args.direction_hint else None
    position_hint = np.array(args.position_hint) if args.position_hint else None

    # Get object center to make position hint relative to the object
    if position_hint is not None:
        target_object = env.unwrapped.object_actors[target_object_id]
        object_position = target_object.pose.p
        position_hint += object_position

    top_grasps_local = get_top_n_grasp_poses(
        grasps_data,
        n=args.num_grasps,
        direction_hint=direction_hint,
        position_hint=position_hint,
    )

    if not top_grasps_local:
        print(f"No valid grasps found for object {target_object_id}")
        env.close()
        return

    # --- Setup Visualizer ---
    grasp_visual = build_two_finger_gripper_grasp_pose_visual(env.unwrapped.scene)
    target_object = env.unwrapped.object_actors[target_object_id]
    print(f"Visualizing top {len(top_grasps_local)} grasp poses. Press Ctrl+C to exit.")

    # --- Visualization Loop ---
    try:
        while True:
            for i, grasp_local in enumerate(top_grasps_local):
                print(f"Displaying grasp pose {i + 1}/{len(top_grasps_local)}")

                # Transform grasp to world frame and update visualizer
                grasp_pose_world = transform_grasp_pose(target_object.pose, grasp_local)
                grasp_visual.set_pose(grasp_pose_world)

                # Render the scene for a fixed duration
                start_time = time.time()
                while time.time() - start_time < 2.0:  # Display each pose for 2 secs
                    env.render()

    except KeyboardInterrupt:
        print("\nExiting.")

    env.close()


if __name__ == "__main__":
    main()
