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
    get_best_grasp_pose,
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
    args = parser.parse_args()

    env = OpenReal2SimEnv(
        scene_json_path=args.scene,
        control_mode="pd_joint_pos",
        render_mode="human",
    )
    env.reset()
    target_object_id = next(iter(env.unwrapped.scene_config.objects))
    grasp_path = env.unwrapped.scene_config.objects[target_object_id].grasps
    grasps = load_grasps_from_path(grasp_path)  # (N, 17)
    best_grasp_local = get_best_grasp_pose(grasps)  # (1, 17)
    grasp_visual = build_two_finger_gripper_grasp_pose_visual(env.unwrapped.scene)

    def visualize_grasp_pose(target_object_id: int, grasp_visual: Actor):
        # --- Grasp Loading and Transformation ---
        target_object = env.unwrapped.object_actors[target_object_id]

        if best_grasp_local is None:
            print(f"No valid grasps found for object {target_object_id}")
            env.close()
            return

        # --- Visualization ---
        # transform the grasp pose to the world frame
        grasp_pose_world = transform_grasp_pose(target_object.pose, best_grasp_local)
        grasp_visual.set_pose(grasp_pose_world)

    visualize_grasp_pose(target_object_id, grasp_visual)
    viewer = env.render_human()

    try:
        while True:
            env.render()
            visualize_grasp_pose(target_object_id, grasp_visual)
            if viewer.window.key_down("c"):
                action = 0.1 * env.action_space.sample()
                env.step(action)
                time.sleep(2)
    except KeyboardInterrupt:
        print("\nExiting.")

    env.close()


if __name__ == "__main__":
    main()
