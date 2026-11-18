import gymnasium as gym
import numpy as np
import sapien
import argparse
from transforms3d.euler import euler2quat

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from motion.planner import PandaArmMotionPlanningSolver
from envs import OpenReal2SimEnv
from mani_skill.utils.structs.actor import Actor
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)


def solve_openr2s(env: OpenReal2SimEnv, debug=True, vis=False):
    obs, _ = env.reset()
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )
    # planner.update_collision_objects()

    FINGER_LENGTH = 0.025
    env = env.unwrapped
    target_object: Actor = env.object_actors["1"]
    obb = get_actor_obb(target_object)

    approaching = np.array([0, 0, -1])
    target_closing = (
        env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    )
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)

    # Search a valid pose
    angles = np.arange(0, np.pi * 2 / 3, np.pi / 2)
    angles = np.repeat(angles, 2)
    angles[1::2] *= -1
    for angle in angles:
        delta_pose = sapien.Pose(q=euler2quat(0, 0, angle))
        grasp_pose2 = grasp_pose * delta_pose
        res = planner.move_to_pose_with_screw(grasp_pose2, dry_run=True)
        if res == -1:
            continue
        grasp_pose = grasp_pose2
        break
    else:
        print("Fail to find a valid grasp pose")

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.1])
    planner.move_to_pose_with_screw(reach_pose)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Lift
    # -------------------------------------------------------------------------- #
    lift_pose = sapien.Pose([0, 0, 0.2]) * grasp_pose
    planner.move_to_pose_with_screw(lift_pose)

    res = planner.open_gripper()
    planner.close()

    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        type=str,
        default="/home/haoyang/project/haoyang/OpenReal2Sim/outputs/demo_video/scene/scene.json",
    )
    parser.add_argument("--vis", action="store_true")
    args = parser.parse_args()

    env = OpenReal2SimEnv(
        scene_json_path=args.scene,
        control_mode="pd_joint_pos",
        render_mode="human" if args.vis else "none",
    )

    result = solve_openr2s(env, debug=True, vis=args.vis)

    print("Motion planning successful!")

    env.close()


if __name__ == "__main__":
    main()
