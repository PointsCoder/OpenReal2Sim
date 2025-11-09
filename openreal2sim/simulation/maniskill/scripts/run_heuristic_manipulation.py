import gymnasium as gym
import numpy as np
import sapien
import argparse
import sys
import os
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.actor import Actor
from transforms3d.quaternions import mat2quat

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs import OpenReal2SimEnv
from motion.grasp import (
    load_grasps_from_path,
    get_top_n_grasp_poses,
    transform_grasp_pose,
)
from utils.scene_loader import load_trajectory
from motion.planner import PandaArmMotionPlanningSolver, HeuristicManipulationAgent
from mani_skill.envs.sapien_env import BaseEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        type=str,
        default="/home/haoyang/project/haoyang/OpenReal2Sim/outputs/demo_video/scene/scene.json",
    )
    parser.add_argument("--num-grasps", type=int, default=10)
    args = parser.parse_args()

    env = OpenReal2SimEnv(
        scene_json_path=args.scene,
        control_mode="pd_joint_pos",
        render_mode="human",
    )
    env.reset()

    # --- Setup Agent and Planner ---
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=True,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=True,
        print_env_info=False,
    )
    agent = HeuristicManipulationAgent(env, planner)
    target_object_id = next(iter(env.unwrapped.scene_config.objects))
    target_object: Actor = env.unwrapped.object_actors[target_object_id]

    # --- Load Grasps ---
    grasp_path = env.unwrapped.scene_config.objects[target_object_id].grasps
    grasps_data = load_grasps_from_path(grasp_path)
    top_grasps_local = get_top_n_grasp_poses(
        grasps_data, n=args.num_grasps, direction_hint=np.array([0, 0, -1])
    )

    # --- Grasp Trial Loop ---
    grasp_succeeded = False
    for i, grasp_local in enumerate(top_grasps_local):
        print(f"--- Attempting Grasp {i + 1}/{len(top_grasps_local)} ---")
        env.reset()  # Reset env for each attempt
        grasp_world = transform_grasp_pose(target_object.pose, grasp_local)
        agent.attempt_grasp(grasp_world)

        if agent.check_grasp_success(target_object_id):
            print("--- Grasp Succeeded! ---")
            grasp_succeeded = True
            break
        else:
            print("--- Grasp Failed, Trying Next ---")

    # --- Trajectory Following ---
    # TODO: Check other scenes
    if grasp_succeeded:
        print("--- Proceeding to Trajectory Following ---")
        # Load the real trajectory for the final run
        traj_path = env.unwrapped.scene_config.objects[target_object_id].trajectory_path
        trajectory = load_trajectory(traj_path)
        if trajectory:
            # Align the start of the trajectory with the object's current pose
            initial_sim_pose = target_object.pose
            initial_traj_pose = trajectory[0]

            T_world_sim_init = initial_sim_pose.to_transformation_matrix()
            T_world_traj_init = initial_traj_pose.to_transformation_matrix()
            transform_matrix = np.array(T_world_sim_init) @ np.linalg.inv(
                T_world_traj_init
            )

            adjusted_trajectory = []
            for traj_pose in trajectory:
                T_world_traj_t = np.array(traj_pose.to_transformation_matrix())
                T_world_adjusted_t = transform_matrix @ T_world_traj_t
                # unsqueeze to 4x4 matrix
                T_world_adjusted_t = T_world_adjusted_t.reshape(4, 4)

                p = T_world_adjusted_t[:3, 3]
                q = mat2quat(T_world_adjusted_t[:3, :3])

                adjusted_pose = Pose.create_from_pq(p=p, q=q)
                adjusted_trajectory.append(adjusted_pose)

            agent.follow_trajectory(target_object_id, adjusted_trajectory)
        else:
            print("Could not load trajectory, stopping.")
    else:
        print("--- All Grasp Attempts Failed ---")

    env.close()


if __name__ == "__main__":
    main()
