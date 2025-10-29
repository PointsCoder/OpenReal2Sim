#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple visualization script for OpenReal2Sim ManiSkill environment.

This script creates the environment and allows you to interact with it
using the GUI viewer or run random actions.

Usage:
    python visualize_env.py                    # Use default scene
    python visualize_env.py --scene PATH       # Use custom scene
    python visualize_env.py --mode static      # Static inspection mode
"""

import argparse
from pathlib import Path
import gymnasium as gym

import sys

# Make the project root directory available to the python path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))
print(project_root)
from simulation.maniskill.envs import OpenReal2SimEnv  # noqa


def visualize_random_actions(scene_path: str, num_steps: int = 1000):
    """
    Visualize the environment with random actions.

    Args:
        scene_path: Path to scene.json
        num_steps: Number of steps to run
    """
    print("=" * 80)
    print("OpenReal2Sim ManiSkill Visualization")
    print("=" * 80)
    print(f"\nScene: {scene_path}")
    print("Mode: Random Actions")
    print(f"Steps: {num_steps}")
    print("\nPress 'q' in the viewer window to quit.")
    print("=" * 80)

    env = gym.make(
        "OpenReal2Sim-v0",
        scene_json_path=scene_path,
        num_envs=1,
        obs_mode="state",  # Use state for faster rendering
        control_mode="pd_ee_delta_pose",  # End-effector control
        render_mode="human",
    )

    print("\nEnvironment created successfully!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print()

    obs, info = env.reset(seed=0, options=dict(reconfigure=True))

    step_count = 0
    episode_count = 0

    print("Starting visualization... (Close viewer window or press Ctrl+C to stop)")
    print()
    viewer = env.render()
    viewer.paused = True
    try:
        while step_count < num_steps:
            action = 0.2 * env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            env.render()

            step_count += 1

            if step_count % 50 == 0:
                print(
                    f"Step {step_count}/{num_steps} | Episode {episode_count} | Reward: {reward.item():.3f}"
                )

            if done:
                episode_count += 1
                obs, info = env.reset(options=dict(reconfigure=True))

    except KeyboardInterrupt:
        print("\n\nVisualization stopped by user.")

    print(f"\nCompleted {step_count} steps across {episode_count} episodes.")
    env.close()
    print("Environment closed.")


def visualize_static(scene_path: str):
    """
    Visualize the environment in a static state (no actions).
    Useful for inspecting the scene setup.

    Args:
        scene_path: Path to scene.json
    """
    print("=" * 80)
    print("OpenReal2Sim ManiSkill Static Visualization")
    print("=" * 80)
    print(f"\nScene: {scene_path}")
    print("\nThis will show the initial scene setup.")
    print("You can use the viewer controls to inspect the scene.")
    print("Close the viewer window when done.")
    print("=" * 80)

    env = gym.make(
        "OpenReal2Sim-v0",
        scene_json_path=scene_path,
        num_envs=1,
        obs_mode="rgbd",  # Use RGB-D to see camera view
        render_mode="human",
    )

    obs, info = env.reset(seed=42, options=dict(reconfigure=True))

    print("\nEnvironment initialized. Viewer should be open.")
    print("Scene information:")
    print(f"  - Objects: {len(env.unwrapped.object_actors)}")
    for obj_id, obj_config in env.unwrapped.scene_config.objects.items():
        print(f"    - {obj_config.name}")
    print()

    input("Press Enter to close the visualization...")

    env.close()
    print("Environment closed.")


def main():
    project_root = Path(__file__).resolve().parents[3]
    default_scene_path = project_root / "../outputs/demo_image/scene/scene.json"
    parser = argparse.ArgumentParser(
        description="Visualize OpenReal2Sim ManiSkill environment"
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=str(default_scene_path),
        help="Path to scene.json file",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10000,
        help="Number of steps to run (for random action mode)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="random",
        choices=["random", "static"],
        help="Visualization mode: 'random' for random actions, 'static' for static scene inspection.",
    )

    args = parser.parse_args()

    scene_path = Path(args.scene)
    if not scene_path.exists():
        print(f"Error: Scene not found at {scene_path}")
        print(f"Current directory: {Path.cwd()}")

        outputs_dir = Path.cwd() / "outputs"
        if outputs_dir.exists():
            print("\nAvailable scenes:")
            for scene_json in sorted(outputs_dir.rglob("scene.json")):
                print(f"  - {scene_json.relative_to(Path.cwd())}")

        return 1

    if args.mode == "random":
        visualize_random_actions(str(scene_path), num_steps=args.steps)
    elif args.mode == "static":
        visualize_static(str(scene_path))

    return 0


if __name__ == "__main__":
    sys.exit(main())
