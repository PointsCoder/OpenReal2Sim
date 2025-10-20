#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for OpenReal2Sim ManiSkill environment.

This script validates that the environment can be created and run correctly.
"""

import argparse
import numpy as np
from pathlib import Path
import gymnasium as gym

# Import to register the environment
import sys

sys.path.append("/home/haoyang/project/haoyang/OpenReal2Sim")
from openreal2sim.simulation.maniskill import OpenReal2SimEnv


def test_environment_creation(scene_json_path: str):
    """Test basic environment creation."""
    print("=" * 80)
    print("TEST 1: Environment Creation")
    print("=" * 80)

    try:
        env = gym.make(
            "OpenReal2Sim-v0",
            scene_json_path=scene_json_path,
            num_envs=1,
            obs_mode="state",
            render_mode="rgb_array",
        )
        print("✓ Environment created successfully")
        return env
    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        raise


def test_reset(env):
    """Test environment reset."""
    print("\n" + "=" * 80)
    print("TEST 2: Environment Reset")
    print("=" * 80)

    try:
        obs, info = env.reset()
        print("✓ Environment reset successfully")
        return obs, info
    except Exception as e:
        print(f"✗ Environment reset failed: {e}")
        raise


def test_step(env):
    """Test environment step."""
    print("\n" + "=" * 80)
    print("TEST 3: Environment Step")
    print("=" * 80)

    try:
        # Take a random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print("✓ Environment step executed successfully")
        print(f"  Action shape: {action.shape}")
        print(f"  Reward: {reward}")
        print(f"  Terminated: {terminated}")
        print(f"  Truncated: {truncated}")
        return obs, reward, terminated, truncated, info
    except Exception as e:
        print(f"✗ Environment step failed: {e}")
        raise


def test_multiple_steps(env, num_steps=10):
    """Test multiple environment steps."""
    print("\n" + "=" * 80)
    print(f"TEST 4: Multiple Steps ({num_steps} steps)")
    print("=" * 80)

    try:
        env.reset()

        for i in range(num_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                print(f"  Episode ended at step {i + 1}")
                env.reset()

        print(f"✓ Successfully executed {num_steps} steps")
    except Exception as e:
        print(f"✗ Multiple steps failed: {e}")
        raise


def test_rendering(env):
    """Test environment rendering."""
    print("\n" + "=" * 80)
    print("TEST 5: Rendering")
    print("=" * 80)

    try:
        env.reset()

        # Render a frame
        img = env.render()

        if img is not None:
            print(f"✓ Rendering successful")
            print(f"  Image shape: {img.shape}")
            print(f"  Image dtype: {img.dtype}")
            return img
        else:
            print("✗ Rendering returned None")
            return None
    except Exception as e:
        print(f"✗ Rendering failed: {e}")
        raise


def test_observation_modes(scene_json_path: str):
    """Test different observation modes."""
    print("\n" + "=" * 80)
    print("TEST 6: Observation Modes")
    print("=" * 80)

    obs_modes = ["state", "rgbd", "state_dict"]

    for obs_mode in obs_modes:
        try:
            print(f"\n  Testing obs_mode='{obs_mode}'...")
            env = gym.make(
                "OpenReal2Sim-v0",
                scene_json_path=scene_json_path,
                num_envs=1,
                obs_mode=obs_mode,
            )
            obs, _ = env.reset()
            print(f"  ✓ obs_mode='{obs_mode}' works")

            if isinstance(obs, dict):
                print(f"    Observation keys: {list(obs.keys())}")

            env.close()
        except Exception as e:
            print(f"  ✗ obs_mode='{obs_mode}' failed: {e}")


def test_scene_info(env):
    """Print information about the loaded scene."""
    print("\n" + "=" * 80)
    print("TEST 7: Scene Information")
    print("=" * 80)

    try:
        print(f"  Scene JSON: {env.scene_json_path}")
        print(f"  Number of objects: {len(env.object_actors)}")
        print(f"  Objects:")
        for obj_id, obj_config in env.scene_config.objects.items():
            print(f"    - {obj_config.name} (ID: {obj_id})")

        print(f"  Camera config:")
        cam = env.scene_config.camera
        print(f"    - Resolution: {cam.width}x{cam.height}")
        print(f"    - Intrinsics: fx={cam.fx:.2f}, fy={cam.fy:.2f}")
        print(f"    - Position: {cam.position}")

        print(f"  Robot: {env.agent.robot.name}")
        print(f"  Robot DOF: {env.agent.robot.dof}")

        print("✓ Scene information retrieved successfully")
    except Exception as e:
        print(f"✗ Failed to retrieve scene information: {e}")


def save_rendered_image(img, output_path="test_render.png"):
    """Save a rendered image."""
    try:
        import matplotlib.pyplot as plt
        import torch

        # Convert CUDA tensor to numpy if needed
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()

        # Handle batch dimension if present
        if img.ndim == 4 and img.shape[0] == 1:
            img = img[0]  # Remove batch dimension

        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        plt.axis("off")
        plt.title("OpenReal2Sim ManiSkill Environment Render")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"\n✓ Saved rendered image to: {output_path}")
    except Exception as e:
        print(f"\n✗ Failed to save image: {e}")


def run_all_tests(scene_json_path: str, save_render: bool = True):
    """Run all tests."""
    print("\n" + "=" * 80)
    print("OpenReal2Sim ManiSkill Environment Test Suite")
    print("=" * 80)
    print(f"\nScene: {scene_json_path}")

    try:
        # Test 1: Create environment
        env = test_environment_creation(scene_json_path)

        # Test 2: Reset
        obs, info = test_reset(env)

        # Test 3: Single step
        test_step(env)

        # Test 4: Multiple steps
        test_multiple_steps(env, num_steps=20)

        # Test 5: Rendering
        img = test_rendering(env)

        # Test 7: Scene info
        test_scene_info(env)

        # Save rendered image if requested
        if save_render and img is not None:
            save_rendered_image(img)

        # Clean up
        env.close()

        # Test 6: Observation modes (creates new envs)
        test_observation_modes(scene_json_path)

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED ✓")
        print("=" * 80)

    except Exception as e:
        print("\n" + "=" * 80)
        print("TESTS FAILED ✗")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test OpenReal2Sim ManiSkill environment"
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="outputs/demo_genvideo/scene/scene.json",
        help="Path to scene.json file",
    )
    parser.add_argument(
        "--no-render-save", action="store_true", help="Don't save rendered image"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run quick test (fewer steps)"
    )

    args = parser.parse_args()

    # Validate scene path
    scene_path = Path(args.scene)
    if not scene_path.exists():
        print(f"Error: Scene JSON not found: {scene_path}")
        print(f"Current working directory: {Path.cwd()}")

        # Try to find available scenes
        outputs_dir = Path.cwd() / "outputs"
        if outputs_dir.exists():
            print("\nAvailable scenes:")
            for scene_json in outputs_dir.rglob("scene.json"):
                print(f"  - {scene_json.relative_to(Path.cwd())}")

        return 1

    # Run tests
    success = run_all_tests(
        scene_json_path=str(scene_path), save_render=not args.no_render_save
    )

    return 0 if success else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
