"""
Simplified policy evaluation script for r2sVLA.
Evaluates ACT policy in Isaac Lab simulation environment.
"""
from __future__ import annotations

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import numpy as np
from datetime import datetime

# Add paths
sys.path.append(str(Path(__file__).parent.parent))

from envs.make_env_isaac import make_env
from envs.sim_wrapper_isaac import PolicyEvaluationWrapper
from envs.cfgs.eval_cfg import EvaluationConfig
from envs.cfgs.task_cfg import TaskCfg, load_task_cfg
from algos.act.act_policy_wrapper import ACTPolicyWrapper


def load_policy_config(config_path: str) -> Dict[str, Any]:
    """Load policy configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_act_policy(
    policy_config: Dict[str, Any],
    ckpt_path: str,
    observation_keys: list,
    action_type: str,
    image_resolution: list,
) -> ACTPolicyWrapper:
    """Create and load ACT policy."""
    
    # Extract policy-specific config
    policy_class = policy_config.get('policy_class', 'ACT')
    temporal_agg = policy_config.get('temporal_agg', False)
    temporal_agg_num_queries = policy_config.get('temporal_agg_num_queries', 16)
    
    # Build policy config dict for ACT
    act_policy_config = {
        'lr': policy_config.get('lr', 1e-4),
        'num_queries': policy_config.get('chunk_size', 100),
        'kl_weight': policy_config.get('kl_weight', 10.0),
        'hidden_dim': policy_config.get('hidden_dim', 512),
        'dim_feedforward': policy_config.get('dim_feedforward', 3200),
        'lr_backbone': policy_config.get('lr_backbone', 1e-5),
        'backbone': policy_config.get('backbone', 'resnet18'),
        'enc_layers': policy_config.get('enc_layers', 4),
        'dec_layers': policy_config.get('dec_layers', 7),
        'nheads': policy_config.get('nheads', 8),
        'camera_names': policy_config.get('camera_names', ['cam_high']),
        'task_name': policy_config.get('task_name', 'default'),
        'state_dim': policy_config.get('state_dim', 8),  # 7 joints + 1 gripper for Franka
    }
    
    # Create policy wrapper
    policy = ACTPolicyWrapper(
        observation_keys=observation_keys,
        action_type=action_type,
        image_resolution=image_resolution,
        policy_class=policy_class,
        ckpt_path=ckpt_path,
        policy_config=act_policy_config,
        temporal_agg=temporal_agg,
        temporal_agg_num_queries=temporal_agg_num_queries,
    )
    
    return policy


def eval_policy(
    env: PolicyEvaluationWrapper,
    policy: ACTPolicyWrapper,
    num_episodes: int = 100,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Evaluate policy for multiple episodes.
    
    Args:
        env: PolicyEvaluationWrapper environment
        policy: ACT policy wrapper
        num_episodes: Number of episodes to evaluate
        seed: Random seed
    
    Returns:
        Dictionary with evaluation results
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Set policy in environment
    env.set_policy(policy)
    
    success_count = 0
    total_steps = 0
    episode_results = []
    
    print(f"\n{'='*60}")
    print(f"Starting evaluation: {num_episodes} episodes")
    print(f"{'='*60}\n")
    
    for episode_idx in range(num_episodes):
        # Reset environment
        env.reset(env_ids=[0])  # Single environment evaluation
        
        # Run episode
        result = env.evaluate_episode(reset_env=False, env_ids=[0])
        
        success = result['success'][0].item()
        steps = result['steps'][0].item()
        
        success_count += int(success)
        total_steps += steps
        
        episode_results.append({
            'episode': episode_idx,
            'success': success,
            'steps': steps,
        })
        
        # Print progress
        success_rate = success_count / (episode_idx + 1) * 100
        print(
            f"Episode {episode_idx+1}/{num_episodes} | "
            f"Success: {'✓' if success else '✗'} | "
            f"Steps: {steps} | "
            f"Success Rate: {success_rate:.1f}%"
        )
    
    # Summary
    final_success_rate = success_count / num_episodes * 100
    avg_steps = total_steps / num_episodes
    
    print(f"\n{'='*60}")
    print(f"Evaluation Summary")
    print(f"{'='*60}")
    print(f"Total Episodes: {num_episodes}")
    print(f"Success Count: {success_count}")
    print(f"Success Rate: {final_success_rate:.1f}%")
    print(f"Average Steps: {avg_steps:.1f}")
    print(f"{'='*60}\n")
    
    return {
        'num_episodes': num_episodes,
        'success_count': success_count,
        'success_rate': final_success_rate,
        'avg_steps': avg_steps,
        'episode_results': episode_results,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate ACT policy')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to evaluation config YAML file (optional if ckpt_dir has eval_config.yaml)')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to policy checkpoint file')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Path to checkpoint directory (for auto-loading config, optional if --config provided)')
    parser.add_argument('--num_episodes', type=int, default=100,
                        help='Number of episodes to evaluate')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Auto-detect config from checkpoint directory if not provided
    if args.config is None:
        if args.ckpt_dir is None:
            # Try to infer ckpt_dir from ckpt_path
            ckpt_path = Path(args.ckpt_path)
            if ckpt_path.is_file():
                args.ckpt_dir = ckpt_path.parent
            else:
                raise ValueError("Either --config or --ckpt_dir must be provided, or ckpt_path must be in a directory with eval_config.yaml")
        
        # Look for eval_config.yaml in checkpoint directory
        eval_config_path = Path(args.ckpt_dir) / 'eval_config.yaml'
        if eval_config_path.exists():
            print(f"Auto-loading config from {eval_config_path}")
            args.config = str(eval_config_path)
        else:
            raise ValueError(f"Config file not found. Please provide --config or ensure {eval_config_path} exists")
    
    # Load config
    config = load_policy_config(args.config)
    
    # Extract configuration
    task_cfg_path = config.get('task_cfg_path', None)
    task_cfg_dict = config.get('task_cfg', {})
    eval_cfg_dict = config.get('eval_cfg', {})
    policy_cfg_dict = config.get('policy_cfg', {})
    
    # Create TaskCfg - load from JSON file if path provided, otherwise use dict
    if task_cfg_path:
        task_cfg = load_task_cfg(Path(task_cfg_path))
    elif task_cfg_dict:
        # For now, require task_cfg_path. Can add from_dict later if needed
        raise ValueError("task_cfg_path is required in config. Please provide path to task.json file.")
    else:
        raise ValueError(
            "task_cfg_path is required in config. "
            "If using auto-loaded config from checkpoint, you still need to provide task_cfg_path. "
            "You can either:\n"
            "  1. Create a full config file with task_cfg_path and eval_cfg\n"
            "  2. Or use --config with a complete config file that includes task_cfg_path"
        )
    
    # Create EvaluationConfig (use defaults if not provided)
    if eval_cfg_dict:
        eval_cfg = EvaluationConfig.from_dict(eval_cfg_dict)
    else:
        # Use default evaluation config if not provided
        print("Warning: eval_cfg not found in config, using defaults")
        eval_cfg = EvaluationConfig()
    
    # Set up save directory
    if args.save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_dir = f"eval_results/{task_cfg.task_name}/{timestamp}"
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    eval_cfg.video_save_dir = save_dir if eval_cfg.record_video else None
    
    # Create environment
    print("Creating environment...")
    sim, scene, env_cfg = make_env(task_cfg=task_cfg)
    num_envs = 1  # Single environment for evaluation
    env = PolicyEvaluationWrapper(
        sim=sim,
        scene=scene,
        task_cfg=task_cfg,
        eval_cfg=eval_cfg,
        num_envs=num_envs,
    )
    print("Environment created successfully")
    
    # Create policy
    print("Loading policy...")
    observation_keys = policy_cfg_dict.get('observation_keys', ['rgb'])
    action_type = policy_cfg_dict.get('action_type', 'qpos')
    image_resolution = policy_cfg_dict.get('image_resolution', [224, 224])
    
    policy = create_act_policy(
        policy_config=policy_cfg_dict,
        ckpt_path=args.ckpt_path,
        observation_keys=observation_keys,
        action_type=action_type,
        image_resolution=image_resolution,
    )
    print("Policy loaded successfully")
    
    # Evaluate
    results = eval_policy(
        env=env,
        policy=policy,
        num_episodes=args.num_episodes,
        seed=args.seed,
    )
    
    # Save results
    results_path = save_dir / "results.yaml"
    with open(results_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    
    print(f"Results saved to {results_path}")
    
    return results


if __name__ == '__main__':
    main()

