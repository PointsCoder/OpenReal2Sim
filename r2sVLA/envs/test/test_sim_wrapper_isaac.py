"""
Test script for PolicyEvaluationWrapper with RandomPolicy.

This test script demonstrates how to use the PolicyEvaluationWrapper
with a RandomPolicy for testing the simulation environment.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import List, Optional, Dict, Any
import torch

# Initialize Isaac Lab AppLauncher FIRST, before any other Isaac Lab imports
# This is required to set up the carb module and other Isaac Sim dependencies
from isaaclab.app import AppLauncher

# Parse arguments
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

# Add parent directories to path
file_path = Path(__file__).resolve()
sys.path.append(str(file_path.parent.parent))
sys.path.append(str(file_path.parent.parent.parent))

# Now we can safely import modules that depend on Isaac Lab
from envs.cfgs.eval_cfg import EvaluationConfig
from envs.cfgs.policy_interface import BasePolicy, Action
from envs.cfgs.task_cfg import load_task_cfg
from envs.sim_wrapper_isaac import PolicyEvaluationWrapper
from envs.make_env_isaac import make_env



class RandomPolicy(BasePolicy):
    """
    Random policy that generates random actions.
    Useful for testing and baseline comparisons.
    
    For 'qpos' action_type: generates random joint positions within safe limits
    For 'ee' action_type: generates random end-effector poses within workspace
    """
    
    def __init__(
        self,
        observation_keys: List[str],
        action_type: str,
        image_resolution: List[int],
        num_joints: int = 7,  # Default for Franka (7 arm joints)
        qpos_low: Optional[torch.Tensor] = None,
        qpos_high: Optional[torch.Tensor] = None,
        ee_pos_low: Optional[torch.Tensor] = None,
        ee_pos_high: Optional[torch.Tensor] = None,
        device: str = "cuda:0",
    ):
        """
        Initialize random policy.
        
        Args:
            observation_keys: List of observation keys (not used for random policy)
            action_type: 'qpos' or 'ee'
            image_resolution: Image resolution (not used for random policy)
            num_joints: Number of joints (for qpos action_type)
            qpos_low: Lower bounds for joint positions [num_joints]
            qpos_high: Upper bounds for joint positions [num_joints]
            ee_pos_low: Lower bounds for end-effector position [3]
            ee_pos_high: Upper bounds for end-effector position [3]
            device: Device to run on
        """
        super().__init__(observation_keys, action_type, image_resolution)
        self.num_joints = num_joints
        self.device = device
        
        # Set default joint limits (Franka Panda safe ranges)
        if qpos_low is None:
            # Franka Panda joint limits (radians)
            self.qpos_low = torch.tensor([
                -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973
            ], device=device, dtype=torch.float32)
        else:
            self.qpos_low = qpos_low.to(device) if torch.is_tensor(qpos_low) else torch.tensor(qpos_low, device=device, dtype=torch.float32)
        
        if qpos_high is None:
            self.qpos_high = torch.tensor([
                2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973
            ], device=device, dtype=torch.float32)
        else:
            self.qpos_high = qpos_high.to(device) if torch.is_tensor(qpos_high) else torch.tensor(qpos_high, device=device, dtype=torch.float32)
        
       
    def get_action(self, observation: Dict[str, Any]) -> Action:
        """
        Generate random action.
        
        Args:
            observation: Observation dict (ignored for random policy)
        
        Returns:
            Action with random values
        """
        # Random gripper state (50% chance open/closed)
        gripper_open = torch.rand(1, device=self.device).item() > 0.5
        
        if self.action_type == 'qpos':
            # Generate random joint positions within limits
            qpos = torch.rand(self.num_joints, device=self.device) * (self.qpos_high - self.qpos_low) + self.qpos_low
            # Add batch dimension [1, num_joints]
            qpos = qpos.unsqueeze(0)
            return Action(action_type='qpos', qpos=qpos, gripper_open=gripper_open)
        
        elif self.action_type == 'ee':
            # Generate random end-effector position within workspace
            ee_pos = torch.rand(3, device=self.device) * (self.ee_pos_high - self.ee_pos_low) + self.ee_pos_low
            
            # Generate random quaternion (wxyz format)
            # Sample from uniform distribution on unit sphere
            quat = torch.randn(4, device=self.device)
            quat = quat / torch.norm(quat)  # Normalize to unit quaternion
            # Ensure w is positive (standard convention)
            if quat[0] < 0:
                quat = -quat
            
            # Combine position and quaternion [7]
            ee_pose = torch.cat([ee_pos, quat])
            # Add batch dimension [1, 7]
            ee_pose = ee_pose.unsqueeze(0)
            
            return Action(action_type='ee', ee_pose=ee_pose, gripper_open=gripper_open)
        
        else:
            raise ValueError(f"Invalid action_type: {self.action_type}")
    
    def reset(self) -> None:
        """Reset policy state (no state to reset for random policy)."""
        pass



def clean_task_cfg(task_cfg):
    """Clean task config paths."""
    def rename_path(path: str, or2s_dir: Path) -> str:
        return path.replace("/app", str(or2s_dir))
    
    o2s_dir = Path("/home/peiqiduan/OpenReal2Sim-dev")
    task_cfg.background_cfg.background_rgb_path = rename_path(
        task_cfg.background_cfg.background_rgb_path, o2s_dir
    )
    task_cfg.background_cfg.background_mesh_path = rename_path(
        task_cfg.background_cfg.background_mesh_path, o2s_dir
    )
    task_cfg.background_cfg.background_usd_path = rename_path(
        task_cfg.background_cfg.background_usd_path, o2s_dir
    )
    for obj in task_cfg.objects:
        obj.mesh_path = rename_path(obj.mesh_path, o2s_dir)
        obj.usd_path = rename_path(obj.usd_path, o2s_dir)
    return task_cfg


def test_sim_wrapper_isaac():
    """Test PolicyEvaluationWrapper with RandomPolicy."""
    # Load task config
    task_cfg = load_task_cfg("/app/tasks/demo_video/task.json")
    # task_cfg = clean_task_cfg(task_cfg)
    
    # Use default evaluation config
    eval_cfg = EvaluationConfig()
    eval_cfg.max_steps = 100  # Limit steps for testing
    eval_cfg.record_video = True
    
    num_envs = 1

    env, _ = make_env(
        task_cfg,
        eval_cfg=eval_cfg,
        num_envs=num_envs,
        device="cuda:0",
        bg_simplify=False
    )
    sim, scene = env.sim, env.scene
    my_sim = PolicyEvaluationWrapper(sim, scene, task_cfg, eval_cfg, num_envs)
    
    # Create random policy
    random_policy = RandomPolicy(
        observation_keys=['rgb'],
        action_type='qpos',  # or 'ee'
        image_resolution=[224, 224],
        num_joints=7,  # Franka has 7 arm joints
        device="cuda:0"
    )
    
    # Set policy and evaluate
    my_sim.set_policy(random_policy)
    my_sim.out_dir = Path("/app/r2sVLA/results")
    res = []
    for i in range(eval_cfg.num_trials):
        result = my_sim.evaluate_episode()
        print(f"Trial {i+1} result: {result}")
        my_sim.save_data()
        res.append(result)
    
    success_num = 0
    for r in res:
        success_num += r['success'].sum().item()
    success_rate = success_num / (eval_cfg.num_trials * num_envs)
    print(f"Success rate: {success_rate}")
    
    env.close()


if __name__ == "__main__":
    try:
        test_sim_wrapper_isaac()
    except Exception as e:
        print(f"[ERR] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        simulation_app.close()

