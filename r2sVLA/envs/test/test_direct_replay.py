"""
Direct Replay Policy: Load actions directly from HDF5 files and replay in simulation.

This policy loads a sequence of actions from an HDF5 file and replays them step by step.
It's useful for testing the simulation environment with recorded demonstrations.
"""
from __future__ import annotations

import h5py
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import yaml
import sys
import argparse

# Add parent directories to path for imports
file_path = Path(__file__).resolve()
sys.path.append(str(file_path.parent.parent))
sys.path.append(str(file_path.parent.parent.parent))

# Initialize Isaac Lab AppLauncher FIRST (before importing modules that need carb)
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

# Now we can import modules that depend on carb
from envs.cfgs.policy_interface import BasePolicy, Action


class DirectReplayPolicy(BasePolicy):
    """
    Policy that directly replays actions from HDF5 file or demo folder.
    
    Supports both 'qpos' (joint position) and 'ee' (end-effector) action types.
    
    For 'qpos' action type:
        - Loads joint positions and gripper commands
        - Expected HDF5 structure:
            /joint_action/joint_pos: (episode_len, 7) - 7D joint positions
            /joint_action/gripper_cmd: (episode_len, 2) - gripper commands [0.00 (close) to 0.04 (open)]
        - Demo folder: joint_pos.npy, gripper_cmd.npy, or actions.npy
    
    For 'ee' action type:
        - Loads end-effector poses (position + quaternion) and gripper commands
        - Expected format: (episode_len, 8) - [pos(3), quat(4), gripper(1)]
        - Demo folder: actions.npy (format: [pos[0:3], quat[3:7], gripper[7]])
        - HDF5: /ee_pose (if available)
    """
    
    # Franka gripper constants (same as in ACT training)
    FRANKA_GRIPPER_OPEN = 0.04
    FRANKA_GRIPPER_CLOSE = 0.00
    
    def __init__(
        self,
        observation_keys: List[str],
        action_type: str,
        image_resolution: List[int],
        hdf5_path: Optional[str] = None,
        demo_folder: Optional[str] = None,
        task_folder: Optional[str] = None,
        h5py_path: Optional[str] = None,
        episode_idx: int = 0,
        env_idx: int = 0,
        device: str = "cuda:0",
    ):
        """
        Initialize DirectReplayPolicy.
        
        Args:
            observation_keys: List of observation keys (not used, but required by BasePolicy)
            action_type: 'qpos' or 'ee' - type of actions to replay
            image_resolution: Image resolution (not used, but required by BasePolicy)
            hdf5_path: Path to HDF5 file containing actions, or path to directory with episode_*.hdf5 files
            demo_folder: Path to demo folder (e.g., outputs/demo_video/demos) - alternative to hdf5_path
            task_folder: Path to task folder containing config.json (optional, for loading task config)
            h5py_path: Path to h5py directory (optional, alternative to task_folder)
            episode_idx: Episode index if hdf5_path is a directory or demo_folder (default: 0)
            env_idx: Environment index for demo_folder (default: 0)
            device: Device to run on
        """
        super().__init__(observation_keys, action_type, image_resolution)
        
        if action_type not in ['qpos', 'ee_direct', 'ee_cam', 'ee_l']:
            raise ValueError(f"DirectReplayPolicy supports 'qpos' or 'ee' action_type, got '{action_type}'")
        
        self.device = device
        self.current_step = 0
        
        # Load task config if provided
        self.task_cfg = None
        if task_folder is not None:
            config_path = Path(task_folder) / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.task_cfg = json.load(f)
                print(f"Loaded task config from {config_path}")
        elif h5py_path is not None:
            # Try to find config.json in h5py_path or parent directory
            config_path = Path(h5py_path) / "config.json"
            if not config_path.exists():
                config_path = Path(h5py_path).parent / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.task_cfg = json.load(f)
                print(f"Loaded task config from {config_path}")
        
        # Load actions - prioritize demo_folder over hdf5_path
        if demo_folder is not None:
            # Load from demo folder
            demo_folder_obj = Path(demo_folder)
            if not demo_folder_obj.exists():
                raise FileNotFoundError(f"Demo folder not found: {demo_folder}")
            
            # Find demo_XXX folders
            demo_dirs = sorted(demo_folder_obj.glob("demo_*"))
            if len(demo_dirs) == 0:
                raise ValueError(f"No demo_* folders found in {demo_folder}")
            if episode_idx >= len(demo_dirs) and episode_idx != -1:
                raise ValueError(f"Episode index {episode_idx} out of range (found {len(demo_dirs)} demos)")
            
            demo_dir = demo_dirs[episode_idx]
            # Find env_XXX folders
            env_dirs = sorted(demo_dir.glob("env_*"))
            if len(env_dirs) == 0:
                raise ValueError(f"No env_* folders found in {demo_dir}")
            if env_idx >= len(env_dirs) and env_idx != -1:
                raise ValueError(f"Environment index {env_idx} out of range (found {len(env_dirs)} envs)")
            
            env_dir = env_dirs[env_idx]
            print(f"Loading from demo folder: {demo_dir.name}/{env_dir.name}")
            
            # Load actions from demo folder
            if self.action_type == 'qpos':
                self.actions = self._load_qpos_actions_from_demo_folder(str(env_dir))
            elif self.action_type == 'ee_cam':
                self.actions = self._load_ee_cam_actions_from_demo_folder(str(env_dir))
            elif self.action_type == 'ee_l':
                self.actions = self._load_ee_l_actions_from_demo_folder(str(env_dir))
            else:  # 'ee' or 'ee_direct'
                self.actions = self._load_sparse_ee_actions_from_demo_folder(str(env_dir))
            print(f"Loaded {len(self.actions)} actions from {env_dir}")
            
        elif hdf5_path is not None:
            # Load actions from HDF5 file
            hdf5_path_obj = Path(hdf5_path)
            if hdf5_path_obj.is_dir():
                # Directory: look for episode_*.hdf5 files
                episode_files = sorted(hdf5_path_obj.glob("episode_*.hdf5"))
                if len(episode_files) == 0:
                    raise ValueError(f"No episode_*.hdf5 files found in {hdf5_path}")
                if episode_idx >= len(episode_files):
                    raise ValueError(f"Episode index {episode_idx} out of range (found {len(episode_files)} episodes)")
                hdf5_file_path = episode_files[episode_idx]
                print(f"Loading episode {episode_idx} from {hdf5_file_path}")
            else:
                # Single file
                hdf5_file_path = hdf5_path_obj
                if not hdf5_file_path.exists():
                    raise FileNotFoundError(f"HDF5 file not found: {hdf5_file_path}")
            
            # Load actions from HDF5
            if self.action_type == 'qpos':
                self.actions = self._load_actions_from_hdf5(str(hdf5_file_path))
            else: 
                self.actions = self._load_ee_actions_from_hdf5(str(hdf5_file_path))
            print(f"Loaded {len(self.actions)} actions from {hdf5_file_path}")
        else:
            raise ValueError("Either 'hdf5_path' or 'demo_folder' must be provided")
    
    def _load_gripper_cmd(self, env_dir_path: Path) -> np.ndarray:
        """Helper function to load and convert gripper_cmd.npy to boolean array."""
        gripper_cmd_path = env_dir_path / "gripper_cmd.npy"
        if not gripper_cmd_path.exists():
            raise FileNotFoundError(f"gripper_cmd.npy not found in {env_dir_path}")
        
        gripper_cmd = np.load(str(gripper_cmd_path))  # (episode_len, 2) or (episode_len,)
        
        # Convert gripper_cmd to boolean (open/close)
        # gripper_cmd[:, 0] is in [0.00 (close), 0.04 (open)]
        # Threshold: > 0.02 means open
        if gripper_cmd.ndim == 2:
            gripper_raw = gripper_cmd[:, 0]  # Use first gripper
        else:
            gripper_raw = gripper_cmd  # Already 1D
        gripper_open = (gripper_raw > 0.02)  # Boolean array
        
        return gripper_open
    
    def _load_qpos_actions_from_demo_folder(self, env_dir: str) -> List[Action]:
        """
        Load joint position actions from demo folder.
        Loads from: joint_pos_des.npy + gripper_cmd.npy
        
        Args:
            env_dir: Path to env_XXX directory containing .npy files
            
        Returns:
            List of Action objects with action_type='qpos'
        """
        env_dir_path = Path(env_dir)
        
        # Load joint_pos_des.npy
        joint_pos_des_path = env_dir_path / "joint_pos_des.npy"
        if not joint_pos_des_path.exists():
            raise FileNotFoundError(f"joint_pos_des.npy not found in {env_dir}")
        
        joint_pos = np.load(str(joint_pos_des_path))  # May be (episode_len, 7) or (episode_len * 7,)
        joint_pos = joint_pos.reshape(-1, 7)  # Ensure (episode_len, 7)
        
        # Load gripper command
        gripper_open = self._load_gripper_cmd(env_dir_path)
        
        # Ensure same length
        min_len = min(len(joint_pos), len(gripper_open))
        joint_pos = joint_pos[:min_len]
        gripper_open = gripper_open[:min_len]
        
        # Convert to list of Action objects
        actions = []
        for i in range(len(joint_pos)):
            qpos_tensor = torch.from_numpy(joint_pos[i]).float().to(self.device)  # [7]
            qpos_batch = qpos_tensor.unsqueeze(0)  # [1, 7]
            
            action = Action(
                action_type='qpos',
                qpos=qpos_batch,
                gripper_open=bool(gripper_open[i])
            )
            actions.append(action)
        
        return actions
    
    def _load_sparse_ee_actions_from_demo_folder(self, env_dir: str) -> List[Action]:
        """
        Load sparse end-effector actions from demo folder.
        Loads from: actions.npy (format: [pos[0:3], quat[3:7], gripper[7]]) + gripper_cmd.npy
        
        Args:
            env_dir: Path to env_XXX directory containing .npy files
            
        Returns:
            List of Action objects with action_type='ee' or 'ee_direct'
        """
        env_dir_path = Path(env_dir)
        
        # Load actions.npy (sparse waypoints)
        actions_path = env_dir_path / "actions.npy"
        if not actions_path.exists():
            raise FileNotFoundError(f"actions.npy not found in {env_dir}")
        
        actions = np.load(str(actions_path))  # (episode_len, 8)
        
        # Extract components
        positions = actions[:, 0:3]  # (episode_len, 3)
        quaternions = actions[:, 3:7]  # (episode_len, 4)
        
        # Load gripper command from gripper_cmd.npy (preferred over actions[:, 7])
        gripper_open = actions[:, 7] < 0.5
        
        # Ensure same length
        min_len = min(len(positions), len(gripper_open))
        positions = positions[:min_len]
        quaternions = quaternions[:min_len]
        gripper_open = gripper_open[:min_len]
        
        # Convert to list of Action objects
        actions_list = []
        for i in range(len(positions)):
            # Normalize quaternion if needed
            quat = quaternions[i]
            quat_norm = np.linalg.norm(quat)
            if abs(quat_norm - 1.0) > 0.1:
                quat = quat / (quat_norm + 1e-8)
            
            # Create ee_pose tensor [1, 7] = [pos(3), quat(4)] in wxyz format
            ee_pose = torch.zeros((1, 7), dtype=torch.float32, device=self.device)
            ee_pose[0, 0:3] = torch.from_numpy(positions[i]).float()
            ee_pose[0, 3:7] = torch.from_numpy(quat).float()
            
            action = Action(
                action_type='ee' if self.action_type == 'ee' else 'ee_direct',
                ee_pose=ee_pose,
                gripper_open=bool(gripper_open[i])
            )
            actions_list.append(action)
        
        return actions_list
    
    def _load_ee_cam_actions_from_demo_folder(self, env_dir: str) -> List[Action]:
        """
        Load dense end-effector actions in camera frame from demo folder.
        Loads from: ee_pose_cam.npy + gripper_cmd.npy
        
        Args:
            env_dir: Path to env_XXX directory containing .npy files
            
        Returns:
            List of Action objects with action_type='ee_cam'
        """
        env_dir_path = Path(env_dir)
        
        # Load ee_pose_cam.npy (camera frame end-effector pose)
        ee_pose_cam_path = env_dir_path / "ee_pose_cam.npy"
        if not ee_pose_cam_path.exists():
            raise FileNotFoundError(
                f"ee_pose_cam.npy not found in {env_dir}. "
                f"For 'ee_cam' action_type, ee_pose_cam.npy with format [pos(3), quat(4)] in camera frame is required."
            )
        
        ee_pose_cam = np.load(str(ee_pose_cam_path))  # (episode_len, 7)
        
        # Load gripper command
        gripper_open = self._load_gripper_cmd(env_dir_path)
        
        # Ensure same length
        min_len = min(len(ee_pose_cam), len(gripper_open))
        ee_pose_cam = ee_pose_cam[:min_len]
        gripper_open = gripper_open[:min_len]
        
        # Convert to list of Action objects
        actions_list = []
        for i in range(len(ee_pose_cam)):
            # ee_pose_cam is [7] = [pos(3), quat(4)] in camera frame, wxyz format
            ee_pose_cam_tensor = torch.from_numpy(ee_pose_cam[i]).float().to(self.device)  # [7]
            ee_pose_cam_batch = ee_pose_cam_tensor.unsqueeze(0)  # [1, 7]
            
            action = Action(
                action_type='ee_cam',
                ee_pose=ee_pose_cam_batch,
                gripper_open=bool(gripper_open[i])
            )
            actions_list.append(action)
        
        return actions_list
    
    def _load_ee_l_actions_from_demo_folder(self, env_dir: str) -> List[Action]:
        """
        Load dense end-effector actions in local frame from demo folder.
        Loads from: ee_pose_l.npy + gripper_cmd.npy
        
        Args:
            env_dir: Path to env_XXX directory containing .npy files
            
        Returns:
            List of Action objects with action_type='ee_l'
        """
        env_dir_path = Path(env_dir)
        
        # Load ee_pose_l.npy (local frame end-effector pose)
        ee_pose_l_path = env_dir_path / "ee_pose_l.npy"
        if not ee_pose_l_path.exists():
            raise FileNotFoundError(
                f"ee_pose_l.npy not found in {env_dir}. "
                f"For 'ee_l' action_type, ee_pose_l.npy with format [pos(3), quat(4)] in local frame is required."
            )
        
        ee_pose_l = np.load(str(ee_pose_l_path))  # (episode_len, 7)
        
        # Load gripper command
        gripper_open = self._load_gripper_cmd(env_dir_path)
        
        # Ensure same length
        min_len = min(len(ee_pose_l), len(gripper_open))
        ee_pose_l = ee_pose_l[:min_len]
        gripper_open = gripper_open[:min_len]
        
        # Convert to list of Action objects
        actions_list = []
        for i in range(len(ee_pose_l)):
            # ee_pose_l is [7] = [pos(3), quat(4)] in local frame, wxyz format
            ee_pose_l_tensor = torch.from_numpy(ee_pose_l[i]).float().to(self.device)  # [7]
            ee_pose_l_batch = ee_pose_l_tensor.unsqueeze(0)  # [1, 7]
            
            action = Action(
                action_type='ee_l',
                ee_pose=ee_pose_l_batch,
                gripper_open=bool(gripper_open[i])
            )
            actions_list.append(action)
        
        return actions_list

    def _load_ee_actions_from_hdf5(self, hdf5_path: str) -> List[Action]:
        """
        Load end-effector actions from HDF5 file.
        
        Args:
            hdf5_path: Path to HDF5 file
            
        Returns:
            List of Action objects with action_type='ee'
        """
        with h5py.File(hdf5_path, 'r') as f:
            # Check for ee_pose in various possible locations
            ee_pose = None
            gripper_cmd = None
            
            # Try different possible paths
            if '/ee_pose' in f:
                ee_pose = f['/ee_pose'][:]  # (episode_len, 7)
            elif '/observation/ee_pose' in f:
                ee_pose = f['/observation/ee_pose'][:]  # (episode_len, 7)
            elif '/action' in f:
                # Check if action is 8D (pos + quat + gripper)
                action = f['/action'][:]  # (episode_len, 8)
                if action.shape[1] == 8:
                    ee_pose = action[:, :7]  # (episode_len, 7)
                    gripper_raw = action[:, 7]  # (episode_len,)
                    gripper_cmd = (gripper_raw > 0.5)  # Boolean array
            
            if ee_pose is None:
                raise ValueError(
                    f"HDF5 file must contain '/ee_pose', '/observation/ee_pose', or '/action' (8D) key for 'ee' action_type"
                )
            
            # Get gripper command if not already extracted
            if gripper_cmd is None:
                if '/joint_action/gripper_cmd' in f:
                    gripper_cmd_array = f['/joint_action/gripper_cmd'][:]  # (episode_len, 2)
                    gripper_raw = gripper_cmd_array[:, 0]
                    gripper_cmd = (gripper_raw > 0.02)  # Boolean array
                else:
                    # Default: assume gripper is open
                    gripper_cmd = np.ones(len(ee_pose), dtype=bool)
            
            # Ensure same length
            min_len = min(len(ee_pose), len(gripper_cmd))
            ee_pose = ee_pose[:min_len]
            gripper_cmd = gripper_cmd[:min_len]
            
            # Convert to list of Action objects
            actions = []
            for i in range(len(ee_pose)):
                # ee_pose is [7] = [pos(3), quat(4)] in wxyz format
                ee_pose_tensor = torch.from_numpy(ee_pose[i]).float().to(self.device)  # [7]
                ee_pose_batch = ee_pose_tensor.unsqueeze(0)  # [1, 7]
                gripper_open_bool = bool(gripper_cmd[i])
                
                action = Action(
                    action_type='ee',
                    ee_pose=ee_pose_batch,
                    gripper_open=gripper_open_bool
                )
                actions.append(action)
            
            return actions
    
    def _load_actions_from_hdf5(self, hdf5_path: str) -> List[Action]:
        """
        Load actions from HDF5 file.
        
        Args:
            hdf5_path: Path to HDF5 file
            
        Returns:
            List of Action objects
        """
        with h5py.File(hdf5_path, 'r') as f:
            # Check if raw format (has /joint_action) or processed format (has /action)
            if '/joint_action' in f:
                # Raw format: load joint_pos and gripper_cmd
                joint_pos = f['/joint_action/joint_pos'][:]  # (episode_len, 7)
                gripper_cmd = f['/joint_action/gripper_cmd'][:]  # (episode_len, 2)
                
                # Convert gripper_cmd to boolean (open/close)
                # gripper_cmd[:, 0] is in [0.00 (close), 0.04 (open)]
                # Threshold: > 0.02 means open
                gripper_raw = gripper_cmd[:, 0]  # Use first gripper
                gripper_open = (gripper_raw > 0.02)  # Boolean array
                
            elif '/action' in f:
                # Processed format: action is already 8D [7 joints, 1 gripper]
                action = f['/action'][:]  # (episode_len, 8)
                joint_pos = action[:, :7]  # (episode_len, 7)
                gripper_normalized = action[:, 7]  # (episode_len,)
                # Convert normalized gripper [0, 1] to boolean
                gripper_open = (gripper_normalized > 0.5)  # Boolean array
                
            else:
                raise ValueError(f"HDF5 file must contain either '/joint_action' or '/action' key")
            
            # Convert to list of Action objects
            actions = []
            for i in range(len(joint_pos)):
                qpos_tensor = torch.from_numpy(joint_pos[i]).float().to(self.device)  # [7]
                qpos_batch = qpos_tensor.unsqueeze(0)  # [1, 7]
                gripper_open_bool = bool(gripper_open[i])
                
                action = Action(
                    action_type='qpos',
                    qpos=qpos_batch,
                    gripper_open=gripper_open_bool
                )
                actions.append(action)
            
            return actions
    
    def get_action(self, observation: Dict[str, Any]) -> Action:
        """
        Get next action from loaded sequence.
        
        Args:
            observation: Observation dict (ignored, actions are pre-loaded)
            
        Returns:
            Next Action in the sequence
        """
        if self.current_step >= len(self.actions):
            # End of episode: return last action
            return self.actions[-1]
        
        action = self.actions[self.current_step]
        self.current_step += 1
        return action
    
    def reset(self) -> None:
        """Reset policy state (reset step counter)."""
        self.current_step = 0
        print(f"DirectReplayPolicy reset: {len(self.actions)} actions ready")


def create_direct_replay_policy(
    hdf5_path: Optional[str] = None,
    demo_folder: Optional[str] = None,
    task_folder: Optional[str] = None,
    h5py_path: Optional[str] = None,
    episode_idx: int = 0,
    env_idx: int = 0,
    observation_keys: List[str] = ['rgb'],
    action_type: str = 'qpos',  # 'qpos' or 'ee'
    image_resolution: List[int] = [224, 224],
    device: str = "cuda:0",
) -> DirectReplayPolicy:
    """
    Factory function to create DirectReplayPolicy.
    
    Args:
        hdf5_path: Path to HDF5 file or directory with episode_*.hdf5 files
        demo_folder: Path to demo folder (e.g., outputs/demo_video/demos) - alternative to hdf5_path
        task_folder: Path to task folder containing config.json (optional)
        h5py_path: Path to h5py directory (optional, alternative to task_folder)
        episode_idx: Episode index if hdf5_path is a directory or demo_folder (default: 0)
        env_idx: Environment index for demo_folder (default: 0)
        observation_keys: List of observation keys
        action_type: 'qpos' or 'ee' - type of actions to replay
        image_resolution: Image resolution
        device: Device to run on
        
    Returns:
        DirectReplayPolicy instance
    """
    return DirectReplayPolicy(
        observation_keys=observation_keys,
        action_type=action_type,
        image_resolution=image_resolution,
        hdf5_path=hdf5_path,
        demo_folder=demo_folder,
        task_folder=task_folder,
        h5py_path=h5py_path,
        episode_idx=episode_idx,
        env_idx=env_idx,
        device=device,
    )


# Test-related imports (only needed for test functions)
# Note: AppLauncher is already initialized above
from envs.cfgs.eval_cfg import EvaluationConfig
from envs.cfgs.task_cfg import load_task_cfg
from envs.sim_wrapper_isaac import PolicyEvaluationWrapper
from envs.make_env_isaac import make_env


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


def test_direct_replay(
    hdf5_path: Optional[str] = None,
    demo_folder: Optional[str] = None,
    task_folder: Optional[str] = None,
    h5py_path: Optional[str] = None,
    episode_idx: int = 0,
    env_idx: int = 0,
    task_json_path: str = "/app/tasks/demo_video/task.json",
    num_envs: int = 1,
    device: str = "cuda:0",
    action_type: str = "qpos",
):
    """
    Test DirectReplayPolicy with PolicyEvaluationWrapper.
    
    Args:
        hdf5_path: Path to HDF5 file or directory with episode_*.hdf5 files
        demo_folder: Path to demo folder (e.g., outputs/demo_video/demos) - alternative to hdf5_path
        task_folder: Path to task folder containing config.json (optional)
        h5py_path: Path to h5py directory (optional, alternative to task_folder)
        episode_idx: Episode index if hdf5_path is a directory or demo_folder (default: 0)
        env_idx: Environment index for demo_folder (default: 0)
        task_json_path: Path to task.json file
        num_envs: Number of parallel environments
        device: Device to run on
    """
    # Load task config
    task_cfg = load_task_cfg(task_json_path)
    # task_cfg = clean_task_cfg(task_cfg)
    
    # Use default evaluation config
    eval_cfg = EvaluationConfig()
    if action_type == "ee" or action_type == "ee_direct":
        eval_cfg.max_steps = 13
    else:
        eval_cfg.max_steps = 1000  # Allow enough steps for replay
    eval_cfg.record_video = True
    eval_cfg.use_randomization = False
    
    # Create environment
    env, _ = make_env(
        task_cfg,
        eval_cfg=eval_cfg,
        num_envs=num_envs,
        device=device,
        bg_simplify=False
    )
    sim, scene = env.sim, env.scene
    my_sim = PolicyEvaluationWrapper(sim, scene, task_cfg, eval_cfg, num_envs)
    
    # Create direct replay policy
    replay_policy = DirectReplayPolicy(
        observation_keys=['rgb'],
        action_type=action_type,
        image_resolution=[224, 224],
        hdf5_path=hdf5_path,
        demo_folder=demo_folder,
        task_folder=task_folder,
        h5py_path=h5py_path,
        episode_idx=episode_idx,
        env_idx=env_idx,
        device=device
    )
    
    # Set policy and evaluate
    my_sim.set_policy(replay_policy)
    my_sim.out_dir = Path("/app/r2sVLA/results")
    
    print(f"Starting direct replay test with {len(replay_policy.actions)} actions")
    result = my_sim.evaluate_episode()
    print(f"Replay result: {result}")
    my_sim.save_data()
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test DirectReplayPolicy")
    parser.add_argument(
        "--hdf5_path",
        type=str,
        default=None,
        help="Path to HDF5 file or directory with episode_*.hdf5 files"
    )
    parser.add_argument(
        "--demo_folder",
        type=str,
        default=None,
        help="Path to demo folder (e.g., outputs/demo_video/demos) - alternative to hdf5_path"
    )
    parser.add_argument(
        "--task_folder",
        type=str,
        default=None,
        help="Path to task folder containing config.json (optional)"
    )
    parser.add_argument(
        "--h5py_path",
        type=str,
        default=None,
        help="Path to h5py directory (optional, alternative to task_folder)"
    )
    parser.add_argument(
        "--episode_idx",
        type=int,
        default=-1,
        help="Episode index if hdf5_path is a directory or demo_folder (default: 0)"
    )
    parser.add_argument(
        "--env_idx",
        type=int,
        default=0,
        help="Environment index for demo_folder (default: 0)"
    )
    parser.add_argument(
        "--task_json_path",
        type=str,
        default="/app/tasks/demo_video/task.json",
        help="Path to task.json file (default: /app/tasks/demo_video/task.json)"
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Number of parallel environments (default: 1)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run on (default: cuda:0)"
    )
    parser.add_argument(
        "--action_type",
        type=str,
        default="qpos",
        choices=["qpos", "ee_direct", "ee_cam", "ee_l"],
        help="Action type: 'qpos' for joint position control, 'ee' for end-effector control (default: qpos)"
    )
    
    args = parser.parse_args()
    
    # Validate that at least one of hdf5_path or demo_folder is provided
    if args.hdf5_path is None and args.demo_folder is None:
        parser.error("Either --hdf5_path or --demo_folder must be provided")
    
    try:
        test_direct_replay(
            hdf5_path=args.hdf5_path,
            demo_folder=args.demo_folder,
            task_folder=args.task_folder,
            h5py_path=args.h5py_path,
            episode_idx=args.episode_idx,
            env_idx=args.env_idx,
            task_json_path=args.task_json_path,
            num_envs=args.num_envs,
            device=args.device,
            action_type=args.action_type
        )
    except Exception as e:
        print(f"[ERR] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        simulation_app.close()
