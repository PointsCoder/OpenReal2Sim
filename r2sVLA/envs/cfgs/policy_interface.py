"""
Policy interface definitions for Isaac Lab simulation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Literal, Union, Optional, List
import torch
import torch.nn.functional as F


class Action:
    """
    Action class supporting both joint position (qpos) and end-effector (ee) control.
    
    Attributes:
        action_type: 'qpos' for joint position control, 'ee' for end-effector control
        qpos: Joint positions [B, n_joints] (for action_type='qpos')
        ee_pose: End-effector pose [B, 7] (pos [3] + quat [4] in wxyz format) (for action_type='ee')
        gripper_open: Gripper state (True = open, False = closed)
    """
    def __init__(
        self,
        action_type: Literal['qpos', 'ee_direct', 'ee_cam', 'ee_l'],
        qpos: Optional[torch.Tensor] = None,
        ee_pose: Optional[torch.Tensor] = None,
        gripper_open: Union[bool, torch.Tensor] = True,
    ):
        self.action_type = action_type
        
        if action_type == 'qpos':
            if qpos is None:
                raise ValueError("qpos must be provided when action_type='qpos'")
            self.qpos = qpos if torch.is_tensor(qpos) else torch.tensor(qpos, dtype=torch.float32)
            self.ee_pose = None
        elif action_type == 'ee_direct' or action_type == 'ee_l' or action_type == 'ee_cam':
            if ee_pose is None:
                raise ValueError("ee_pose must be provided when action_type='ee'")
            self.ee_pose = ee_pose if torch.is_tensor(ee_pose) else torch.tensor(ee_pose, dtype=torch.float32)
            self.qpos = None
        else:
            raise ValueError(f"Invalid action_type: {action_type}. Must be 'qpos' or 'ee'")
        
        # Handle gripper_open: can be bool or tensor
        if isinstance(gripper_open, bool):
            self.gripper_open = gripper_open
        elif torch.is_tensor(gripper_open):
            self.gripper_open = gripper_open
        else:
            self.gripper_open = torch.tensor(gripper_open, dtype=torch.bool)
    
    def __repr__(self):
        if self.action_type == 'qpos':
            return f"Action(type='qpos', qpos_shape={self.qpos.shape}, gripper_open={self.gripper_open})"
        else:
            return f"Action(type='ee', ee_pose_shape={self.ee_pose.shape}, gripper_open={self.gripper_open})"


from abc import ABC, abstractmethod
from typing import List

class BasePolicy(ABC):
    """
    Abstract base class for policies.
    All policies should inherit from this class and implement the required methods.

    Required attributes:
        - observation_keys: List[str], specifies what observation fields the policy needs
        - action_type: str, 'qpos' or 'ee'
        - image_resolution: List[int], e.g., [128, 128]
    """

    # Should be set by the derived class (not enforced at construction)
    observation_keys: List[str]
    action_type: str
    image_resolution: List[int]

    @abstractmethod
    def __init__(self, observation_keys: List[str], action_type: str, image_resolution: List[int]):
        self.observation_keys = observation_keys
        self.action_type = action_type
        self.image_resolution = image_resolution
    
    @abstractmethod
    def get_action(self, observation: Dict[str, Any]) -> Union[Action, List[Action]]:
        """
        Get action from policy given observation.
        
        Args:
            observation: Dictionary containing observation data (images, states, etc.)
        
        Returns:
            Either a single Action object or a list of Action objects (action sequence/chunk).
            If returning a list, each Action will be executed sequentially in the environment.
            This allows policies to return action chunks (like ACT, RDT, etc.) for better temporal consistency.
        """
        

    @abstractmethod
    def reset(self) -> None:
        """
        Reset policy state (e.g., clear observation history, reset temporal aggregation).
        Called at the beginning of each episode.
        """
        pass

    def update_obs(self, observation: Dict[str, Any]) -> None:
        """
        Update policy's internal observation buffer (optional).
        Useful for policies that maintain observation history.
        
        Args:
            observation: New observation to add to history
        """
        pass
    
    def preprocess_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess observation before passing to policy.
        Default implementation resizes images to policy's required resolution.
        Only resizes keys that are in observation_keys and are image tensors.
        
        Args:
            observation: Raw observation dictionary
        
        Returns:
            Preprocessed observation dictionary with images resized to image_resolution
        """
        processed_obs = {}
        
        # Image keys that typically need resizing (can be extended)
        image_keys = ['rgb', 'composed_rgb', 'depth', 'mask', 'fg_mask', 'obj_mask']
        
        for key, value in observation.items():
            # Only process keys that are in observation_keys and are image-like
            if key in self.observation_keys and key in image_keys and torch.is_tensor(value):
                # Check if this is an image tensor (4D: [B, H, W, C] or [B, C, H, W])
                if value.ndim == 4:
                    # Determine format and resize
                    # Case 1: [B, H, W, C] format (channels last)
                    if value.shape[-1] in [1, 3, 4]:
                        B, H, W, C = value.shape
                        # Convert to [B, C, H, W] for F.interpolate
                        value = value.permute(0, 3, 1, 2)
                        # Convert to float if needed (F.interpolate requires float)
                        if value.dtype != torch.float32:
                            value = value.float()
                            # Normalize if values are in [0, 255] range
                            if value.max() > 1.0:
                                value = value / 255.0
                        # Resize
                        mode = 'bilinear' if C in [3, 4] else 'nearest'
                        value = F.interpolate(
                            value,
                            size=self.image_resolution,
                            mode=mode,
                            align_corners=False if mode == 'bilinear' else None
                        )
                        # Convert back to [B, H, W, C]
                        value = value.permute(0, 2, 3, 1)
                    
                    # Case 2: [B, C, H, W] format (channels first)
                    elif value.shape[1] in [1, 3, 4]:
                        # Convert to float if needed (F.interpolate requires float)
                        if value.dtype != torch.float32:
                            value = value.float()
                            # Normalize if values are in [0, 255] range
                            if value.max() > 1.0:
                                value = value / 255.0
                        # Resize directly
                        mode = 'bilinear' if value.shape[1] in [3, 4] else 'nearest'
                        value = F.interpolate(
                            value,
                            size=self.image_resolution,
                            mode=mode,
                            align_corners=False if mode == 'bilinear' else None
                        )
                    
                    processed_obs[key] = value
                else:
                    # Not a 4D tensor, keep as is
                    processed_obs[key] = value
            else:
                # Not in observation_keys or not an image key, keep as is
                processed_obs[key] = value
        
        return processed_obs
