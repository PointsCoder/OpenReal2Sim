import json
from pathlib import Path
import yaml
import torch
import random
import numpy as np
import transforms3d
import numpy as np
import transforms3d

def pose_to_mat(pos, quat):
    if torch.is_tensor(pos):  pos  = pos.cpu().numpy()
    if torch.is_tensor(quat): quat = quat.cpu().numpy()
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = transforms3d.quaternions.quat2mat(quat)
    m[:3,  3] = pos
    return m

def mat_to_pose(mat):
    pos  = mat[:3, 3]
    quat = transforms3d.quaternions.mat2quat(mat[:3, :3])
    return pos, quat

def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q, axis=1, keepdims=True) + 1e-9
    return q / norm


def pose_distance(T1, T2):
    """
    Compute translation and rotation (angle) distance between two SE(3) transformation matrices in torch.
    Args:
        T1, T2: [..., 4, 4] torch.Tensor or np.ndarray, can be batched
    Returns:
        trans_dist: translation distance(s)
        angle: rotational angle(s)
    """
    if not torch.is_tensor(T1):
        T1 = torch.tensor(T1, dtype=torch.float32)
    if not torch.is_tensor(T2):
        T2 = torch.tensor(T2, dtype=torch.float32)
    
    # Translation distance
    t1 = T1[..., :3, 3]
    t2 = T2[..., :3, 3]
    trans_dist = torch.norm(t2 - t1, dim=-1)

    # Rotation difference (angle)
    R1 = T1[..., :3, :3]
    R2 = T2[..., :3, :3]
    dR = torch.matmul(R2, R1.transpose(-2, -1))
    trace = dR[..., 0, 0] + dR[..., 1, 1] + dR[..., 2, 2]
    cos_angle = (trace - 1) / 2
    
    # Handle numerical precision
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    angle = torch.acos(cos_angle)
    return trans_dist, angle

 