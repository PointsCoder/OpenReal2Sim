# transform_utils.py
import torch
import numpy as np
import transforms3d

def grasp_to_world(grasp) -> tuple[np.ndarray, np.ndarray]:
    """
    GraspNet -> EE:
      X_grasp=approach, Y_grasp=width, Z_grasp=thickness
      We want: Z_EE=approach, Y_EE=-width, X_EE=thickness
    """
    Rg = grasp.rotation_matrix.astype(np.float32)
    delta = np.array([[0, 0, 1],
                      [0, -1, 0],
                      [1, 0, 0]], dtype=np.float32)
    Ree = Rg @ delta
    q   = transforms3d.quaternions.mat2quat(Ree).astype(np.float32)  # wxyz
    p   = grasp.translation.astype(np.float32)
    return p, q

def grasp_approach_axis_batch(qw_batch: np.ndarray) -> np.ndarray:
    """
    get the approach axis (Z axis) of each env in world frame
    qw_batch: (B,4) quaternion (wxyz)
    return: (B,3) each env's grasping approach axis (Z axis) in world frame
    """
    qw_batch = np.asarray(qw_batch, dtype=np.float32)
    a_list = []
    for q in qw_batch:
        R = transforms3d.quaternions.quat2mat(q)
        a = R[:, 2]
        a_list.append((a / (np.linalg.norm(a) + 1e-8)).astype(np.float32))
    return np.stack(a_list, axis=0)

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
