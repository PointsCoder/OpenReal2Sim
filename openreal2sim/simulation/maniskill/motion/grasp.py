import numpy as np
from mani_skill.utils.structs.pose import Pose
import transforms3d


def load_grasps_from_path(filepath: str) -> np.ndarray:
    """Loads grasp data from a .npy file."""
    if not filepath or not filepath.endswith(".npy"):
        raise ValueError("A valid .npy file path must be provided.")
    try:
        return np.load(filepath, allow_pickle=True)
    except FileNotFoundError:
        print(f"Error: Grasp file not found at {filepath}")
        return None


def get_best_grasp_pose(grasps: np.ndarray) -> np.ndarray:
    """
    Extracts the grasp with the highest score.

    The grasp data format is based on the GraspNet API, where:
    - Index 0: Score
    - Index 4-13: Rotation matrix (9 elements)
    - Index 13-16: Translation vector (3 elements)

    Returns:
        A 4x4 transformation matrix of the best grasp in the object's local frame.
    """
    if grasps is None or len(grasps) == 0:
        return None

    best_grasp = grasps[np.argmax(grasps[:, 0])]

    translation = best_grasp[13:16]
    rotation = best_grasp[4:13].reshape(3, 3)

    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = rotation
    pose_matrix[:3, 3] = translation

    return pose_matrix


def transform_grasp_pose(
    object_world_pose: Pose, object_local_grasp_pose: np.ndarray
) -> Pose:
    """
    Transforms the grasp pose from the object's local frame to the world frame,
    and applies a correction to match the robot's TCP frame convention.
    """
    # Correction matrix to align GraspNet's frame with ManiSkill's TCP frame
    # This is derived from the reference SAPIEN script and common robotics conversions.
    # It swaps axes to match (X-approach, Y-width) with (X-forward, Y-left).
    # GraspNet_X -> TCP_Z, GraspNet_Y -> TCP_-Y, GraspNet_Z -> TCP_X
    correction_rotation = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
    correction_matrix = np.eye(4)
    correction_matrix[:3, :3] = correction_rotation

    # Apply the correction to the local grasp pose
    corrected_local_grasp_pose = object_local_grasp_pose @ correction_matrix

    # Transform to world frame
    object_world_matrix = object_world_pose.to_transformation_matrix()
    grasp_world_matrix = object_world_matrix @ corrected_local_grasp_pose

    # get position and quaternion

    import torch

    def to_np(mat):
        if torch.is_tensor(mat):
            mat = mat.detach().cpu().double().numpy()
        return np.asarray(mat, dtype=np.float64)

    grasp_world_matrix = to_np(grasp_world_matrix)
    if grasp_world_matrix.ndim == 3:  # e.g., (B,4,4)
        grasp_world_matrix = grasp_world_matrix[0]
    assert grasp_world_matrix.shape == (4, 4), (
        f"Expected (4,4), got {grasp_world_matrix.shape}"
    )

    # 4) Extract pose
    R = grasp_world_matrix[:3, :3]
    t = grasp_world_matrix[:3, 3]

    # transforms3d returns (w, x, y, z)
    q_wxyz = transforms3d.quaternions.mat2quat(R)
    return Pose.create_from_pq(p=t, q=q_wxyz)
