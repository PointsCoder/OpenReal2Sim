import numpy as np
from mani_skill.utils.structs.pose import Pose
import transforms3d
from transforms3d.quaternions import mat2quat


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


def get_top_n_grasp_poses(
    grasps: np.ndarray,
    n: int,
    direction_hint: np.ndarray | None = None,
    position_hint: np.ndarray | None = None,
) -> list[np.ndarray]:
    """
    Extracts the top n grasps, optionally re-scoring them based on hints.

    The re-scoring logic is adapted from the Isaac Lab implementation, combining:
    - Direction alignment: Prefers grasps whose approach vector (+X axis)
      matches the `direction_hint`.
    - Position proximity: Prefers grasps whose center is close to the
      `position_hint`.
    - Original network score.

    Args:
        grasps: The raw grasp data from the .npy file.
        n: The number of top grasps to return.
        direction_hint: A (3,) numpy array specifying the desired approach direction.
        position_hint: A (3,) numpy array specifying the desired grasp position.

    Returns:
        A list of 4x4 transformation matrices for the top n grasps,
        sorted by the new score.
    """
    if grasps is None or len(grasps) == 0:
        return []

    # --- Re-scoring based on hints ---
    w_dir, w_pos, w_net = 1.0, 1.0, 0.5  # Weights for combining scores
    if direction_hint is None:
        w_dir = 0
    if position_hint is None:
        w_pos = 0

    net_scores = grasps[:, 0]
    translations = grasps[:, 13:16]
    rotations = grasps[:, 4:13].reshape(-1, 3, 3)
    approach_dirs = rotations[:, :, 0]  # GraspNet's +X is the approach direction

    # Normalize network score
    net_term = (
        (net_scores - net_scores.min()) / (net_scores.max() - net_scores.min())
        if net_scores.max() > net_scores.min()
        else np.zeros_like(net_scores)
    )

    # Direction alignment term
    dir_term = np.zeros_like(net_scores)
    if w_dir > 0 and direction_hint is not None:
        hint_norm = np.linalg.norm(direction_hint)
        if hint_norm > 1e-6:
            normalized_hint = direction_hint / hint_norm
            cos_sim = np.clip(np.dot(approach_dirs, normalized_hint), -1.0, 1.0)
            dir_term = 0.5 * (cos_sim + 1.0)  # Map from [-1, 1] to [0, 1]

    # Position proximity term (using Radial Basis Function)
    pos_term = np.zeros_like(net_scores)
    if w_pos > 0 and position_hint is not None:
        sigma = 0.05  # Effective radius in meters
        dists_sq = np.sum((translations - position_hint) ** 2, axis=1)
        pos_term = np.exp(-0.5 * dists_sq / (sigma**2))

    # Combine scores and sort
    total_scores = w_dir * dir_term + w_pos * pos_term + w_net * net_term
    sorted_indices = np.argsort(total_scores)[::-1]
    top_n_indices = sorted_indices[:n]

    # --- Construct pose matrices for the top n grasps ---
    pose_matrices = []
    for i in top_n_indices:
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = rotations[i]
        pose_matrix[:3, 3] = translations[i]
        pose_matrices.append(pose_matrix)

    return pose_matrices


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
