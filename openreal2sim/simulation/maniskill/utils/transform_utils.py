# -*- coding: utf-8 -*-
"""
Transform utilities for converting between OpenCV and SAPIEN coordinate systems.

Coordinate Systems:
- OpenCV (cv): Z forward, X right, Y down
- SAPIEN/ROS (ros): X forward, Y left, Z up

Reference: Based on camera pose conversion utilities from reconstruction pipeline.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple
from mani_skill.utils.structs.pose import Pose
import transforms3d


def convert_camera_pose(pose, in_type, out_type):
    """
    Convert camera pose between different coordinate conventions.

    Args:
        pose: 4x4 pose matrix
        in_type: Input convention ("cv", "gl", "blender", "ros")
        out_type: Output convention ("cv", "gl", "blender", "ros")

    Returns:
        4x4 pose matrix in output convention
    """
    accepted_types = ["cv", "gl", "blender", "ros"]

    assert in_type in accepted_types, f"Input type {in_type} not in {accepted_types}"
    assert out_type in accepted_types, f"Output type {out_type} not in {accepted_types}"

    if in_type == "blender":
        in_type = "gl"
    if out_type == "blender":
        out_type = "gl"

    if in_type == out_type:
        return pose

    if (in_type == "cv" and out_type == "gl") or (in_type == "gl" and out_type == "cv"):
        return pose @ np.array(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ]
        )
    elif in_type == "ros" and out_type == "cv":
        return pose @ np.array(
            [
                [0, 0, 1, 0],
                [-1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 0, 1],
            ]
        )
    elif in_type == "cv" and out_type == "ros":
        # Inverse of ros -> cv
        return pose @ np.array(
            [
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
            ]
        )
    else:
        raise NotImplementedError(
            f"Conversion from {in_type} -> {out_type} is not implemented!"
        )


def qvec2rotmat(qvec):
    """Convert quaternion (wxyz) to rotation matrix."""
    qvec = qvec / np.linalg.norm(qvec)
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def rotmat2qvec(R):
    """Convert rotation matrix to quaternion (wxyz)."""
    qvec = np.empty(4)
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qvec[0] = 0.25 / s
        qvec[1] = (R[2, 1] - R[1, 2]) * s
        qvec[2] = (R[0, 2] - R[2, 0]) * s
        qvec[3] = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qvec[0] = (R[2, 1] - R[1, 2]) / s
            qvec[1] = 0.25 * s
            qvec[2] = (R[0, 1] + R[1, 0]) / s
            qvec[3] = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qvec[0] = (R[0, 2] - R[2, 0]) / s
            qvec[1] = (R[0, 1] + R[1, 0]) / s
            qvec[2] = 0.25 * s
            qvec[3] = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qvec[0] = (R[1, 0] - R[0, 1]) / s
            qvec[1] = (R[0, 2] + R[2, 0]) / s
            qvec[2] = (R[1, 2] + R[2, 1]) / s
            qvec[3] = 0.25 * s

    qvec = qvec / np.linalg.norm(qvec)
    return qvec


def opencv_to_sapien_pose(opencv_extrinsic: np.ndarray) -> Pose:
    """
    Convert OpenCV camera extrinsic matrix to SAPIEN Pose.

    Args:
        opencv_extrinsic: 4x4 extrinsic matrix (camera to world in OpenCV convention)

    Returns:
        Pose: SAPIEN pose object in ROS convention
    """
    # Convert from OpenCV (cv) to ROS (sapien) convention
    sapien_pose = convert_camera_pose(opencv_extrinsic, in_type="cv", out_type="ros")

    # Extract position and rotation
    position = sapien_pose[:3, 3]
    rotation_matrix = sapien_pose[:3, :3]

    # Convert rotation matrix to quaternion (wxyz format for ManiSkill)
    quat_wxyz = rotmat2qvec(rotation_matrix)

    return Pose.create_from_pq(p=position, q=quat_wxyz)


def intrinsic_to_fov(
    fx: float, width: int, fy: float = None, height: int = None
) -> float:
    """
    Convert camera intrinsic parameters to field of view.

    Args:
        fx: Focal length in x direction
        width: Image width in pixels
        fy: Focal length in y direction (optional, uses fx if not provided)
        height: Image height in pixels (optional)

    Returns:
        float: Horizontal field of view in radians
    """
    # Horizontal FOV from fx and width
    fov_x = 2 * np.arctan(width / (2 * fx))

    if fy is not None and height is not None:
        # Vertical FOV from fy and height
        fov_y = 2 * np.arctan(height / (2 * fy))
        # Return the average (or you could return both)
        return (fov_x + fov_y) / 2

    return fov_x


def world_pose_to_robot_relative(world_pose: Pose, robot_pose: Pose) -> Pose:
    """
    Convert a world-frame pose to robot-base-relative pose.

    In ManiSkill, cameras mounted on robots need poses relative to the robot base.

    Args:
        world_pose: Pose in world frame
        robot_pose: Robot base pose in world frame

    Returns:
        Pose: Camera pose relative to robot base
    """
    # Get world pose components
    if hasattr(world_pose, "p") and hasattr(world_pose, "q"):
        world_p = (
            world_pose.p
            if isinstance(world_pose.p, np.ndarray)
            else world_pose.p.cpu().numpy()
        )
        world_q = (
            world_pose.q
            if isinstance(world_pose.q, np.ndarray)
            else world_pose.q.cpu().numpy()
        )
    else:
        raise ValueError("Invalid world_pose format")

    # Get robot pose components
    if hasattr(robot_pose, "p") and hasattr(robot_pose, "q"):
        robot_p = (
            robot_pose.p
            if isinstance(robot_pose.p, np.ndarray)
            else robot_pose.p.cpu().numpy()
        )
        robot_q = (
            robot_pose.q
            if isinstance(robot_pose.q, np.ndarray)
            else robot_pose.q.cpu().numpy()
        )
    else:
        raise ValueError("Invalid robot_pose format")

    # Convert quaternions to transformation matrices
    # Note: world_q and robot_q are in xyzw format from Pose
    # Convert to wxyz for qvec2rotmat
    world_q_wxyz = np.array([world_q[3], world_q[0], world_q[1], world_q[2]])
    robot_q_wxyz = np.array([robot_q[3], robot_q[0], robot_q[1], robot_q[2]])

    # Build transformation matrices
    T_world = np.eye(4)
    T_world[:3, :3] = qvec2rotmat(world_q_wxyz)
    T_world[:3, 3] = world_p

    T_robot = np.eye(4)
    T_robot[:3, :3] = qvec2rotmat(robot_q_wxyz)
    T_robot[:3, 3] = robot_p

    # Compute relative transform: T_cam_in_robot = T_robot^-1 @ T_world
    T_robot_inv = np.linalg.inv(T_robot)
    T_relative = T_robot_inv @ T_world

    # Extract relative pose
    relative_p = T_relative[:3, 3]
    relative_q_wxyz = rotmat2qvec(T_relative[:3, :3])
    relative_q_xyzw = np.array(
        [relative_q_wxyz[1], relative_q_wxyz[2], relative_q_wxyz[3], relative_q_wxyz[0]]
    )

    return Pose.create_from_pq(p=relative_p, q=relative_q_xyzw)


def qt2pose(q, t):
    """Convert quaternion (wxyz) and translation to 4x4 pose matrix."""
    T = np.eye(4)
    T[:3, :3] = qvec2rotmat(q)
    T[:3, 3] = t
    return T


def pose2qt(pose):
    """Convert 4x4 pose matrix to quaternion (wxyz) and translation."""
    q = rotmat2qvec(pose[:3, :3])
    t = pose[:3, 3]
    return q, t


def rotx_np(angle_deg):
    """Get 3x3 rotation matrix around x-axis."""
    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_a, -sin_a],
            [0.0, sin_a, cos_a],
        ]
    )


def roty_np(angle_deg):
    """Get 3x3 rotation matrix around y-axis."""
    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    return np.array(
        [
            [cos_a, 0.0, sin_a],
            [0.0, 1.0, 0.0],
            [-sin_a, 0.0, cos_a],
        ]
    )


def rotz_np(angle_deg):
    """Get 3x3 rotation matrix around z-axis."""
    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    return np.array(
        [
            [cos_a, -sin_a, 0.0],
            [sin_a, cos_a, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def extract_initial_poses(scene_data: dict) -> Tuple[list, np.ndarray]:
    """
    Extract initial object poses from scene data.

    Args:
        scene_data: Dictionary loaded from scene.json

    Returns:
        Tuple of (list of object names, array of poses [N, 7] in xyzwxyz format)
    """
    objects = scene_data.get("objects", {})

    object_names = []
    object_poses = []

    for obj_info in objects.values():
        object_names.append(obj_info["name"])

        # Use object center as position
        center = obj_info.get("object_center", [0, 0, 0])

        # Default to identity rotation (can be refined later)
        quat_wxyz = [1, 0, 0, 0]

        # Pose format: [x, y, z, qw, qx, qy, qz]
        pose = center + quat_wxyz
        object_poses.append(pose)

    return object_names, np.array(object_poses, dtype=np.float32)
