import numpy as np
import transforms3d


def robot_placement_candidates(
    scene_info: dict,
    reachability=(0.3, 0.65),
    reachability_zcenter_offset=0.3,
    robot_aabb=(0.12, 0.12, 0.72),
    num_radius_steps=10,
    num_angle_steps=36,
    num_z_steps=5,
    occlusion_clearance=(0.02, 0.02, 0.02),
    obj_world_radius=0.02,
    num_obj_rays=8,
    min_cam_base_separation_deg=45.0,
):
    """
    Return robot base poses (position + wxyz rotation) satisfying:
      (1) Reachability
      (2) No overlap with background AABB
      (3) Same half-space as camera w.r.t object & min separation angle
      (4) No occlusion from camera to object (center ray + ring rays)

    Args:
        scene_info: Dict with keys: object_center, scene_min, scene_max, move_to (camera position)
        reachability: (min, max) radial distance from object center (meters)
        reachability_zcenter_offset: Z offset for reachability calculation
        robot_aabb: (width, depth, height) of robot footprint
        num_radius_steps: Number of radial positions to test
        num_angle_steps: Number of angular positions to test
        num_z_steps: Number of Z heights to test
        occlusion_clearance: Extra clearance for occlusion checks
        obj_world_radius: Radius of ring around object center for occlusion checks
        num_obj_rays: Number of rays from ring points
        min_cam_base_separation_deg: Minimum angle between camera and robot (degrees)

    Returns:
        List of candidate poses, each with "position" (xyz) and "rotation" (wxyz quaternion)
    """
    object_center = np.array(scene_info["object_center"], dtype=float)
    scene_min = np.array(scene_info["scene_min"], dtype=float)
    scene_max = np.array(scene_info["scene_max"], dtype=float)
    cam_pos = np.array(scene_info["move_to"], dtype=float)

    def segment_intersects_aabb(p0, p1, aabb_min, aabb_max, eps=1e-9):
        """Check if line segment [p0, p1] intersects AABB."""
        d = p1 - p0
        t0, t1 = 0.0, 1.0
        for i in range(3):
            if abs(d[i]) < eps:
                if p0[i] < aabb_min[i] or p0[i] > aabb_max[i]:
                    return False
            else:
                inv_d = 1.0 / d[i]
                t_near = (aabb_min[i] - p0[i]) * inv_d
                t_far = (aabb_max[i] - p0[i]) * inv_d
                if t_near > t_far:
                    t_near, t_far = t_far, t_near
                t0 = max(t0, t_near)
                t1 = min(t1, t_far)
                if t0 > t1:
                    return False
        return True

    def edge_ring_points(obj_center, cam_pos, radius, m):
        """Generate ring of points around object for occlusion testing."""
        if m <= 0 or radius <= 0.0:
            return np.empty((0, 3), dtype=float)
        view = obj_center - cam_pos
        n = np.linalg.norm(view)
        if n < 1e-9:
            return np.empty((0, 3), dtype=float)
        u = view / n
        tmp = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(np.dot(tmp, u)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0], dtype=float)
        e1 = np.cross(u, tmp); e1 /= (np.linalg.norm(e1) + 1e-12)
        e2 = np.cross(u, e1);  e2 /= (np.linalg.norm(e2) + 1e-12)
        phis = np.linspace(0.0, 2.0 * np.pi, num=m, endpoint=False)
        ring = [obj_center + radius * (np.cos(phi) * e1 + np.sin(phi) * e2) for phi in phis]
        return np.asarray(ring, dtype=float)

    candidates = []
    radius_values = np.linspace(reachability[0], reachability[1], num=num_radius_steps)
    angle_values = np.linspace(0, 2 * np.pi, num=num_angle_steps, endpoint=False)
    z_min, z_max = 0.0, 2.0 * float(object_center[2])
    z_values = np.linspace(z_min, z_max, num=num_z_steps)
    ring_pts = edge_ring_points(object_center, cam_pos, obj_world_radius, num_obj_rays)

    cos_min_sep = np.cos(np.deg2rad(float(min_cam_base_separation_deg)))

    for z in z_values:
        reach_center_z = z + reachability_zcenter_offset
        for r in radius_values:
            for theta in angle_values:
                dx = r * np.cos(theta); dy = r * np.sin(theta)
                base_pos = np.array([object_center[0] - dx, object_center[1] - dy, z], dtype=float)

                # (1) reachability
                reach_dist = np.linalg.norm(object_center - np.array([base_pos[0], base_pos[1], reach_center_z]))
                if reach_dist > reachability[1] or reach_dist < reachability[0]:
                    continue

                # (2) no overlap with background AABB
                half = 0.5 * np.array(robot_aabb, dtype=float)
                base_min = base_pos - half
                base_max = base_pos + half
                if np.all(base_max > scene_min) and np.all(base_min < scene_max):
                    continue

                # (3) same-side + min angle
                v_cam = cam_pos - object_center
                v_base = base_pos - object_center
                n1 = np.linalg.norm(v_cam); n2 = np.linalg.norm(v_base)
                if n1 < 1e-9 or n2 < 1e-9:
                    continue
                cos_val = float(np.dot(v_cam, v_base) / (n1 * n2))
                if (cos_val < 0.0) or (cos_val > cos_min_sep):
                    continue

                # (4a) no occlusion on center ray
                extra = np.array(occlusion_clearance, dtype=float)
                occ_min = base_min - extra
                occ_max = base_max + extra
                if segment_intersects_aabb(cam_pos, object_center, occ_min, occ_max):
                    continue
                # (4b) ring rays
                if any(segment_intersects_aabb(cam_pos, P, occ_min, occ_max) for P in ring_pts):
                    continue

                # (5) yaw towards object in XY plane
                facing_vec = object_center[:2] - base_pos[:2]
                yaw = np.arctan2(facing_vec[1], facing_vec[0])
                quat_wxyz = transforms3d.euler.euler2quat(0.0, 0.0, yaw)

                candidates.append({
                    "position": base_pos.tolist(),
                    "rotation": quat_wxyz.tolist(),
                })

    return candidates


def compute_robot_pose_from_scene(scene_config: dict, extrinsics: np.ndarray = None):
    """
    Compute robot pose from scene configuration.

    Args:
        scene_config: Scene JSON dict
        extrinsics: Optional 4x4 transform matrix from camera to robot base

    Returns:
        robot_pose: List of [x, y, z, qw, qx, qy, qz]
    """
    # If extrinsics provided, use calibration
    if extrinsics is not None:
        return calibration_to_robot_pose(scene_config, extrinsics)

    # Otherwise, use heuristic placement
    # Build scene_info dict
    camera = scene_config.get("camera", {})
    aabb = scene_config.get("aabb", {})
    objects = scene_config.get("objects", {})

    # Get first object as reference
    if not objects:
        raise ValueError("No objects in scene for robot placement")

    first_obj = list(objects.values())[0]

    scene_info = {
        "move_to": camera.get("camera_position", [0, 0, 1]),
        "scene_min": aabb.get("scene_min", [-0.5, -0.5, -0.5]),
        "scene_max": aabb.get("scene_max", [0.5, 0.5, 0.5]),
        "object_center": first_obj.get("object_center", [0, 0, 0]),
    }

    candidates = robot_placement_candidates(scene_info)

    if not candidates:
        raise RuntimeError("No valid robot placements found. Try adjusting scene or robot parameters.")

    # Return first candidate
    candidate = candidates[0]
    robot_pose = candidate["position"] + candidate["rotation"]

    return robot_pose


def calibration_to_robot_pose(scene_config: dict, T_camopencv_to_robotbase: np.ndarray):
    """
    Compute robot pose from calibrated extrinsics.

    Args:
        scene_config: Scene JSON dict
        T_camopencv_to_robotbase: 4x4 transform from camera (OpenCV) to robot base

    Returns:
        robot_pose: List of [x, y, z, qw, qx, qy, qz]
    """
    def _as_homo(T):
        A = np.asarray(T, dtype=np.float64)
        if A.shape == (4, 4):
            return A
        if A.shape == (3, 4):
            M = np.eye(4, dtype=np.float64); M[:3, :4] = A; return M
        if A.shape == (3, 3):
            M = np.eye(4, dtype=np.float64); M[:3, :3] = A; return M
        raise ValueError(f"Unsupported shape: {A.shape}")

    def _quat_wxyz(R):
        q = transforms3d.quaternions.mat2quat(R)  # (w,x,y,z)
        return -q if q[0] < 0 else q

    T_camopencv_to_world = _as_homo(scene_config["camera"]["camera_opencv_to_world"])
    T_camopencv_to_robotbase = _as_homo(T_camopencv_to_robotbase)
    T_robotbase_to_world = T_camopencv_to_world @ np.linalg.inv(T_camopencv_to_robotbase)

    pos_w = T_robotbase_to_world[:3, 3].copy()
    quat_w = _quat_wxyz(T_robotbase_to_world[:3, :3])

    robot_pose = list(pos_w) + list(quat_w)
    return robot_pose
