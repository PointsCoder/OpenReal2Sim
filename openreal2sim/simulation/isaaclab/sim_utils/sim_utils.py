import json
from pathlib import Path
import yaml
import torch
import random
import numpy as np
import transforms3d

from sim_utils.calibration_utils import calibration_to_robot_pose, load_extrinsics

def load_sim_parameters(basedir, key) -> dict:
    scene_json_path = Path(basedir) / "outputs" / key / "simulation" / "scene.json"
    scene_json   = json.load(open(scene_json_path, "r"))
    exp_config = yaml.load(open(Path(basedir) / "config/config.yaml"), Loader=yaml.FullLoader)
    exp_config = exp_config["local"][key]["simulation"]

    # cam configs
    cam_cfg = {
        "width":  int(scene_json["camera"]["width"]),
        "height": int(scene_json["camera"]["height"]),
        "fx": float(scene_json["camera"]["fx"]),
        "fy": float(scene_json["camera"]["fy"]),
        "cx": float(scene_json["camera"]["cx"]),
        "cy": float(scene_json["camera"]["cy"]),
        "cam_orientation": tuple(scene_json["camera"]["camera_heading_wxyz"]),
        "scene_info": {
            "move_to":      list(scene_json["camera"]["camera_position"]),
            "scene_min":    list(scene_json["aabb"]["scene_min"]),
            "scene_max":    list(scene_json["aabb"]["scene_max"]),
            "object_center":list(scene_json["objects"]["1"]["object_center"]),
        },
    }

    # robot configs
    E = load_extrinsics(exp_config, key)
    if E is None:
        robot_cfg = random.choice(robot_placement_candidates_v2(cam_cfg["scene_info"]))
        pos_w, quat_w = robot_cfg["position"], robot_cfg["rotation"]
    else:
        pos_w, quat_w = calibration_to_robot_pose(scene_json, E)
    robot_pose = list(pos_w) + list(quat_w)
    robot_cfg = {
        "robot_pose": robot_pose,
    }

    # demo configs
   
    goal_offset = exp_config.get("goal_offset", 0)
    grasp_idx = exp_config.get("grasp_idx", -1)
    grasp_pre = exp_config.get("grasp_pre", None)
    grasp_delta = exp_config.get("grasp_delta", None)
    manip_object_id = scene_json["manipulated_oid"]
    traj_path = scene_json["objects"][manip_object_id]["final_trajs"]
    grasp_path = scene_json["objects"][manip_object_id].get("grasps", None)
    final_gripper_closed = scene_json["final_gripper_closed"]
    demo_cfg = {
        "manip_object_id": manip_object_id,
        "traj_path": traj_path,
        "goal_offset": goal_offset,
        "grasp_idx": grasp_idx,
        "grasp_pre": grasp_pre,
        "grasp_delta": grasp_delta,
        "grasp_path": grasp_path,
        "final_gripper_closed": final_gripper_closed,
    }

    # physics configs
    physics_key = exp_config.get("physics", "default")
    physics_cfg = {}
    if physics_key == "default":
        physics_cfg["bg_physics"] = default_bg_physics
        physics_cfg["obj_physics"] = default_obj_physics
    elif physics_key == "viz":
        physics_cfg["bg_physics"] = default_bg_physics
        physics_cfg["obj_physics"] = viz_obj_physics
    else:
        print(f"[WARN] Unrecognized physics key '{physics_key}', no physics specified in exp configs.")
        physics_cfg["bg_physics"] = None
        physics_cfg["obj_physics"] = None

    return {
        "key": key,
        "scene_cfg": scene_json,
        "exp_cfg": exp_config,
        "cam_cfg": cam_cfg,
        "robot_cfg": robot_cfg,
        "demo_cfg": demo_cfg,
        "physics_cfg": physics_cfg,
    }


# Default physics for background and objects
default_bg_physics = {
    "mass_props": {"mass": 100.0},
    "rigid_props": {"disable_gravity": True, "kinematic_enabled": True},
    "collision_props": {"collision_enabled": True,},
}

default_obj_physics = {
    "mass_props": {"mass": 0.2},
    "rigid_props": {"disable_gravity": False, "kinematic_enabled": False},
    "collision_props": {"collision_enabled": True,},
}

# For viz
viz_obj_physics = {
    "mass_props": {"mass": 0.5},
    "rigid_props": {"disable_gravity": True, "kinematic_enabled": False},
    "collision_props": {"collision_enabled": False,},
}

def robot_placement_candidates_v2(
    scene_info: dict,
    reachability=(0.3, 0.85),
    reachability_zcenter_offset=0.3,
    robot_aabb=(0.12, 0.12, 0.72),
    num_radius_steps=10,
    num_angle_steps=36,
    num_z_steps=5,
    occlusion_clearance=(0.02, 0.02, 0.02),  # inflated base AABB for occlusion checks (m)
    obj_world_radius=0.02,                   # small ring near object center (m)
    num_obj_rays=8,                          # # of rays from ring points (>=8)
    min_cam_base_separation_deg=45.0,        # minimum separation angle wrt camera direction
):
    """
    Return robot base poses (position + wxyz rotation) satisfying:
      (1) Reachability
      (2) No overlap with background AABB
      (3) Same half-space as camera w.r.t object & min separation angle
      (4) No occlusion from camera to object (center ray + ring rays)
    """
    object_center = np.array(scene_info["object_center"], dtype=float)
    scene_min = np.array(scene_info["scene_min"], dtype=float)
    scene_max = np.array(scene_info["scene_max"], dtype=float)
    cam_pos = np.array(scene_info["move_to"], dtype=float)

    def segment_intersects_aabb(p0, p1, aabb_min, aabb_max, eps=1e-9):
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
                    "rotation": [float(x) for x in quat_wxyz],  # (w,x,y,z)
                    "yaw_deg": float(np.degrees(yaw)),
                    "base_z": float(z),
                })

    if not candidates:
        raise RuntimeError("No valid base found under reach/collision/same-side(min-angle)/occlusion constraints.")
    return candidates
