"""
Trajectory smoothing with axis-flip/axis-permutation disambiguation in the object-local frame.
Note this is just an experimental feature and may not work well.
"""

import numpy as np
import transforms3d
from pathlib import Path
import yaml
import json
import argparse

def _proj_to_SO3(R: np.ndarray) -> np.ndarray:
    """Project a near-rotation 3x3 matrix to the closest element in SO(3) via SVD."""
    U, _, Vt = np.linalg.svd(R)
    R_star = U @ Vt
    if np.linalg.det(R_star) < 0:
        U[:, -1] *= -1
        R_star = U @ Vt
    return R_star

def _angle_between_rotations(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Geodesic angle on SO(3): arccos( (trace(Ra^T Rb) - 1)/2 )."""
    M = Ra.T @ Rb
    cos_th = (np.trace(M) - 1.0) * 0.5
    cos_th = np.clip(cos_th, -1.0, 1.0)
    return float(np.arccos(cos_th))

def _quat_slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Shortest-arc SLERP between unit quaternions q0 -> q1. Quaternions are (w,x,y,z)."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        q = (1.0 - alpha) * q0 + alpha * q1
        return q / np.linalg.norm(q)
    theta0 = np.arccos(dot)
    s0 = np.sin((1 - alpha) * theta0) / np.sin(theta0)
    s1 = np.sin(alpha * theta0) / np.sin(theta0)
    q = s0 * q0 + s1 * q1
    return q / np.linalg.norm(q)

def _generate_group24() -> np.ndarray:
    """
    Generate 24 orientation-preserving signed permutation matrices (cube rotations).
    Each is 3x3, has one ±1 per row/column, det=+1.
    """
    mats = []
    perms = [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]
    for p in perms:
        for sx in (-1,1):
            for sy in (-1,1):
                for sz in (-1,1):
                    M = np.zeros((3,3), float)
                    M[0, p[0]] = sx
                    M[1, p[1]] = sy
                    M[2, p[2]] = sz
                    if np.linalg.det(M) > 0:
                        mats.append(M)
    mats = np.stack(mats, axis=0)
    assert mats.shape[0] == 24
    return mats

def smooth_object_local_trajectory(
    poses: np.ndarray,
    rot_smooth_alpha: float = 0.2,
    trans_smooth_alpha: float = 0.2,
    repair_SO3: bool = True,
) -> np.ndarray:
    """
    Smooth a (N,4,4) trajectory with axis-flip/axis-permutation disambiguation in the OBJECT-LOCAL frame.

    Assumptions:
      - Each pose T_t = [R_t, t_t; 0 0 0 1], rotation acts actively on column vectors.
      - Coordinate re-labeling / flips (right-handed only) may occur in the object's local frame.
      - Therefore we correct R_meas by RIGHT-multiplying S in G24, i.e., R_corr = R_meas @ S.

    Args:
        poses: np.ndarray of shape (N, 4, 4), the raw trajectory.
        rot_smooth_alpha: rotation smoothing factor in [0,1] (higher -> less smoothing).
        trans_smooth_alpha: translation smoothing factor in [0,1].
        repair_SO3: if True, project measured rotations to SO(3) per frame (robust to noise/reflection).

    Returns:
        np.ndarray of shape (N, 4, 4): the smoothed trajectory.
    """
    assert poses.ndim == 3 and poses.shape[1:] == (4,4), "poses must be (N,4,4)"
    N = poses.shape[0]
    if N == 0:
        return poses.copy()

    G = _generate_group24()

    out = np.zeros_like(poses, dtype=float)
    R_prev = None
    q_prev = None
    t_prev = None

    for i in range(N):
        T_in = poses[i].astype(float)
        R_meas = T_in[:3, :3]
        t_meas = T_in[:3, 3]

        if repair_SO3:
            R_meas = _proj_to_SO3(R_meas)

        # --- Disambiguation in object-local frame: choose S to minimize distance to previous rotation.
        if R_prev is not None:
            best_angle = np.inf
            R_best = R_meas
            for S in G:
                Rc = R_meas @ S  # RIGHT-multiply: local/object-frame relabel/flip
                ang = _angle_between_rotations(R_prev, Rc)
                if ang < best_angle:
                    best_angle = ang
                    R_best = Rc
        else:
            R_best = R_meas

        # --- Quaternion sign continuity + SLERP smoothing.
        q_best = transforms3d.quaternions.mat2quat(R_best)  # (w,x,y,z)
        if q_prev is not None and float(np.dot(q_best, q_prev)) < 0.0:
            q_best = -q_best
        q_smooth = q_best if q_prev is None else _quat_slerp(q_prev, q_best, rot_smooth_alpha)
        R_smooth = transforms3d.quaternions.quat2mat(q_smooth)

        # --- Translation EMA smoothing (object-local relabeling doesn't change translation component).
        t_smooth = t_meas if t_prev is None else (1.0 - trans_smooth_alpha) * t_prev + trans_smooth_alpha * t_meas

        # --- Compose output pose.
        T_out = np.eye(4, dtype=float)
        T_out[:3, :3] = R_smooth
        T_out[:3, 3]  = t_smooth
        out[i] = T_out

        # --- Update memory.
        R_prev = R_smooth
        q_prev = q_smooth
        t_prev = t_smooth

    return out


def main():
    parser = argparse.ArgumentParser(description="Smooth trajectory poses")
    parser.add_argument("--traj", type=str, default="fdpose_trajs", help="Path to config file")
    args = parser.parse_args()

    cfg_path = Path.cwd() / "config" / "config.yaml"
    cfg = yaml.safe_load(cfg_path.open("r"))
    keys = cfg["keys"]

    base_dir = Path.cwd()
    out_dir = base_dir / "outputs"
    for key in keys:
        print(f"\n========== [Trajectory Smoothing] Processing key: {key} ==========")

        scene_json = out_dir / key / "simulation" / "scene.json"
        if not scene_json.exists():
            raise FileNotFoundError(scene_json)

        scene_dict = json.load(open(scene_json))
        objects = scene_dict.get("objects", {})
        if not isinstance(objects, dict) or len(objects) == 0:
            print(f"[WARN][{key}] scene_dict['objects'] is empty.")
            return

        key_dir = out_dir / key
        for i, obj in objects.items():
            traj_path = obj[args.traj]
            traj = np.load(traj_path).astype(np.float32) # relative poses (N,4,4)
            
            smoothed_trajs = smooth_object_local_trajectory(
                traj,
                rot_smooth_alpha=0.2,
                trans_smooth_alpha=0.2,
                repair_SO3=True,
            ).astype(np.float32)
            smoothed_path = key_dir / "reconstruction" / "motions" / f"{i}_{obj['name']}_smoothed_trajs.npy"
            np.save(smoothed_path, smoothed_trajs)
            scene_dict["objects"][i]["smooth_trajs"] = str(smoothed_path)

        json.dump(scene_dict, open(scene_json, "w"), indent=2)

if __name__ == "__main__":
    main()