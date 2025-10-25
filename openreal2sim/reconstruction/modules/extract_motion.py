### we may first decide the starting and ending point of the trajectory.
### Then we consider:
### First we decide whether this traj involves more than one object.
### This is easy, just check whether there exists static objects in the scene. If so set the nearest one to the endpoint as the target object.
### If no, we consider this trajectory as 'random-destination-centric' meaning the destination can take anywhere on the ground plane.
### Then we decide a 'stacking' sequence for the objects. If some objects are stacked together, they are transformed together. This is already computed in the recon state.
import yaml
import numpy as np
from pyquaternion import Quaternion


def downsample_traj(trajectory, trans_threshold=0.05, rot_threshold=np.radians(20)):
    # ref: nvidia SPOT code.
    if len(trajectory) <= 2:
        return trajectory
    
    # ref: https://math.stackexchange.com/questions/2124361/quaternions-multiplication-order-to-rotate-unrotate
    def calculate_action(current_pose: np.ndarray, goal_pose: np.ndarray):
        x1, y1, z1, qx1, qy1, qz1, qw1 = current_pose
        x2, y2, z2, qx2, qy2, qz2, qw2 = goal_pose
        
        #QW == Qp * Qch
        QW = Quaternion(qw2, qx2, qy2, qz2)
        Qch = Quaternion(qw1, qx1, qy1, qz1)
        # Qp == QW * Qch.Inversed
        Qp = QW * Qch.inverse

        a_qw, a_qx, a_qy, a_qz = list(Qp)
        a_x, a_y, a_z = x2-x1, y2-y1, z2-z1

        action = [a_x, a_y, a_z] + [a_qx, a_qy, a_qz, a_qw]
        return np.array(action)
    
    def compute_pos_diff(pose1, pose2, trans_threshold=0.05, rot_threshold=np.radians(20)):
        action = calculate_action(pose1, pose2)
        dist = np.linalg.norm(action[:3])
        
        rot = R.from_quat(pose1[3:]) 
        rotvec1 = rot.as_rotvec()
        rot = R.from_quat(pose2[3:]) 
        rotvec2 = rot.as_rotvec()
        angle_diff = rotvec2 - rotvec1
        
        return dist >= trans_threshold or np.max(angle_diff) >= rot_threshold

    downsampled_indices = [0] 
    prev_idx = 0
    
    for i in range(1, len(trajectory) - 1):
        if compute_pos_diff(trajectory[prev_idx], trajectory[i], trans_threshold, rot_threshold):
            downsampled_indices.append(i)
            prev_idx = i
    

    if len(downsampled_indices) > 0:
        last_kept_idx = downsampled_indices[-1]
        if compute_pos_diff(trajectory[last_kept_idx], trajectory[-1], 
                           trans_threshold * 0.5, rot_threshold * 0.8):
            downsampled_indices.append(len(trajectory) - 1)

    return downsampled_indices




def extract_motion(keys, key_scene_dicts, key_cfgs):
    base_dir = Path.cwd()
    for key in keys:
        scene_dict = key_scene_dicts[key]
        key_cfg = key_cfgs[key]
        start_frame_idx = scene_dict["recon"]["start_frame_idx"]
        end_frame_idx = scene_dict["recon"]["end_frame_idx"]
        manipulated_oid = scene_dict["info"]["manipulated_oid"]
        manipulated_trajs_path = scene_dict["info"]["objects"][manipulated_oid]["fdpose_trajs"]
        manipulated_trajs = np.load(manipulated_trajs_path)
        downsampled_indices = downsample_traj(manipulated_trajs, trans_threshold=0.05, rot_threshold=np.radians(20))
        for obj_id, obj in scene_dict["info"]["objects"].items():
            obj_name = obj["name"]
            if obj_name == "ground" or obj_name == "hand" or obj_name == "robot":
                continue
            if obj["type"] == "static":
                continue
            obj_trajs_path = obj["fdpose_trajs"]
            abs_trajs_path = obj["abs_fdpose_trajs"]
            obj_trajs = np.load(obj_trajs_path)
            downsampled_indices = downsample_traj(obj_trajs, trans_threshold=0.05, rot_threshold=np.radians(20))
            new_trajs = obj_trajs[downsampled_indices]
            new_trajs[0] = obj_trajs[0]
            scene_dict["info"]["objects"][obj_id]["rel_trajs"] = new_trajs
            abs_trajs = np.load(abs_trajs_path)
            new_abs_trajs = abs_trajs[downsampled_indices]
            new_abs_trajs[0] = abs_trajs[0]
            scene_dict["info"]["objects"][obj_id]["abs_trajs"] = new_abs_trajs
            
            mesh_in_path = obj["registered"]
            m_base = trimesh.load(mesh_in_path, force='mesh')
            m_fd = m_base.copy(); m_fd.apply_transform(new_abs_trajs[-1])
            fdpose_path = base_dir / f"outputs/{key}/reconstruction/objects" / f"{obj_id}_{obj_name}_ending_pose.glb"
            m_fd.export(str(fdpose_path))
            scene_dict["info"]["objects"][obj_id]["ending_mesh"] = str(fdpose_path)

        return key_scene_dicts