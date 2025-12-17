
import yaml
import numpy as np
from pyquaternion import Quaternion
from pathlib import Path
import pickle
import cv2
import trimesh
import sys
import json
import os
base_dir = Path.cwd()
sys.path.append(str(base_dir / "openreal2sim" / "motion" / "modules"))

def recompute_traj(traj, c2w, extrinsics):
    for i in range(len(traj)):
        traj[i] = c2w @ np.linalg.inv(extrinsics[0]) @ extrinsics[i] @ np.linalg.inv(c2w) @ traj[i]
    return traj

def recompute_all_traj(scene_json_dict, scene_dict):
    for obj_id, obj in scene_json_dict["objects"].items():
       for key in 'fdpose_trajs', 'simple_trajs', 'hybrid_trajs':
        traj_path = obj[key]
        traj = np.load(traj_path)
        traj = traj.reshape(-1, 4, 4)
        c2w = np.array(scene_json_dict["camera"]["camera_opencv_to_world"]).astype(np.float64).reshape(4,4) 
        extrinsics = np.array(scene_dict["extrinsics"]).astype(np.float64).reshape(-1,4,4) 
        traj = recompute_traj(traj, c2w, extrinsics)
        new_traj_path = traj_path.replace(".npy", "_recomputed.npy").replace("reconstruction", "motion")
        scene_json_dict["objects"][obj_id][f"{key}_recomputed"] = str(new_traj_path)
        np.save(new_traj_path, traj)


def get_mask_from_frame(scene_dict, frame_index, oid):
    mask = scene_dict["mask"][frame_index][oid]["mask"]
    return mask


def get_object_bbox(mesh_in_path):
    mesh = trimesh.load(mesh_in_path)
    bbox = mesh.bounds
    return bbox



def lift_traj(traj, height=0.02):
    new_traj = []
    for i in range(len(traj)):
        new_traj.append(traj[i])
    for i in range(1, len(new_traj) - 1):
        new_traj[i][2][3] += height
    return new_traj

def determine_stacking_bbox(bbox_0, bbox_1):
    """
    Determine if two bboxes have intersection on x and y axes.

    bbox_0, bbox_1: np.ndarray, (2,3) or (2,2) or (2,?) where axis 0 is (min,max) and axis 1 is x,y,(z)

    Returns: True if x and y intervals overlap (i.e. stacking is possible), False otherwise

    Generally this is enough for start frame.
    """
    # Extract x/y intervals
    x0_min, x0_max = bbox_0[0][0], bbox_0[1][0]
    y0_min, y0_max = bbox_0[0][1], bbox_0[1][1]
    x1_min, x1_max = bbox_1[0][0], bbox_1[1][0]
    y1_min, y1_max = bbox_1[0][1], bbox_1[1][1]
    z0_min, z0_max = bbox_0[0][2], bbox_0[1][2]
    z1_min, z1_max = bbox_1[0][2], bbox_1[1][2]

    x_overlap = not (x0_max < x1_min or x1_max < x0_min)
    y_overlap = not (y0_max < y1_min or y1_max < y0_min)
    if x_overlap and y_overlap:
        return True

    return False



def load_obj_masks(data: dict):
    """
    Return object list for frame-0:
        [{'mask': bool array, 'name': name, 'bbox': (x1,y1,x2,y2)}, ...]
    Filter out names: 'ground' / 'hand' / 'robot'
    """
    frame_objs = data.get(0, {})  # only frame 0
    objs = []
    for oid, item in frame_objs.items():
        name = item["name"]
        if name in ("ground", "hand", "robot"):
            continue
        objs.append({
            "oid":  oid,
            "mask":  item["mask"].astype(bool),
            "name": name,
            "bbox":  item["bbox"]          # used for cropping
        })
    # Keep original behavior: sort by mask area (desc)
    objs.sort(key=lambda x: int(x["oid"]))
    return objs



def pose_distance(T1, T2):
    # Translation distance
    t1 = T1[:3, 3]
    t2 = T2[:3, 3]
    trans_dist = np.linalg.norm(t2 - t1)
    # Rotation difference (angle)
    R1 = T1[:3, :3]
    R2 = T2[:3, :3]
    dR = R2 @ R1.T
    # numerical errors can make the trace slightly out of bounds
    cos_angle = (np.trace(dR) - 1) / 2
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    return trans_dist, angle

def build_mask_array(
    oid: int,
    scene_dict: dict[str, any]
) -> np.ndarray:

    H, W, N = scene_dict["height"], scene_dict["width"], scene_dict["n_frames"]
    out = np.zeros((N, H, W), dtype=np.uint8)
    for i in range(N):
        m = scene_dict["mask"][i][oid]["mask"]
        out[i] = m.astype(np.uint8)
    return out



def downsample_traj(trajectory, trans_threshold=0.05, rot_threshold=np.radians(20)):
    """
    Downsample pose trajectory where each element is a 4x4 transformation matrix (SE3).
    Retains keyframes with sufficient translation or rotation from the previous kept frame.
    Returns: list of indices for retained frames.
    """
    if len(trajectory) <= 2:
        return list(range(len(trajectory)))


    downsampled_indices = [0]
    prev_idx = 0
    for i in range(1, len(trajectory) - 1):
        trans_dist, angle = pose_distance(trajectory[prev_idx], trajectory[i])
        if trans_dist >= trans_threshold or angle >= rot_threshold:
            downsampled_indices.append(i)
            prev_idx = i
    if len(downsampled_indices) > 0:
        last_kept_idx = downsampled_indices[-1]
        trans_dist, angle = pose_distance(trajectory[last_kept_idx], trajectory[-1])
        if trans_dist > 0.6 * trans_threshold or angle > 0.6 * rot_threshold:
            downsampled_indices.append(len(trajectory) - 1)
        elif trans_dist > 0.1 * trans_threshold or angle > 0.1 * rot_threshold:
            downsampled_indices[-1] = len(trajectory) - 1

    return downsampled_indices

def refine_first_frame(hand_masks, object_masks, first_frame):
    N = min(len(hand_masks), len(object_masks))
    kernel = np.ones((10, 10), np.uint8)
    for i in range(first_frame, N):
        if hand_masks[i] is None:
            continue
        hand_mask = hand_masks[i].astype(np.uint8)
        obj_mask = object_masks[i].astype(np.uint8)
        obj_mask_dilated = cv2.dilate(obj_mask, kernel, iterations=1)
        overlap = (hand_mask & obj_mask_dilated).sum()
        if overlap > 0:
            return i
    return first_frame

def refine_end_frame(hand_masks, object_masks):
    N = min(len(hand_masks), len(object_masks))
    kernel = np.ones((10, 10), np.uint8)
    for i in range(N-1, -1, -1):
        if hand_masks[i] is None:
            continue
        hand_mask = hand_masks[i].astype(np.uint8)
        obj_mask = object_masks[i].astype(np.uint8)
        obj_mask_dilated = cv2.dilate(obj_mask, kernel, iterations=1)
        overlap = (hand_mask & obj_mask_dilated).sum()
        if overlap > 0:
            return i + 1
    return 0                 
    

def demo_motion_process(keys, key_scene_dicts, key_cfgs):
    base_dir = Path.cwd()
    for key in keys:
        scene_dict = key_scene_dicts[key]
        scene_json_dict_path = base_dir / f"outputs/{key}/simulation/scene.json"
        with open(scene_json_dict_path, "r") as f:
            scene_json_dict = json.load(f)
        recompute_all_traj(scene_json_dict, scene_dict)
        key_cfg = key_cfgs[key]
        max_placement_oid = None
        max_placement_distance = 0.0
        placement_distances = {}
        fd_traj_dict = {}
        for obj_id, obj in scene_json_dict["objects"].items():
            obj_traj = obj["simple_trajs_recomputed"]
            obj_traj = np.load(obj_traj)
            obj_traj = obj_traj.reshape(-1, 4, 4)
            abs_distance, _ = pose_distance(obj_traj[0], obj_traj[-1])
            fd_traj = obj["fdpose_trajs_recomputed"]
            fd_traj = np.load(fd_traj)
            fd_traj = fd_traj.reshape(-1, 4, 4)
            fd_dist, fd_angle = pose_distance(fd_traj[0], fd_traj[-1])
          
            fd_traj_dict[obj_id] = {
                "fd_distance": fd_dist,
                "fd_angle": fd_angle
            }
            print(f"Object: {obj_id}, distance: {abs_distance}")
            if abs_distance >= max_placement_distance:
                max_placement_distance = abs_distance
                max_placement_oid = obj_id
            scene_json_dict["objects"][obj_id]["type"] = "static"  
            placement_distances[obj_id] = abs_distance
        scene_json_dict["manipulated_oid"] = max_placement_oid
        scene_dict["info"]["manipulated_oid"] = max_placement_oid
        name = scene_json_dict["objects"][max_placement_oid]["name"]
        scene_json_dict["objects"][max_placement_oid]["type"] = "manipulated"
        print(f"Manipulated object: {max_placement_oid}, distance: {abs_distance}")
        manipulated_fd_trajs_path = scene_json_dict["objects"][max_placement_oid]["fdpose_trajs_recomputed"]
        manipulated_fd_trajs = np.load(manipulated_fd_trajs_path)
        angle_displacement = [pose_distance(manipulated_fd_trajs[i], manipulated_fd_trajs[i+1])[1] for i in range(len(manipulated_fd_trajs)-1)]
        angle_displacement_sum = sum(angle_displacement)
        # manipulated_simple_trajs_path = scene_json_dict["objects"][max_placement_oid]["simple_trajs_recomputed"]
        # manipulated_simple_trajs = np.load(manipulated_simple_trajs_path)
        # pos_displacement = [pose_distance(manipulated_simple_trajs[i], manipulated_fd_trajs[i])[0] for i in range(len(manipulated_fd_trajs))]
        fd_traj_dict.pop(max_placement_oid)
        traj_key = 'fdpose_trajs_recomputed'
        if len(fd_traj_dict.keys()) > 0:
            max_fd_distance = max(fd_traj_dict.values(), key=lambda x: x["fd_distance"])
            max_fd_angle = max(fd_traj_dict.values(), key=lambda x: x["fd_angle"])
        else:
            max_fd_distance = 0.0
            max_fd_angle = 0.0
        if angle_displacement_sum > 180 or max_fd_angle > np.radians(20):
            traj_key = 'simple_trajs_recomputed'
        else:
            if max_fd_distance > 0.05:
                traj_key = 'hybrid_trajs_recomputed'
        print(f"Traj key: {traj_key}")
        scene_json_dict["traj_key"] = traj_key
        scene_dict["info"]["traj_key"] = traj_key
        manipulated_trajs_path = scene_json_dict["objects"][str(max_placement_oid)][traj_key]
        manipulated_trajs = np.load(manipulated_trajs_path)
        object_masks = build_mask_array(int(max_placement_oid), scene_dict)
        hand_masks = scene_dict["motion"]["hand_masks"]
        first_frame = 0
        first_frame = refine_first_frame(hand_masks, object_masks, first_frame)
        end_frame = refine_end_frame(hand_masks, object_masks)
        print(f"End frame: {end_frame}")
        print(f"First frame: {first_frame}")
      

        if end_frame == len(manipulated_trajs) - 1:
            scene_json_dict["gripper_closed"] = True
        elif end_frame == 0:
            if np.abs(manipulated_trajs[0][2][3]) < 0.03:
                scene_json_dict["gripper_closed"] = False
            else:
                scene_json_dict["gripper_closed"] = True
            end_frame = len(manipulated_trajs) - 1
        else:
            scene_json_dict["gripper_closed"] = False
        
        
        trajs = manipulated_trajs[first_frame:end_frame]

        downsampled_indices = downsample_traj(trajs, trans_threshold=0.03, rot_threshold=np.radians(20))
        clean_downsampled_indices = [0]
        for i in downsampled_indices[1:]:
            clean_downsampled_indices.append(i + first_frame)
        downsampled_indices = clean_downsampled_indices
        scene_json_dict["chosen_indices"] = downsampled_indices
        trajs = manipulated_trajs[downsampled_indices]
        

        # trajs = lift_traj(trajs, height=0.02)
        # scene_json_dict["lift_height"] = 0.02
        print(f"Downsampled indices: {downsampled_indices}")
        downsampled_traj_path = base_dir / f"outputs/{key}/simulation" / f"{max_placement_oid}_final_traj.npy"
        
      
        print(f"Trajs: {trajs[-1]}")
        np.save(downsampled_traj_path, trajs)
        scene_json_dict["objects"][max_placement_oid]["final_trajs"] = str(downsampled_traj_path)
        if len(downsampled_indices) > 1:
            scene_dict["info"]["start_frame_idx"] = downsampled_indices[1]
        else:
            scene_dict["info"]["start_frame_idx"] = 0
        end_pose = trajs[-1]
        mesh_in_path = scene_json_dict["objects"][max_placement_oid]["optimized"]
        mesh = trimesh.load(mesh_in_path)
        mesh.apply_transform(end_pose)
        path = base_dir / f"outputs/{key}/simulation/{max_placement_oid}_{name}_end.glb"
        mesh.export(path)
        scene_json_dict["objects"][max_placement_oid]["end_mesh"] = str(path)
        bbox_end = get_object_bbox(path)

        start_related = []
        end_related = []
        mesh_in_path = scene_json_dict["objects"][max_placement_oid]["optimized"]
        bbox_start = get_object_bbox(mesh_in_path)
        for oid, obj in scene_json_dict["objects"].items():
            if not isinstance(oid, int):
                continue
            if oid == max_placement_oid:
                continue
            mesh_in_path = obj["optimized"]
            bbox = get_object_bbox(mesh_in_path)
            if determine_stacking_bbox(bbox_start, bbox):
                start_related.append(oid)
            if determine_stacking_bbox(bbox_end, bbox):
                end_related.append(oid)

        print(f"Start related: {start_related}")
        print(f"End related: {end_related}")

        scene_json_dict["start_related"] = list(start_related)
        scene_json_dict["end_related"] = list(end_related)


        scene_path = scene_json_dict["scene_mesh"]["optimized"]
        scene_mesh = trimesh.load(scene_path)
        scene_mesh = scene_mesh + mesh
        if not os.path.exists(base_dir / f"outputs/{key}/motion"):
            os.makedirs(base_dir / f"outputs/{key}/motion")
        path = base_dir / f"outputs/{key}/motion/scene_end.glb"
        scene_mesh.export(path)



        if len(end_related) > 0 :
            target_oid = end_related[0]
            scene_json_dict["target_oid"] = target_oid
            scene_json_dict["task_type"] = "targetted_pick_place"
            if len(scene_json_dict["task_desc"]) > 0:
                task_desc = "Pick up the " + name + " and place it on the " + target_oid + "."
                if task_desc not in scene_json_dict["task_desc"]:
                    scene_json_dict["task_desc"].append(task_desc)
            else:
                task_desc = "Pick up the " + name + " and place it on the " + target_oid + "."
                if task_desc not in scene_json_dict["task_desc"]:
                    scene_json_dict["task_desc"] = [task_desc]
        else:
            if not scene_json_dict["gripper_closed"] and scene_dict["motion"]["has_hand"]:
                scene_json_dict["task_type"] = "simple_pick_place"
                task_desc = "Pick up the " + name + " and place it on the ground."
                if len(scene_json_dict.get("task_desc", [])) == 0:
                    scene_json_dict["task_desc"] = [task_desc]
                if task_desc not in scene_json_dict["task_desc"]:
                    scene_json_dict["task_desc"].append(task_desc)
            else:
                scene_json_dict["task_type"] = "simple_pick"
                task_desc = "Pick up the " + name + "."
                if len(scene_json_dict.get("task_desc", [])) == 0:
                    scene_json_dict["task_desc"] = [task_desc]
                if task_desc not in scene_json_dict["task_desc"]:
                    scene_json_dict["task_desc"].append(task_desc)

        print(f"Task description: {scene_json_dict['task_desc']}")
        print(f"Task type: {scene_json_dict['task_type']}")
        
        with open(base_dir / f"outputs/{key}/simulation/scene.json", "w") as f:
            json.dump(scene_json_dict, f, indent=2)
        with open(base_dir / f"outputs/{key}/scene/scene.pkl", "wb") as f:
            pickle.dump(scene_dict, f)
    return key_scene_dicts


if __name__ == "__main__":
    base_dir = Path.cwd()
    cfg_path = base_dir / "config" / "config.yaml"
    cfg = yaml.safe_load(cfg_path.open("r"))
    keys = cfg["keys"]
    from utils.compose_config import compose_configs
    key_cfgs = {key: compose_configs(key, cfg) for key in keys} 
    print(f"Key cfgs: {key_cfgs}")
    key_scene_dicts = {}
    for key in keys:
        scene_pkl = base_dir / f'outputs/{key}/scene/scene.pkl'
        with open(scene_pkl, 'rb') as f:
            scene_dict = pickle.load(f)
        key_scene_dicts[key] = scene_dict
    demo_motion_process(keys, key_scene_dicts, key_cfgs)

        

