import os
import sys
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'third_party', 'SpaTrackerV2'))
import pycolmap
from models.SpaTrackV2.models.predictor import Predictor
import yaml
import easydict
import os
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from PIL import Image
import io
import moviepy.editor as mp
import tqdm
from models.SpaTrackV2.models.utils import get_points_on_a_grid
import glob
from rich import print
import pickle
import torchvision
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image
from models.SpaTrackV2.models.vggt4track.utils.pose_enc import pose_encoding_to_extri_intri
from models.SpaTrackV2.utils.visualizer import Visualizer


def load_obj_masks(data: dict):
    """
    Return object list for frame-0:
        [{'mask': bool array, 'name': name, 'bbox': (x1,y1,x2,y2)}, ...]
    Filter out names: 'ground' / 'hand' / 'robot'
    """
    frame_objs = data.get(0, {})  # only frame 0
    objs = []
    for oid, item in frame_objs.items():
        lbl = item["name"]
        if lbl in ("ground", "hand", "robot"):
            continue
        objs.append({
            "oid":  oid,
            "mask":  item["mask"].astype(bool),
            "name": lbl,
            "bbox":  item["bbox"]          # used for cropping
        })
    # Keep original behavior: sort by mask area (desc)
    objs.sort(key=lambda x: int(x["oid"]))
    return objs
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

def compute_tracks(single_scene_dict, key_cfg: dict, key: str, scale): 
    gpu_id = key_cfg["gpu"]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device(f"cuda:{int(gpu_id)}" if torch.cuda.is_available() else "cpu")
    video_tensor = np.array(single_scene_dict["images"])
    video_tensor = torch.from_numpy(video_tensor).permute(0, 3, 1, 2).float()
    resize_h = int(scale * video_tensor.shape[2])
    resize_w = int(scale * video_tensor.shape[3])
    video_tensor = torchvision.transforms.Resize((resize_h, resize_w))(video_tensor).to(device)
    depth_tensor = single_scene_dict["depths"]
    depth_tensor = torch.from_numpy(depth_tensor).float()
    depth_tensor = torchvision.transforms.Resize((resize_h, resize_w))(depth_tensor).to(device)
    N =  depth_tensor.shape[0]
    intrs = np.array(single_scene_dict["intrinsics"], dtype=np.float32)
    intrs = intrs * scale
    intrs[2,2] = 1.0
    intrs = np.tile(intrs, (N, 1, 1))

    intrs = torch.from_numpy(intrs).float().to(device)
    extrs = np.linalg.inv(single_scene_dict["extrinsics"]).astype(np.float32)
    extrs = torch.from_numpy(extrs).float().to(device)

    unc_metric = None
    first_frame_mask = single_scene_dict["recon"]["object_mask"]
    first_frame_mask = torch.from_numpy(first_frame_mask).to(device)
    first_frame_mask = torchvision.transforms.Resize((resize_h, resize_w))(first_frame_mask.unsqueeze(0))
    first_frame_mask = first_frame_mask.squeeze(0)
    first_frame_mask = first_frame_mask > 0
    mask_area = first_frame_mask.sum()

    model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline")

    # Compute grid_size such that: mask_area / all_size * (grid_size * grid_size) = 300
    all_size = video_tensor.shape[2] * video_tensor.shape[3]
    estimated_pts = 100
    ratio = mask_area.float() / float(all_size)
    grid_size = int(np.sqrt(estimated_pts / float(ratio)))
    print(f"[INFO] grid_size: {grid_size}")

    model.spatrack.track_num = 756
    model.eval()
    model =model.to(device)

    frame_H, frame_W = video_tensor.shape[2:]
    grid_pts = get_points_on_a_grid(grid_size, (frame_H, frame_W), device="cpu")
    grid_pts_int = grid_pts[0].long()
    mask_values = first_frame_mask[grid_pts_int[...,1], grid_pts_int[...,0]].cpu().numpy()
    grid_pts = grid_pts[:, mask_values]
    print(f"[INFO] grid_pts: {grid_pts.shape}")
    query_xyt = torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)[0].numpy().astype(np.float32)

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            (
                c2w_traj, intrs, point_map, conf_depth,
                track3d_pred, track2d_pred, vis_pred, conf_pred, video
            ) = model.forward(video_tensor, depth=depth_tensor,
                                intrs=intrs, extrs=extrs, 
                                queries=query_xyt,
                                fps=1, full_point=False, iters_track=4,
                                query_no_BA=True, fixed_cam=False, stage=1, unc_metric=unc_metric,
                                support_frame=len(video_tensor)-1, replace_ratio=0.2) 
    

    track2d_pred_cpu = track2d_pred.cpu()
    track3d_pred_cpu = track3d_pred.cpu()
    vis_pred_cpu = vis_pred.cpu()
    video_cpu = video.cpu()
    c2w_traj_cpu = c2w_traj.cpu()
    
    # Convert track3d_pred from camera coordinate to world coordinate
    # track3d_pred is in camera coordinate, need to transform to world coordinate using c2w_traj
    # Formula: P_world = c2w_traj[:3,:3] @ P_cam.T + c2w_traj[:3,3] for each frame
    # Using einsum: 'tij,tnj->tni' means for each frame t: R[t] @ P[t].T -> [t, n, i]
    track3d_pred_world = torch.einsum(
        'tij,tnj->tni', 
        c2w_traj_cpu[:, :3, :3], 
        track3d_pred_cpu[:, :, :3]
    ) + c2w_traj_cpu[:, None, :3, 3]  # [N_frames, 1, 3] broadcasting to [N_frames, num_points, 3]
    
    track3d_pred_cpu = track3d_pred_world
    
    del track2d_pred, track3d_pred, vis_pred, video, c2w_traj, intrs, point_map, conf_depth, conf_pred
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if key is not None:
        viser = Visualizer(save_dir=f"./outputs/{key}/reconstruction")
        viser.visualize(video=video_cpu[None],
                        tracks=track2d_pred_cpu[None][...,:2],
                        visibility=vis_pred_cpu[None],
                        filename=f"2d")
        del viser

    del video_cpu

    return track2d_pred_cpu.numpy(), track3d_pred_cpu.numpy(), vis_pred_cpu.numpy(), grid_pts


def optimize_trans(prev_coords, curr_coords):
    import numpy as np
    assert prev_coords.shape == curr_coords.shape
    N = prev_coords.shape[0]
    prev_mean = prev_coords.mean(axis=0)
    curr_mean = curr_coords.mean(axis=0)
    prev_centered = prev_coords - prev_mean
    curr_centered = curr_coords - curr_mean
    H = prev_centered.T @ curr_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    T = curr_mean - R @ prev_mean
    P = np.eye(4, dtype=np.float32)
    P[:3, :3] = R
    P[:3, 3] = T
    assert P.shape == (4, 4)
    return P



def compute_trans(track3d_pred, vis_pred):
    Ps = []
    prev_coords = track3d_pred[0]
    prev_vis = vis_pred[0] > 0.5
    for i in range(1,track3d_pred.shape[0]):
        curr_coords = track3d_pred[i]
        curr_vis = vis_pred[i] > 0.5
        overall_vis = (prev_vis & curr_vis).reshape(-1)
        P = optimize_trans(prev_coords[overall_vis][:,:3], curr_coords[overall_vis][:,:3])
        prev_coords = curr_coords
        prev_vis = curr_vis
        Ps.append(P)
    return Ps

def compute_total_move(track3d_pred):
    return np.linalg.norm(track3d_pred[1:][...,:3] - track3d_pred[:-1][...,:3], axis=2).sum() / track3d_pred.shape[1]

def compute_start_end_frame(track3d_pred, offset = 5):
    """
    track3d_pred: np.ndarray of shape (num_frames, num_points, 3)
    Returns:
        start_frame_idx, end_frame_idx (both are integers)
    """
    # Convert to numpy array if not already
    track3d_pred = np.asarray(track3d_pred)
    num_frames = track3d_pred.shape[0]
    num_points = track3d_pred.shape[1]
    # Compute velocities (frame-to-frame displacement)
    # velocity[fi] means velocity from frame fi-1 to fi
    velocities = np.linalg.norm(track3d_pred[1:][...,:3] - track3d_pred[:-1][...,:3], axis=2) # (num_frames-1, num_points)
    
    # Compute total movement per frame (sum across all points)
    frame_totals = np.sum(velocities, axis=1)  # (num_frames-1,)
    
    # Compute cumulative movement from start
    cumulative_movement = np.cumsum(frame_totals)  # (num_frames-1,)
    total_movement = cumulative_movement[-1]  # Total movement across all frames
    
    # Use percentage thresholds (more aggressive: start earlier, end later)
    start_percent = 0.15  # Start when 5% of total movement has occurred
    end_percent = 0.95    # End when 95% of total movement has occurred
    
    if total_movement < 1e-6:  # Very small movement, return full range
        start_frame_idx = 0
        end_frame_idx = num_frames - 1
    else:
        # Find first frame where cumulative movement exceeds start_percent
        start_frame_idx = None
        start_threshold = total_movement * start_percent
        for i in range(len(cumulative_movement)):
            if cumulative_movement[i] >= start_threshold:
                start_frame_idx = i
                break
        if start_frame_idx is None:
            start_frame_idx = 0
        
        # Find last frame where cumulative movement is below end_percent
        end_frame_idx = None
        end_threshold = total_movement * end_percent
        for i in range(len(cumulative_movement)-1, -1, -1):
            if cumulative_movement[i] <= end_threshold:
                end_frame_idx = i + 1  # i+1 is the frame index
                break
        if end_frame_idx is None:
            end_frame_idx = num_frames - 1

    start_frame_idx = max(0, start_frame_idx - offset)
    end_frame_idx = min(num_frames - 1, end_frame_idx + offset)

    return start_frame_idx, end_frame_idx


def seg_mask(object_mask, grid_pts, scale=0.5):
    """
    Select the index of kpts_2d_pred that lies on the object_mask.

    Args:
        object_mask (np.ndarray): 2D array representing the binary mask of the object. Shape: (H, W)
        kpts_2d_pred (np.ndarray): 2D points predicted of shape (N, 2) or (N, X), where the first 2 cols are x and y.

    Returns:
        indices (np.ndarray): Indices of kpts_2d_pred that lie within the mask (object_mask == 1)
    """
    object_mask = np.asarray(object_mask)
    kpts = np.asarray(grid_pts) / scale
    # Round and convert coordinates to int for indexing
    x = np.round(kpts[0, :, 0]).astype(int)
    y = np.round(kpts[0, :, 1]).astype(int)
    H, W = object_mask.shape[:2]

    # Filter points that are inside the image bounds
    valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    x_valid = x[valid]
    y_valid = y[valid]

    # Now, for valid points, check if they are on the mask
    mask_hits = np.zeros_like(valid, dtype=bool)
    mask_hits[valid] = object_mask[y_valid, x_valid] > 0

    indices = np.where(mask_hits)[0]
    return indices





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
    


def object_spatracker_cal(keys, key_scene_dicts, key_cfgs):
    base_dir = Path.cwd()
    for key in keys:
        scene_dict = key_scene_dicts[key]
        key_cfg = key_cfgs[key]
        scale = key_cfg["spatial_tracking_scale"]
        track2d_pred, track3d_pred, vis_pred, grid_pts = compute_tracks(scene_dict, key_cfg, key=key, scale=scale)
        objs = load_obj_masks(scene_dict["mask"])
        obj_total_moves = {}
        for obj in objs:
            oid = obj["oid"]
            name = obj["name"]
            obj_mask = obj["mask"]
            obj_indices = seg_mask(obj_mask, grid_pts, scale=scale)
            obj_track2d_pred = track2d_pred[:,obj_indices,:]
            obj_track3d_pred = track3d_pred[:,obj_indices,:]
            obj_vis_pred = vis_pred[:,obj_indices,:]
            obj_total_move = compute_total_move(obj_track3d_pred)
            obj_total_moves[oid] = obj_total_move
            scene_dict["info"]["objects"][oid]["type"] = "static"
           

        manipulated_oid = max(obj_total_moves, key=obj_total_moves.get)
        print(f"[INFO] object {oid} {name} is manipulated, total move: {obj_total_move}")
        for obj in objs:
            if obj["oid"] == manipulated_oid:
                mask = obj["mask"]
                break
        obj_indices = seg_mask(mask, grid_pts, scale=scale)
        obj_track3d_pred = track3d_pred[:,obj_indices,:]
        obj_vis_pred = vis_pred[:,obj_indices,:]
        obj_trans = compute_trans(obj_track3d_pred, obj_vis_pred)
        obj_trans = np.array(obj_trans)
        obj_trans = obj_trans.reshape(-1, 4, 4)
        obj_trans = obj_trans.tolist()
        start_frame_idx, end_frame_idx = compute_start_end_frame(obj_track3d_pred)
        if "obj_trans" not in scene_dict["recon"]:
            scene_dict["recon"]["obj_trans"] = {}
        scene_dict["recon"]["obj_trans"][manipulated_oid] = obj_trans
        scene_dict["info"]["objects"][manipulated_oid]["type"] = "manipulated"
        
        scene_dict["info"]["manipulated_oid"] = manipulated_oid
        scene_dict["recon"]["start_frame_idx"] = start_frame_idx
      
        object_masks = build_mask_array(manipulated_oid, scene_dict)
        hand_masks = scene_dict["recon"]["hand_masks"]
        refined_end_frame = refine_end_frame(hand_masks, object_masks)
        if refined_end_frame == track3d_pred.shape[0]:
            scene_dict["info"]["final_gripper_closed"] = True
            scene_dict["recon"]["end_frame_idx"] = track3d_pred.shape[0] - 1
        else:
            scene_dict["info"]["final_gripper_closed"] = False
            scene_dict["recon"]["end_frame_idx"] = min(refined_end_frame, end_frame_idx)

        end_frame_idx = scene_dict["recon"]["end_frame_idx"]
        start_frame_idx = scene_dict["recon"]["start_frame_idx"]
        print(f"[INFO] start_frame_idx: {start_frame_idx}, end_frame_idx: {end_frame_idx}, manipulated_oid: {manipulated_oid}, name: {name}")
        with open(base_dir / f'outputs/{key}/scene/scene.pkl', 'wb') as f:
            pickle.dump(scene_dict, f)
        key_scene_dicts[key] = scene_dict
        return key_scene_dicts

if __name__ == "__main__":
    base_dir = Path.cwd()
    cfg_path = base_dir / "config" / "config.yaml"
    cfg = yaml.safe_load(cfg_path.open("r"))
    keys = cfg["keys"]
    from utils.compose_config import compose_configs
    key_cfgs = {key: compose_configs(key, cfg) for key in keys}
    key_scene_dicts = {}
    for key in keys:
        scene_pkl = base_dir / f'outputs/{key}/scene/scene.pkl'
        with open(scene_pkl, 'rb') as f:
            scene_dict = pickle.load(f)
        key_scene_dicts[key] = scene_dict
    object_spatracker_cal(keys, key_scene_dicts, key_cfgs)