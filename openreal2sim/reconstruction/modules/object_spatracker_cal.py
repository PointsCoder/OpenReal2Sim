
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
import decord
import pickle
import torchvision
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image
from models.SpaTrackV2.models.vggt4track.utils.pose_enc import pose_encoding_to_extri_intri

def compute_tracks(single_scene_dict, key_cfg: dict): 
    gpu_id = key_cfg["gpu"]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device(f"cuda:{int(gpu_id)}" if torch.cuda.is_available() else "cpu")
   
    video_tensor = np.array(single_scene_dict["images"])
    video_tensor = torch.from_numpy(video_tensor).permute(0, 3, 1, 2).float().to(device)
    resize_h = int(0.5 * video_tensor.shape[2])
    resize_w = int(0.5 * video_tensor.shape[3])
    video_tensor = torchvision.transforms.Resize((resize_h, resize_w))(video_tensor)
    depth_tensor = single_scene_dict["depths"]
    depth_tensor = torch.from_numpy(depth_tensor).float().to(device)
    depth_tensor = torchvision.transforms.Resize((resize_h, resize_w))(depth_tensor)
    N =  depth_tensor.shape[0]
    intrs = np.array(single_scene_dict["intrinsics"], dtype=np.float32)
    intrs = intrs * 0.5
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
    model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline")
    grid_size = key_cfg.get("grid_size", 100)
    model.spatrack.track_num = grid_size
    model.eval()
    model =model.to(device)

    frame_H, frame_W = video_tensor.shape[2:]
    grid_pts = get_points_on_a_grid(grid_size, (frame_H, frame_W), device="cpu")
    grid_pts_int = grid_pts[0].long()
    mask_values = first_frame_mask[grid_pts_int[...,1], grid_pts_int[...,0]].cpu().numpy()
    grid_pts = grid_pts[:, mask_values]
    query_xyt = torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)[0].numpy().astype(np.float32)

    with torch.no_grad():
        (
            c2w_traj, intrs, point_map, conf_depth,
            track3d_pred, track2d_pred, vis_pred, conf_pred, video
        ) = model.forward(video_tensor, depth=depth_tensor,
                            intrs=intrs, extrs=extrs, 
                            queries=query_xyt,
                            fps=1, full_point=False, iters_track=4,
                            query_no_BA=True, fixed_cam=False, stage=1, unc_metric=unc_metric,
                            support_frame=len(video_tensor)-1, replace_ratio=0.2) 

    ### it's also feasible to compute using pred2dtracks.
        
    return track2d_pred.cpu().numpy(), track3d_pred.cpu().numpy(), vis_pred.cpu().numpy()


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



def compute_start_end_frame(track3d_pred, offset = 5):
    """
    track3d_pred: np.ndarray of shape (num_frames, num_points, 3)
    Returns:
        start_frame_idx, end_frame_idx (both are integers)
    """
    # Convert to numpy array if not already
    track3d_pred = np.asarray(track3d_pred)
    num_frames = track3d_pred.shape[0]

    # Compute velocities (frame-to-frame displacement)
    # velocity[fi] means velocity from frame fi-1 to fi
    velocities = np.linalg.norm(track3d_pred[1:][...,:3] - track3d_pred[:-1][...,:3], axis=2) # (num_frames-1, num_points)

    # Find first frame where at least 100 points velocity > 0.005 (5mm = 0.5cm)
    start_frame_idx = None
    for i in range(velocities.shape[0]):
        if np.sum(velocities[i] > 0.005) >= 100:
            start_frame_idx = i
            break
    if start_frame_idx is None:
        start_frame_idx = 0

    # Find last frame where all points velocity < 0.01 (1cm)
    end_frame_idx = None
    for i in range(velocities.shape[0]-1, -1, -1):
        if np.all(velocities[i] < 0.01):
            end_frame_idx = i+1 # i+1 is the "end" frame
            break
    if end_frame_idx is None:
        end_frame_idx = num_frames - 1


    start_frame_idx = max(0, start_frame_idx - offset)
    end_frame_idx = min(num_frames - 1, end_frame_idx + offset)

    return start_frame_idx, end_frame_idx


def object_spatracker_cal(keys, key_scene_dicts, key_cfgs):
    base_dir = Path.cwd()
    for key in keys:
        scene_dict = key_scene_dicts[key]
        key_cfg = key_cfgs[key]
        track2d_pred, track3d_pred, vis_pred = compute_tracks(scene_dict, key_cfg)
        spatrack_trans = compute_trans(track3d_pred, vis_pred)
        start_frame_idx, end_frame_idx = compute_start_end_frame(track3d_pred, offset=key_cfg["startend_offset"])
        scene_dict["recon"]["spatrack_trans"] = spatrack_trans
        scene_dict["recon"]["start_frame_idx"] = start_frame_idx
        scene_dict["recon"]["end_frame_idx"] = end_frame_idx
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