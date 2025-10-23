#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run depth and camera tracking from images
Inputs:
    - outputs/{key_name}/images/frame_00000.jpg, frame_00001.jpg, ...
    - outputs/{key_name}/scene/scene.pkl (with initial frames)
Outputs:
    - outputs/{key_name}/scene/scene.pkl (saving frames, depths, and camera infos)
Parameters:
    - 
Note:
    - if for a single image, running monocular metric depth prediction
    - if for multiple frames, running Mega-SaM pipeline
"""

import os
import argparse
import subprocess
from pathlib import Path
import yaml
import numpy as np
import pickle

from utils.compose_config import compose_configs

def run(cmd_list, env):
    """Run a command and stream output; raise on non-zero exit."""
    print("Running:", " ".join(str(x) for x in cmd_list))
    subprocess.run(cmd_list, check=True, env=env)


def run_megasam(video_name: str, key_cfgs: dict):
    gpu = key_cfgs["gpu"]

    base_dir = Path.cwd()
    data_dir = base_dir / "outputs" / video_name
    frame_dir = data_dir / "images"
    depth_anything_dir = data_dir / "geometry" / "depthanything"
    uni_depth_dir = data_dir / "geometry" / "unidepth"
    recon_dir = data_dir / "geometry" / "reconstruction"

    depth_anything_dir.mkdir(parents=True, exist_ok=True)
    uni_depth_dir.mkdir(parents=True, exist_ok=True)
    recon_dir.mkdir(parents=True, exist_ok=True)

    # Common environment (GPU binding)
    env_base = os.environ.copy()
    env_base["CUDA_VISIBLE_DEVICES"] = gpu

    # 1) DepthAnything
    env_da = env_base.copy()
    da_py_path = str(base_dir / "third_party" / "mega-sam")
    env_da["PYTHONPATH"] = env_da.get("PYTHONPATH", "")
    env_da["PYTHONPATH"] = (env_da["PYTHONPATH"] + os.pathsep if env_da["PYTHONPATH"] else "") + da_py_path

    run([
        "python",
        str(base_dir / "third_party" / "mega-sam" / "Depth-Anything" / "run_videos.py"),
        "--encoder", "vitl",
        "--load-from", str(base_dir / "third_party" / "mega-sam" / "Depth-Anything" / "checkpoints" / "depth_anything_vitl14.pth"),
        "--img-path", str(frame_dir),
        "--outdir", str(depth_anything_dir)
    ], env_da)

    # 2) UniDepth
    env_uni = env_base.copy()
    uni_py_path = str(base_dir / "third_party" / "mega-sam" / "UniDepth")
    env_uni["PYTHONPATH"] = env_uni.get("PYTHONPATH", "")
    env_uni["PYTHONPATH"] = (env_uni["PYTHONPATH"] + os.pathsep if env_uni["PYTHONPATH"] else "") + uni_py_path

    run([
        "python",
        str(base_dir / "third_party" / "mega-sam" / "UniDepth" / "scripts" / "demo_mega-sam.py"),
        "--img-path", str(frame_dir),
        "--outdir", str(uni_depth_dir)
    ], env_uni)

    # 3) Camera Tracking
    env_track = env_base.copy()
    base_path = str(base_dir / "third_party" / "mega-sam" / "base")
    droid_path = str(base_dir / "third_party" / "mega-sam" / "base" / "droid_slam")
    parts = [p for p in [env_track.get("PYTHONPATH", ""), base_path, droid_path] if p]
    env_track["PYTHONPATH"] = os.pathsep.join(parts)

    ckpt_path = base_dir / "third_party" / "mega-sam" / "checkpoints" / "megasam_final.pth"
    run([
        "python",
        str(base_dir / "third_party" / "mega-sam" / "camera_tracking_scripts" / "test_demo.py"),
        "--datapath=" + str(frame_dir),
        "--weights=" + str(ckpt_path),
        "--output_path", str(recon_dir),
        "--mono_depth_path", str(depth_anything_dir),
        "--metric_depth_path", str(uni_depth_dir),
        "--disable_vis"
    ], env_track)

    # 4) RAFT Optical Flow
    env_raft = env_base.copy()
    raft_core_path = str(base_dir / "third_party" / "mega-sam" / "cvd_opt" / "core")
    env_raft["PYTHONPATH"] = env_raft.get("PYTHONPATH", "")
    env_raft["PYTHONPATH"] = (env_raft["PYTHONPATH"] + os.pathsep if env_raft["PYTHONPATH"] else "") + raft_core_path

    run([
        "python",
        str(base_dir / "third_party" / "mega-sam" / "cvd_opt" / "preprocess_flow.py"),
        "--datapath=" + str(frame_dir),
        "--model=" + str(base_dir / "third_party" / "mega-sam" / "cvd_opt" / "raft-things.pth"),
        "--output_path", str(recon_dir),
        "--mixed_precision"
    ], env_raft)

    # 5) CVD optimization
    env_cvd = env_base.copy()
    run([
        "python",
        str(base_dir / "third_party" / "mega-sam" / "cvd_opt" / "cvd_opt.py"),
        "--output_dir", str(recon_dir),
        "--data_dir", str(recon_dir),
        "--image_dir", str(data_dir / "resized_images"),
        "--w_grad", "2.0",
        "--w_normal", "5.0"
    ], env_cvd)

    # 6) Saving scene.pkl
    recon_npz_path = recon_dir / "sgd_cvd_hr.npz"
    recon_npz = np.load(recon_npz_path)
    scene_path = data_dir / "scene" / "scene.pkl"
    saved_dict = {
        "images": recon_npz["images"], # [N, H, W, 3], uint8
        "depths": recon_npz["depths"], # [N, H, W], float32
        "intrinsics": recon_npz["intrinsic"], # [3, 3], float32
        "extrinsics": recon_npz["cam_c2w"], # [N, 4, 4], float32 camera to world transform
        "n_frames": recon_npz["images"].shape[0], # N
        "height": recon_npz["images"].shape[1],
        "width": recon_npz["images"].shape[2]
    }
    with open(scene_path, "wb") as f:
        pickle.dump(saved_dict, f)
    
    print(f"[Visualization] Saving dynamic PCD HTML")
    from utils.viz_dynamic_pcd import save_dynamic_pcd
    save_dynamic_pcd(video_name, max_points=5000)

    print(f"[Done] {video_name}")

def run_moge(key_name: str, key_cfgs: dict):
    """
    Run MoGe-2 depth predictor on a single RGB image (uint8 HxWx3).
    Returns float32 depth (H,W) in arbitrary metric (to be aligned).
    """
    import torch
    from moge.model.v2 import MoGeModel
    device = torch.device(f"cuda:{key_cfgs['gpu']}")
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device).eval()
    scene_path = Path("outputs") / key_name / "scene" / "scene.pkl"
    with open(scene_path, "rb") as f:
        scene_data = pickle.load(f)
    images = scene_data["images"] # [1, H, W, 3], uint8
    assert images.shape[0] == 1, "MoGe-2 only supports single image"
    image = images[0]  # [H, W, 3], uint8
    with torch.no_grad():
        inp = torch.tensor(image / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
        out = model.infer(inp[0])
        depth = out["depth"].detach().float().cpu().numpy() # [H, W], float32
        intrinsics = out["intrinsics"].detach().float().cpu().numpy() # [3, 3], float32
        fx = intrinsics[0,0]
        fy = intrinsics[1,1]
        cx = intrinsics[0,2]
        cy = intrinsics[1,2]
        H, W = image.shape[0], image.shape[1]
        intrinsics = np.array(
            [fx*W, 0, cx*W,
             0, fy*H, cy*H,
             0, 0, 1], dtype=np.float32
        ).reshape(3,3)
        
        saved_dict = {
            "images": images, # [1, H, W, 3], uint8
            "depths": depth[np.newaxis, ...], # [1, H, W], float32
            "intrinsics": intrinsics, # [3, 3], float32
            "extrinsics": np.eye(4, dtype=np.float32)[np.newaxis, ...], # [1, 4, 4] camera to world transform
            "n_frames": 1,
            "height": images.shape[1],
            "width": images.shape[2]
        }
        with open(scene_path, "wb") as f:
            pickle.dump(saved_dict, f)

    print(f"[Visualization] Saving dynamic PCD HTML")
    from utils.viz_dynamic_pcd import save_dynamic_pcd
    save_dynamic_pcd(key_name, max_points=None)

    print(f"[Done] {key_name}")
    return

def mode_check(key_name: str) -> str:
    scene_path = Path("outputs") / key_name / "scene" / "scene.pkl"
    with open(scene_path, "rb") as f:
        scene_data = pickle.load(f)
    n_frames = scene_data["n_frames"]
    if n_frames == 1:
        return "image"
    else:
        return "video"

def main(config_file: str = "config/config.yaml", key_name: str = None):
    """Main function: load config and run depth prediction."""
    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)

    if key_name is not None:
        keys = [key_name]
    else:
        keys = [k for k in cfg["keys"]]

    for key in keys:
        mode = mode_check(key)
        key_cfgs = compose_configs(key, cfg)
        if mode == "video":
            print(f"[Info] Running mega-sam for video: {key}")
            run_megasam(key, key_cfgs)
        else:
            print(f"[Info] Running moge for image: {key}")
            run_moge(key, key_cfgs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--key_name", type=str, default=None, help="If set, run only this key")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="YAML with keys: [lab1, ...]")
    args = parser.parse_args()

    main(args.config, args.key_name)