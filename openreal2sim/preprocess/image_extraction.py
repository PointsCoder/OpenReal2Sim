#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# r3d extraction refers to SPOT.

"""
Extract frames from images or videos, reading configs from YAML file.
Inputs:
    - data/{key_name}.png or .jpg (if present).
    - data/{key_name}.mp4 (if present).
Outputs:
    - outputs/{key_name}/images/frame_00000.jpg, frame_00001.jpg, ...
    - outputs/{key_name}/scene/scene.pkl (saving frames, depths, and camera infos)
Parameters:
    - fps: extraction frame rate for videos.
    - quality: JPEG quality for extracted frames, 1(best)..31(worst).
    - resize: resize width and height for both images and videos.
"""

import os
import argparse
import subprocess
import yaml
from pathlib import Path
from typing import Tuple
from PIL import Image
import json
import numpy as np
import pickle
import cv2
import liblzfse
from zipfile import ZipFile
import shutil
from scipy.spatial.transform import Rotation
from scipy.ndimage import distance_transform_edt
import torch
from utils.compose_config import compose_configs

def read_image_size(image_path) -> Tuple[int, int]:
    with Image.open(image_path) as im:
        return im.width, im.height

def run_image(key_name: str, cfgs: dict):
    """
    Convert data/{key_name}.png or .jpg to outputs/.../frame_00000.jpg.
    If frame_00000.jpg already exists (e.g. from video), overwrite it with the same resolution. 
    Always store at highest JPEG quality.
    """
    input_path = Path("data") / f"{key_name}.png" if (Path("data") / f"{key_name}.png").is_file() else Path("data") / f"{key_name}.jpg"
    output_dir = Path("outputs") / key_name / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "frame_00000.jpg"
    width, height = read_image_size(input_path)
    width, height = int(width * cfgs["resize"]), int(height * cfgs["resize"])
    if output_path.is_file():
        # If frame_00000.jpg already exists from video extraction, resize to match its size
        existing_width, existing_height = read_image_size(output_path)
        width, height = existing_width, existing_height
    print(f"[Info] Resizing image {input_path} to {width}x{height} and saving to {output_path}")

    with Image.open(input_path) as im:
        rgb = im.convert("RGB")        
        resized = rgb.resize((width, height), Image.Resampling.LANCZOS)
        resized.save(
            output_path,
            format="JPEG",
            quality=100,
            subsampling=0,
            optimize=True
        )

def get_video_resolution(video_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env={**os.environ, "LD_LIBRARY_PATH": ""})
    info = json.loads(result.stdout)
    
    width = info["streams"][0]["width"]
    height = info["streams"][0]["height"]
    return width, height

def load_depth(filepath, H, W):
    """Load depth data from compressed file and fill NaNs by nearest valid neighbor."""
    with open(filepath, 'rb') as depth_fh:
        raw_bytes = depth_fh.read()
        decompressed_bytes = liblzfse.decompress(raw_bytes)
        depth_img = np.frombuffer(decompressed_bytes, dtype=np.float32)
        depth_img = depth_img.reshape((H, W))
    # Normalize infinities to NaN so they get filled as well
    depth_img = np.where(np.isfinite(depth_img), depth_img, np.nan).astype(np.float32)

    mask = np.isnan(depth_img)
    # If there are no NaNs, return directly
    if not np.any(mask):
        return depth_img

    # If everything is NaN, return zeros to avoid crashes downstream
    if np.all(mask):
        return np.zeros_like(depth_img, dtype=np.float32)

    # Use distance transform to find nearest valid pixel for each NaN and fill with its value
    # distance_transform_edt(mask, return_indices=True) returns indices of the nearest False
    # element (i.e., nearest valid depth) for each position.
    inds = distance_transform_edt(mask, return_indices=True, return_distances=False)
    filled = np.copy(depth_img)
    filled[mask] = depth_img[inds[0][mask], inds[1][mask]]
    return filled.astype(np.float32)
 

def judge_picture_side(depth: np.ndarray) -> str:
    h, w = depth.shape
    
    mid_point = h // 2
    upper_half = np.nan_to_num(depth[:mid_point, :], nan=0)
    lower_half = np.nan_to_num(depth[mid_point:, :], nan=0)

    upper_avg_depth = np.mean(upper_half)
    lower_avg_depth = np.mean(lower_half)
    #import pdb; pdb.set_trace()
    if upper_avg_depth > lower_avg_depth:
        return False
    else:
        print(f"[Info] Detected upside-down image based on depth analysis. Flip it.")
        return True


def to_tensor_func(arr):
    import torch
    from copy import deepcopy
    if arr.ndim == 2: 
        arr = arr[:, :, np.newaxis]
    arr = deepcopy(arr)
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def run_r3d(key_name: str, cfgs: dict):
    ### FIXME: SOMETIMES
    """Extract frames and depth from R3D file."""
    import sys
    repo_dir = Path.cwd() / "third_party/PromptDA"
    sys.path.append(str(repo_dir))
    from promptda.promptda import PromptDA
    from promptda.utils.io_wrapper import load_image
    import torch
    device = torch.device(f"cuda:{cfgs['gpu']}")
    model = PromptDA.from_pretrained("depth-anything/prompt-depth-anything-vitl").to(device).eval()
    input_path = Path("data") / f"{key_name}.r3d"
    output_dir = Path("outputs") / key_name / "images"
    depth_dir = Path("outputs") / key_name / "depth"

    output_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = Path("outputs") / key_name / "temp_r3d"
    temp_dir.mkdir(parents=True, exist_ok=True)
    interval = cfgs.get("fps", 1)
    try:
        print("[Info] Extracting R3D file...")
        with ZipFile(input_path) as zip_file:
            zip_file.extractall(temp_dir)
        
        metadata_path = temp_dir / 'metadata'
        with open(metadata_path, "rb") as f:
            metadata = json.loads(f.read())
        fps = metadata.get("fps", 1)
        rgbd_dir = temp_dir / 'rgbd'
        rgb_path_list = list(rgbd_dir.glob('*.jpg'))
        rgb_path_list.sort(key = lambda x: int(x.stem))
        interval = int(fps / cfgs.get("fps", 1))
        print(f"[Info] Interval: {interval}")
        
        H, W = cv2.imread(str(rgb_path_list[0])).shape[:2]
        if H < W:
            H_dc, W_dc = 192, 256 
        else:
            H_dc, W_dc = 256, 192
                
        resize_factor = cfgs.get("resize", 1.0)
        new_width = int(W * resize_factor)
        new_height = int(H * resize_factor)
        flip = False
        depth0 = rgb_path_list[0].with_suffix('.depth')
        depth0 = load_depth(str(depth0), H_dc, W_dc)
        if judge_picture_side(depth0):
            flip = True
        images = []
        import tqdm
      
        for frame_idx in tqdm.tqdm(range(0, len(rgb_path_list), interval), desc="Loading frames"):
            
            rgb_path = rgb_path_list[frame_idx]

            rgb_output_path = output_dir / f"frame_{frame_idx:05d}.jpg"
            
            depth_path = rgb_path.with_suffix('.depth')
            depth_output_path = Path("data") / f"{key_name}_depth.png"
            image = load_image(str(rgb_path), new_width=new_width, new_height=new_height).to(device)
            if flip:
                image = torch.flip(image, dims=[2,3])
            
            if  frame_idx == 0:
                depth = load_depth(str(depth_path), H_dc, W_dc) 
                if flip:   
                    depth = depth[::-1, ::-1]
                depth = to_tensor_func(depth).to(device)
                depth = model.predict(image, depth).cpu().numpy()[0,0]
                cv2.imwrite(str(depth_output_path), (depth * 1000.).astype(np.uint16))
            image = image.cpu().numpy()[0].transpose(1, 2, 0)*255.0
            images.append(image.astype(np.uint8))
            image = Image.fromarray(image.astype(np.uint8))
            image.save(rgb_output_path, format="JPEG", quality=100, subsampling=0, optimize=True)
            new_height = image.height
            new_width = image.width
          
        video_writer = cv2.VideoWriter(str(output_dir / "video.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), 30, (new_width, new_height))
        for image in images:
            video_writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        video_writer.release()
 
    except Exception as e:
        print(f"[Error] Error processing R3D file: {e}")
        return
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    



def run_ffmpeg(key_name: str, cfgs: dict):
    """Run ffmpeg to extract frames from one video, scaling to reference image size."""
    input_path = Path("data") / f"{key_name}.mp4"
    output_dir = Path("outputs") / key_name / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    width, height = get_video_resolution(input_path)
    width, height = int(width * cfgs["resize"]), int(height * cfgs["resize"])
    print(f"[Info] Resizing video {input_path} to {width}x{height} and saving frames to {output_dir}")

    # if image and video both exist, resize video frames to match the image size
    if Path(f"data/{key_name}.jpg").is_file() or Path(f"data/{key_name}.png").is_file():
        ref_image_path = Path("data") / f"{key_name}.png" if (Path("data") / f"{key_name}.png").is_file() else Path("data") / f"{key_name}.jpg"
        ref_width, ref_height = read_image_size(ref_image_path)
        ref_width, ref_height = int(ref_width * cfgs["resize"]), int(ref_height * cfgs["resize"])
        width, height = ref_width, ref_height
        print(f"[Info] Found reference image {ref_image_path}, resizing video frames to match its size: {width}x{height}")

    # Build ffmpeg filter chain: fps + scale to match the reference image
    vf = f"fps={cfgs['fps']},scale={width}:{height}:flags=lanczos"

    # ffmpeg command:
    cmd = [
        "ffmpeg",
        "-y",  # overwrite without prompt
        "-i", str(input_path),
        "-vf", vf,
        "-q:v", str(cfgs["quality"]),          # 1(best)..31(worst) for mjpeg
        "-start_number", "0",          # start from frame_00000.jpg
        str(output_dir / "frame_%05d.jpg"),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env={**os.environ, "LD_LIBRARY_PATH": ""})
    print(f"[Info] Wrote frames from video: {input_path}")

def run_collect_info(key_name: str):
    data_dir = Path("outputs") / key_name / "images"
    depth_dir = Path("outputs") / key_name / "depth"
    scene_dir = Path("outputs") / key_name / "scene"
    scene_dir.mkdir(parents=True, exist_ok=True)
    scene_path = scene_dir / "scene.pkl"
   
    frame_files = sorted(data_dir.glob("frame_*.jpg"), key=lambda x: int(x.stem.split("_")[1]))
    frames = []
    for f in frame_files:
        img = Image.open(f).convert("RGB")
        frames.append(np.array(img, dtype=np.uint8))  # [H, W, 3]
    frames = np.stack(frames, axis=0)    
    n_frames = len(frame_files)
    
    if depth_dir.exists():
        depth_files = sorted(depth_dir.glob("frame_*.png"), key=lambda x: int(x.stem.split("_")[1]))
        if len(depth_files) > 0:
            depths = []
            for f in depth_files:
                depth = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
                depths.append(depth / 1000.0)
            depths = np.stack(depths, axis=0).astype(np.float32)
        else:
            depths = None
    else:
        depths = None 

   

    saved_dict = {}
    if scene_path.exists():
        with open(scene_path, "rb") as f:
            saved_dict = pickle.load(f)
    
    # Update with image information
    saved_dict.update({
        "images": frames,
        "depths": depths,
        "n_frames": n_frames,
        "height": frames.shape[1],
        "width": frames.shape[2]
    })
    
    with open(scene_path, "wb") as f:
        pickle.dump(saved_dict, f)

    print(f"[Info] Wrote geometry information to: {scene_path}")

def mode_check(key_name: str) -> Path:
    """
    Check which input modes are available for this key_name.
    """
    png_path = Path("data") / f"{key_name}.png"
    jpg_path = Path("data") / f"{key_name}.jpg"
    video_path = Path("data") / f"{key_name}.mp4"
    r3d_path = Path("data") / f"{key_name}.r3d"
    mode = []
    if png_path.is_file() or jpg_path.is_file():
        mode.append("image")
    if video_path.is_file():
        mode.append("video")
    if r3d_path.is_file():
        mode.append("r3d")
    return mode

def main(config_file: str = "config/config.yaml", key_name: str = None):
    """Main function: load config and process videos."""
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    keys = config["keys"]
    if key_name is not None:
        keys = [key_name]

    for key_name in keys:
        key_config = compose_configs(key_name, config)
        key_modes = mode_check(key_name)
        if "video" in key_modes:
            run_ffmpeg(key_name, key_config)
        if "image" in key_modes:
            run_image(key_name, key_config)
        if "r3d" in key_modes:
            run_r3d(key_name, key_config)
        run_collect_info(key_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--key_name", type=str, default=None, help="If set, run only this key")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="YAML with keys: [lab1, ...]")
    args = parser.parse_args()

    main(args.config, args.key_name)
