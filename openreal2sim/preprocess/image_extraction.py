#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
    
    saved_dict = {
        "images": frames,
        "n_frames": n_frames,
        "height": frames.shape[1],
        "width": frames.shape[2]
    }
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
    mode = []
    if png_path.is_file() or jpg_path.is_file():
        mode.append("image")
    if video_path.is_file():
        mode.append("video")
    return mode

def main(config_file: str = "config/config.yaml"):
    """Main function: load config and process videos."""
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    keys = config["keys"]

    for key_name in keys:
        key_config = compose_configs(key_name, config)
        key_modes = mode_check(key_name)
        if "video" in key_modes:
            run_ffmpeg(key_name, key_config)
        if "image" in key_modes:
            run_image(key_name, key_config)
        run_collect_info(key_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml", help="YAML with keys: [lab1, ...]")
    args = parser.parse_args()

    main(args.config)
