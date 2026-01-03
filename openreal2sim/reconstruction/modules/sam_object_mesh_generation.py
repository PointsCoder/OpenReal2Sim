#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate textured meshes for each segmented object using SAM-3D.
Step 1: Extract object crops with alpha-channel from frame-0 & masks,
        saving them to outputs/{key}/sam-3d/ folder.
Step 2: Run SAM-3D inference to generate GLB meshes.

Inputs:
    - outputs/{key}/scene/scene.pkl (must contain the "mask" key)
Outputs:
    - outputs/{key}/sam-3d/{idx}.png (object masked RGBA image, idx is 0-based)
    - outputs/{key}/sam-3d/image.png (original RGB image)
    - outputs/{key}/sam-3d/{oid}_{name}.glb (generated 3D mesh)
    - outputs/{key}/scene/scene.pkl (updated with "objects" key)
Note:
    - added key "objects": {
            "oid": {
                "oid":   # object id,
                "name":  # object name,
                "glb":   # object glb path,
                "mask":  # object mask [H, W] boolean array,
            },
            ...
        }
"""

import os
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

base_dir = Path.cwd()
output_dir = base_dir / "outputs"

# Add SAM-3D to path
sam3d_dir = str(base_dir / "third_party/sam-3d-objects/notebook")
sys.path.insert(0, sam3d_dir)

# ------------------------------------------------------------------


def rgb_with_transparency(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Create an RGBA image where only the masked region is visible.

    img  : (H, W, 3) RGB array
    mask : (H, W) boolean or uint8 mask

    Returns: (H, W, 4) RGBA array with transparency outside the mask
    """
    mask = mask.astype(bool)

    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)

    h, w = mask.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)
    out[mask, :3] = img[mask]
    out[mask, 3] = 255

    return out


def load_obj_masks(data: dict):
    """
    Return object list for frame-0:
        [{'mask': bool array, 'name': name, 'oid': int}, ...]
    Filter out names: 'ground' / 'hand' / 'robot'
    """
    frame_objs = data.get(0, {})  # only frame 0
    objs = []
    for oid, item in frame_objs.items():
        lbl = item["name"]
        if lbl in ("ground", "hand", "robot"):
            continue
        objs.append({
            "oid": oid,
            "mask": item["mask"].astype(bool),
            "name": lbl,
        })
    # Sort by object id
    objs.sort(key=lambda x: int(x["oid"]))
    return objs


def create_sam_3d_images(keys, key_scene_dicts):
    """
    Create masked RGBA images for SAM-3D processing.
    Saves images to outputs/{key}/sam-3d/ folder.

    Args:
        keys: list of video keys to process
        key_scene_dicts: dict mapping keys to their scene dictionaries

    Returns:
        dict mapping keys to dict with:
            - "sam_3d_dir": Path to sam-3d folder
            - "objects": list of object info dicts with oid, name, mask, idx
    """
    key_info = {}

    for key in keys:
        print(f"[Info] Creating SAM-3D images for {key}...")
        scene_dict = key_scene_dicts[key]
        objs = load_obj_masks(scene_dict["mask"])

        # Create output directory: outputs/{key}/sam-3d/
        sam_3d_dir = output_dir / key / "sam-3d"
        sam_3d_dir.mkdir(parents=True, exist_ok=True)

        # Get original image from frame 0
        image = scene_dict["images"][0]

        # Save original image
        original_img_path = sam_3d_dir / "image.png"
        Image.fromarray(image).save(original_img_path)
        print(f"[Info] [{key}] Saved original image → {original_img_path}")

        obj_info_list = []

        # Create masked RGBA image for each object
        for idx, item in enumerate(objs):
            oid = item["oid"]
            mask = item["mask"]
            name = item["name"]

            # Create RGBA image with transparency
            rgba = rgb_with_transparency(image, mask)

            # Save as {idx}.png (0-based index for SAM-3D load_masks)
            png_path = sam_3d_dir / f"{idx}.png"
            Image.fromarray(rgba).save(png_path)
            print(f"[Info] [{key}] Saved mask for '{name}' (oid={oid}) → {png_path}")

            obj_info_list.append({
                "idx": idx,
                "oid": oid,
                "name": name,
                "mask": mask,
            })

        key_info[key] = {
            "sam_3d_dir": sam_3d_dir,
            "objects": obj_info_list,
        }
        print(f"[Info] [{key}] SAM-3D images created: {len(objs)} objects")

    return key_info


def sam_object_mesh_generation(keys, key_scene_dicts, key_cfgs):
    """
    Main entry point for SAM-3D based object mesh generation.

    Args:
        keys: list of video keys to process
        key_scene_dicts: dict mapping keys to their scene dictionaries
        key_cfgs: dict mapping keys to their configurations

    Returns:
        Updated key_scene_dicts
    """
    # Step 1: Create SAM-3D input images
    key_info = create_sam_3d_images(keys, key_scene_dicts)

    # Step 2: Initialize SAM-3D inference pipeline
    from inference import Inference, load_image, load_masks

    config_path = str(base_dir / "third_party/sam-3d-objects/checkpoints/hf/pipeline.yaml")
    print(f"[Info] Loading SAM-3D inference pipeline from {config_path}...")
    inference = Inference(config_path, compile=False)
    print("[Info] SAM-3D inference pipeline loaded.")

    # Step 3: Run inference and export GLBs for each key
    for key in keys:
        print(f"[Info] Processing SAM-3D inference for {key}...")
        scene_dict = key_scene_dicts[key]
        info = key_info[key]
        sam_3d_dir = info["sam_3d_dir"]
        obj_info_list = info["objects"]

        if len(obj_info_list) == 0:
            print(f"[Info] [{key}] No objects to process, skipping.")
            continue

        # Load image and masks using SAM-3D utilities
        image_path = sam_3d_dir / "image.png"
        image = load_image(str(image_path))
        masks = load_masks(str(sam_3d_dir), extension=".png")

        # Run inference for each mask
        print(f"[Info] [{key}] Running inference on {len(masks)} objects...")
        outputs = [inference(image, mask, seed=42) for mask in masks]

        # Export GLBs and build object metadata
        object_meta = {}
        for idx, (obj_info, output) in enumerate(zip(obj_info_list, outputs)):
            oid = obj_info["oid"]
            name = obj_info["name"]
            mask = obj_info["mask"]
            stem = f"{oid}_{name}"

            glb = output.get("glb", None)
            glb_path = sam_3d_dir / f"{stem}.glb"

            if glb is not None:
                glb.export(str(glb_path))
                print(f"[Info] [{key}] Saved GLB: {glb_path}")
            else:
                print(f"[Warn] [{key}] No GLB in output for {stem}, skipping export.")
                glb_path = None

            # Save mask as image for reference
            mask_png = sam_3d_dir / f"{stem}_mask.jpg"
            Image.fromarray(mask.astype(np.uint8) * 255).save(mask_png)

            object_meta[oid] = {
                "oid": oid,
                "name": name,
                "glb": str(glb_path) if glb_path else None,
                "mask": mask,
            }

        # Step 4: Update scene_dict
        scene_dict["objects"] = object_meta
        key_scene_dicts[key] = scene_dict

        # Save updated scene.pkl
        scene_pkl_path = base_dir / f"outputs/{key}/scene/scene.pkl"
        with open(scene_pkl_path, "wb") as f:
            pickle.dump(scene_dict, f)
        print(f"[Info] [{key}] scene_dict updated and saved.")

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
        scene_pkl = base_dir / f"outputs/{key}/scene/scene.pkl"
        with open(scene_pkl, "rb") as f:
            scene_dict = pickle.load(f)
        key_scene_dicts[key] = scene_dict

    sam_object_mesh_generation(keys, key_scene_dicts, key_cfgs)

