#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SAM-3D inference script - runs inside sam3d container.
Called from openreal2sim container via docker exec/run.

This script handles the GPU-intensive SAM-3D inference, loading the model
and generating GLB meshes from masked images.

Usage:
    python sam_object_mesh_inference.py --sam_3d_dir /app/outputs/scene_00002/sam-3d
"""

import argparse
import sys
from pathlib import Path

base_dir = Path("/app")
sam3d_dir = str(base_dir / "third_party/sam-3d-objects/notebook")
sys.path.insert(0, sam3d_dir)


def run_inference(sam_3d_dir: str, config_path: str):
    """
    Run SAM-3D inference on prepared images.
    
    Args:
        sam_3d_dir: Path to directory containing image.png and mask files (0.png, 1.png, ...)
        config_path: Path to SAM-3D pipeline config YAML
    
    Outputs:
        Saves output_{idx}.glb files to sam_3d_dir for each mask
    """
    from inference import Inference, load_image, load_masks
    
    sam_3d_dir = Path(sam_3d_dir)
    
    print(f"[Info] Loading SAM-3D inference pipeline from {config_path}...")
    inference = Inference(config_path, compile=False)
    print("[Info] SAM-3D inference pipeline loaded.")
    
    image_path = sam_3d_dir / "image.png"
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = load_image(str(image_path))
    masks = load_masks(str(sam_3d_dir), extension=".png")
    
    if len(masks) == 0:
        print("[Warn] No masks found, nothing to process.")
        return
    
    print(f"[Info] Running inference on {len(masks)} objects...")
    
    for idx, mask in enumerate(masks):
        print(f"[Info] Processing mask {idx}/{len(masks)-1}...")
        output = inference(image, mask, seed=42)
        
        glb = output.get("glb", None)
        if glb is not None:
            glb_path = sam_3d_dir / f"output_{idx}.glb"
            glb.export(str(glb_path))
            print(f"[Info] Saved GLB: {glb_path}")
        else:
            print(f"[Warn] No GLB in output for mask {idx}")
    
    print("[Info] SAM-3D inference completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM-3D inference for object mesh generation")
    parser.add_argument(
        "--sam_3d_dir", 
        type=str, 
        required=True,
        help="Path to directory containing image.png and mask PNGs"
    )
    parser.add_argument(
        "--config_path", 
        type=str,
        default="/app/third_party/sam-3d-objects/checkpoints/hf/pipeline.yaml",
        help="Path to SAM-3D pipeline config"
    )
    args = parser.parse_args()
    
    run_inference(args.sam_3d_dir, args.config_path)
