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
import json
import sys
from pathlib import Path

import numpy as np

base_dir = Path("/app")
sam3d_dir = str(base_dir / "third_party/sam-3d-objects/notebook")
sys.path.insert(0, sam3d_dir)

# The GLB that SAM-3D exports is canonical and Y-up: its vertices are rotated
# from the Z-up model frame by `v_yup = v_zup @ GLB_YUP_TO_ZUP` (see
# sam3d_objects/model/backbone/tdfy_dit/utils/postprocessing_utils.py). The
# matrix below is the column-vector form of the inverse, Y-up back to Z-up.
GLB_YUP_TO_ZUP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

# The predicted pose places the object in the PyTorch3D camera frame (X left,
# Y up, Z forward), while the rest of the pipeline uses OpenCV (X right, Y
# down, Z forward). This is the inverse of the rotation of
# camera_to_pytorch3d_camera in sam3d_objects/pipeline/inference_pipeline_pointmap.py.
PYTORCH3D_TO_OPENCV = np.diag([-1.0, -1.0, 1.0])


def glb_to_camera_transform(output):
    """
    Build the 4x4 transform that takes the exported GLB vertices into the
    OpenCV camera frame, from the layout pose SAM-3D predicts next to the mesh.

    The composition mirrors the one upstream uses to assemble a scene
    (notebook/inference.py, make_scene), so the PyTorch3D row-vector
    convention is handled by PyTorch3D itself and only the result is
    transposed into the column-vector convention the pipeline uses.
    """
    import torch
    from pytorch3d.transforms import quaternion_to_matrix
    from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform

    rotation = output["rotation"].detach().cpu().float().reshape(1, 4)
    translation = output["translation"].detach().cpu().float().reshape(1, 3)
    scale = output["scale"].detach().cpu().float().reshape(-1)
    if scale.numel() == 1:
        scale = scale.expand(3)
    scale = scale.reshape(1, 3)

    local_to_camera = compose_transform(
        scale=scale,
        rotation=quaternion_to_matrix(rotation),
        translation=translation,
    )
    matrix_row = local_to_camera.get_matrix()[0].detach().cpu().numpy().astype(np.float64)

    linear = PYTORCH3D_TO_OPENCV @ matrix_row[:3, :3].T @ GLB_YUP_TO_ZUP
    offset = PYTORCH3D_TO_OPENCV @ matrix_row[3, :3]

    transform = np.eye(4)
    transform[:3, :3] = linear
    transform[:3, 3] = offset

    determinant = float(np.linalg.det(transform[:3, :3]))
    if determinant <= 0.0:
        raise ValueError(
            f"SAM-3D layout transform is not a scaled rotation, determinant {determinant}"
        )

    return transform, torch.as_tensor(scale).reshape(-1).tolist()


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

            # Persist the predicted layout pose next to the mesh. Without it
            # the canonical GLB carries no orientation and the registration
            # downstream has to guess one.
            try:
                transform, scale = glb_to_camera_transform(output)
            except Exception as error:
                print(f"[Warn] Could not build the layout transform for mask {idx}: {error}")
            else:
                pose = {
                    "glb_to_camera": transform.tolist(),
                    "rotation_wxyz": output["rotation"].detach().cpu().reshape(-1).tolist(),
                    "translation": output["translation"].detach().cpu().reshape(-1).tolist(),
                    "scale": scale,
                    "frame": "opencv_camera",
                }
                pose_path = sam_3d_dir / f"output_{idx}_pose.json"
                pose_path.write_text(json.dumps(pose, indent=2))
                print(f"[Info] Saved layout pose: {pose_path}")
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
