#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch object removal and background inpainting with ObjectClear on the first frame of each key.
Inputs:
    - outputs/{key_name}/scene/scene.pkl (must contain a "mask" key with mask dict)
Outputs:
    - outputs/{key_name}/scene/scene.pkl (updated with a "recon" key containing inpainting results)
    - outputs/{key_name}/reconstruction/foreground.jpg (copy of the first frame)
    - outputs/{key_name}/reconstruction/background.jpg (inpainted background image)
    - outputs/{key_name}/reconstruction/object_mask.jpg (object mask)
    - outputs/{key_name}/reconstruction/ground_mask.jpg (ground mask)
Note:
    - added keys in "recon": "background", "foreground", "ground_mask", "object_mask"
"""


import os
import glob
import torch
from pathlib import Path
import sys
from PIL import Image
import numpy as np
import pickle
import cv2
import yaml

# ---------------- basic paths ----------------
base_dir = Path.cwd()
repo_dir = str(base_dir / 'third_party/ObjectClear')
sys.path.append(repo_dir)

from objectclear.pipelines import ObjectClearPipeline
from objectclear.utils import resize_by_short_side

def background_pixel_inpainting(keys, key_scene_dicts, key_cfgs):
    
    # hyperparameters
    USE_FP16 = True
    SEED = 42
    NUM_STEPS = 20
    STRENGTH = 0.99
    GUIDANCE_SCALE = 2.5

    # Set up ObjectClear pipeline once
    torch_dtype = torch.float16 if USE_FP16 else torch.float32
    variant = "fp16" if USE_FP16 else None
    gpu_id = key_cfgs[keys[0]]["gpu"] # it has to be running on the same GPU
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    generator = torch.Generator(device=device).manual_seed(SEED)
    pipe = ObjectClearPipeline.from_pretrained_with_custom_modules(
        "jixin0101/ObjectClear",
        torch_dtype=torch_dtype,
        apply_attention_guided_fusion=True,
        cache_dir=None,
        variant=variant,
    ).to(device)

    for key in keys:
        # ---------- paths per key ----------
        scene_dict = key_scene_dicts[key]
        input_image = scene_dict["images"][0]  # H x W x 3, uint8
        recon_dir = base_dir / f'outputs/{key}/reconstruction'
        recon_dir.mkdir(parents=True, exist_ok=True)
        pil_image = Image.fromarray(input_image)
        input_path = recon_dir / "foreground.jpg"
        pil_image.save(input_path, format="JPEG", quality=100)  # save foreground.jpg to recon/

        mask_img_path = str(recon_dir / 'object_mask.jpg')
        ground_mask_path = str(recon_dir / 'ground_mask.jpg')

        # ---------- build masks from mask_dict.pkl (frame 0) ----------
        mask_dict = scene_dict.get("mask", None)
        if mask_dict is None:
            print(f'[Error] No masks found in {key}/scene/scene.pkl, please run segmentation_annotator.py first!')
            continue

        frame_id = 0
        frame_objs = mask_dict.get(frame_id, {})

        mask_accum = None
        ground_accum = None

        for oid, obj in frame_objs.items():
            name = obj["name"].lower()
            mask = obj["mask"]  # bool array

            if "ground" not in name:
                if mask_accum is None:
                    mask_accum = np.zeros_like(mask, dtype=bool)
                mask_accum |= mask
            else:
                if ground_accum is None:
                    ground_accum = np.zeros_like(mask, dtype=bool)
                ground_accum |= mask

        if mask_accum is not None:
            cv2.imwrite(mask_img_path, (mask_accum.astype(np.uint8)) * 255)

        if ground_accum is not None:
            cv2.imwrite(ground_mask_path, (ground_accum.astype(np.uint8)) * 255)
        else:
            print(f'[Warning] No ground mask found in {key}/scene/scene.pkl! This may affect gravity alignment later.')

        # ---------- single image + single mask ----------
        input_img_list = [input_path]
        input_mask_list = [mask_img_path]
        result_root = str(recon_dir)
        os.makedirs(result_root, exist_ok=True)

        # ---------- run ObjectClear ----------
        for i, (img_path, m_path) in enumerate(zip(input_img_list, input_mask_list)):
            img_name = os.path.basename(img_path)
            print(f'[{key}] [{i+1}/{len(input_img_list)}] Processing: {img_name}')

            image = Image.open(img_path).convert("RGB")
            mask = Image.open(m_path).convert("L")
            image_orig = image.copy()

            # Model was trained on 512 short side
            image = resize_by_short_side(image, 512, resample=Image.BICUBIC)
            mask = resize_by_short_side(mask, 512, resample=Image.NEAREST)

            w, h = image.size

            result = pipe(
                prompt="remove the instance of object",
                image=image,
                mask_image=mask,
                generator=generator,
                num_inference_steps=NUM_STEPS,
                strength=STRENGTH,
                guidance_scale=GUIDANCE_SCALE,
                height=h,
                width=w,
                return_attn_map=False,
            )

            fused_img_pil = result.images[0]
            fused_img_pil = fused_img_pil.resize(image_orig.size)

            # Save as background.jpg in the scene folder
            save_path = os.path.join(result_root, 'background.jpg')
            fused_img_pil.save(save_path)
            
        # Save in scene_dict
        if "recon" not in scene_dict:
            scene_dict["recon"] = {}
        scene_dict["recon"]["background"] = np.ascontiguousarray(np.array(fused_img_pil, dtype=np.uint8))
        scene_dict["recon"]["foreground"] = np.ascontiguousarray(np.array(image_orig,   dtype=np.uint8))
        scene_dict["recon"]["ground_mask"] = ground_accum if ground_accum is not None else None # H x W, bool
        scene_dict["recon"]["object_mask"] = mask_accum if mask_accum is not None else None   # H x W, bool
        key_scene_dicts[key] = scene_dict
        with open(base_dir / f'outputs/{key}/scene/scene.pkl', 'wb') as f:
            pickle.dump(scene_dict, f)

        print(f'[Info] Inpainting results are saved in {result_root}')
    
    return key_scene_dicts


if __name__ == '__main__':
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
    background_pixel_inpainting(keys, key_scene_dicts, key_cfgs)

