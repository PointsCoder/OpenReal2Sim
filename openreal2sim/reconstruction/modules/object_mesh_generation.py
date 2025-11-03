#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate textured meshes for each segmented object in the scene.
Extract object crops with alpha-channel from frame-0 & masks,
then feed each crop to Hunyuan3D to get individual 3D assets.
Inputs:
    - outputs/{key_name}/scene/scene.pkl (must contain the "mask" key)
Outputs:
    - outputs/{key_name}/scene/scene.pkl (updated with "objects" key)
    - outputs/{key_name}/reconstruction/objects/{oid}_{name}.glb (object mesh)
    - outputs/{key_name}/reconstruction/objects/{oid}_{name}.png (object masked image)
Note:
    - added key "objects": {
            "oid": {
                "oid":   # object id,
                "name": # object name,
                "glb": # object glb path,
                "mask": # object mask [H, W] boolean array,
            },
            ...
        }
"""

import os, pickle, json, random, sys
from pathlib import Path
import numpy as np
from PIL import Image
import yaml

base_dir   = Path.cwd()
output_dir = base_dir / "outputs"
repo_dir   = str(base_dir / 'third_party/Hunyuan3D-2')
sys.path.append(repo_dir)

# --- Switched from Trellis to Hunyuan3D ---
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.shapegen import FaceReducer, FloaterRemover, DegenerateFaceRemover

# ------------------------------------------------------------------


def save_object_png(orig_img: Image.Image,
                    mask: np.ndarray,
                    out_png: Path,
                    bbox=None,
                    margin: int = 5):
    """
    Save cropped RGBA PNG for one object.

    orig_img : PIL RGB
    mask     : (H,W) bool
    bbox     : (x1,y1,x2,y2)  if None, compute from mask
    margin   : extend bbox by N pixels on each side
    """
    h, w = mask.shape
    if bbox is None:
        ys, xs = np.where(mask)
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    else:
        x1, y1, x2, y2 = map(int, bbox)

    # expand bbox
    x1 -= margin; y1 -= margin; x2 += margin; y2 += margin
    # clamp to image bounds
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w - 1), min(y2, h - 1)

    crop_rgb = orig_img.crop((x1, y1, x2 + 1, y2 + 1)).convert("RGBA")
    crop_mask = mask[y1:y2 + 1, x1:x2 + 1]

    alpha = np.zeros((*crop_mask.shape, 1), np.uint8)
    alpha[crop_mask] = 255
    crop_rgb.putalpha(Image.fromarray(alpha.squeeze(), mode="L"))

    # upscale if the short side < 128
    short_side = min(crop_rgb.width, crop_rgb.height)
    if short_side < 128:
        scale = 128 / short_side
        new_w = int(round(crop_rgb.width * scale))
        new_h = int(round(crop_rgb.height * scale))
        crop_rgb = crop_rgb.resize((new_w, new_h), Image.LANCZOS)

    crop_rgb.save(out_png)


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

# ------------------------------------------------------------------
def object_mesh_generation(keys, key_scene_dicts, key_cfgs):

    # Init Hunyuan3D pipelines once; reuse for all keys
    pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        'tencent/Hunyuan3D-2', subfolder='hunyuan3d-dit-v2-0-turbo'
    )
    pipeline_texgen = Hunyuan3DPaintPipeline.from_pretrained(
        'tencent/Hunyuan3D-2', subfolder='hunyuan3d-paint-v2-0'
    )
    rembg = BackgroundRemover()

    for key in keys:
        print(f"[Info] Processing {key}...\n")
        scene_dict = key_scene_dicts[key]
        key_cfg = key_cfgs[key]
        objs = load_obj_masks(scene_dict["mask"])

        out_dir = output_dir / key / "reconstruction" / "objects"
        out_dir.mkdir(parents=True, exist_ok=True)

        orig_img = Image.fromarray(scene_dict["images"][0], mode="RGB")

        object_meta = {}
        # fixed seed kept (not used internally by these calls but preserved for parity)
        seed = random.randint(0, 99999)
        # generate object mesh for each object
        for idx, item in enumerate(objs):
            mask  = item['mask'].astype(bool)
            name = item['name']
            stem  = f"{item['oid']}_{name}"
            png_path = out_dir / f"{stem}.png"

            # 1) save transparent PNG
            save_object_png(orig_img, mask, png_path)
            print(f"[Info] [{key}] saved crop → {png_path}")

            # 2) Hunyuan3D shape + texture
            img_rgba = Image.open(png_path).convert("RGBA")
            if img_rgba.mode == 'RGB':  # fallback: ensure RGBA
                img_rgba = rembg(img_rgba)
            # shape generation
            mesh = pipeline_shapegen(image=img_rgba)[0]
            # simplify mesh for much faster texturing
            for cleaner in [FloaterRemover(), DegenerateFaceRemover(), FaceReducer()]:
                mesh = cleaner(mesh)
            print(f"[Info] [{key}] Hunyuan3D shape done for {stem}")
            # texturing
            mesh = pipeline_texgen(mesh, image=img_rgba)
            print(f"[Info] [{key}] Hunyuan3D texture done for {stem}")

            mesh.export(out_dir / f"{stem}.glb")

            # 3) update scene_dict & save mask
            mask_png = out_dir / f"{stem}_mask.jpg"
            Image.fromarray(mask.astype(np.uint8) * 255).save(mask_png)

            object_meta[item['oid']] = {
                "oid": item['oid'],
                "name": name,
                "glb": str(out_dir / f"{stem}.glb"),
                "mask": mask,
            }

            print(f"[Info] [{key}] Hunyuan3D finished for {stem}")

        scene_dict["objects"] = object_meta
        key_scene_dicts[key] = scene_dict
        with open(base_dir / f'outputs/{key}/scene/scene.pkl', 'wb') as f:
            pickle.dump(scene_dict, f)

        print(f"[Info] [{key}] scene_dict updated.")

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

    object_mesh_generation(keys, key_scene_dicts, key_cfgs)


