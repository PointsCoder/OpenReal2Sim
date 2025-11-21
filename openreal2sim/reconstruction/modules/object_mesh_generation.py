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
import torch
import cv2
base_dir   = Path.cwd()
output_dir = base_dir / "outputs"
repo_dir   = str(base_dir / 'third_party/Hunyuan3D-2')
sys.path.append(repo_dir)


from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.shapegen import FaceReducer, FloaterRemover, DegenerateFaceRemover

base_dir = Path.cwd()
repo_dir = str(base_dir / 'third_party/ObjectClear')
sys.path.append(repo_dir)
from objectclear.pipelines import ObjectClearPipeline
from objectclear.utils import resize_by_short_side


sys.path.append(str(base_dir / 'third_party/Grounded-SAM-2'))
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor    

# TODO: super-resolution
# ------------------------------------------------------------------

def object_stacking(single_scene_dict):
    objects = load_obj_masks(single_scene_dict['mask'])
    # objects is a list of dicts, convert it into a dict mapping oid to object dict for fast access.
    objects_by_oid = {obj['oid']: obj for obj in objects}
    if not objects or len(objects) <= 1:
        # Nothing to stack
        stacking = {oid: [] for oid in objects_by_oid}
        return stacking

    K      = single_scene_dict["intrinsics"]
    depth  = single_scene_dict["depths"][0]
    H, W   = depth.shape
    ground = single_scene_dict["info"]["groundplane_in_cam"]
    g_point = np.array(ground["point"]).reshape(3)
    g_normal = np.array(ground["normal"]).reshape(3)
    g_normal = g_normal/np.linalg.norm(g_normal)

    # Camera center calculation: assume pinhole at [0,0,0]
    cam_center = np.zeros(3)

   
    stacking_normal = g_normal / np.linalg.norm(g_normal)
   
    cam_ground_sign = np.sign(g_normal[2])  

    def depth_to_points(mask):
        # Project 2D pixel to camera coordinates
        ys, xs = np.where(mask)
        ds = depth[ys, xs]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        xs_ = (xs - cx) * ds / fx
        ys_ = (ys - cy) * ds / fy
        return np.stack([xs_, ys_, ds], axis=1)  # (N, 3)
    
    def dilate_mask(mask, radius=1):
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((radius, radius), np.uint8))
        return mask.astype(bool)

    # Compute "height from ground" for each object (median, robust to outliers)
    heights = {}
    obj_cloud = {}
    for idx, obj in enumerate(objects):
        oid = obj['oid']
        msk = obj['mask']
        pts = depth_to_points(msk)
        obj_cloud[oid] = pts
        if pts.shape[0] == 0:
            heights[oid] = float("nan")
            continue
        vecs = pts - g_point  # (N,3)
        hts = vecs @ g_normal  # project to normal
        heights[oid] = np.median(hts)

    stacking = {obj['oid']: [] for obj in objects}
    
    # Compute mask overlap between objects
    for idx, obj_a in enumerate(objects):
        oid_a = obj_a['oid']
        orig_mask_a = np.array(obj_a["mask"]).astype(bool)
        mask_a = dilate_mask(orig_mask_a, radius=10)
        if not np.any(mask_a):
            continue
        for idx_b, obj_b in enumerate(objects):
            oid_b = obj_b['oid']
            if oid_a == oid_b:
                continue
            orig_mask_b = np.array(obj_b["mask"]).astype(bool)
            mask_b = dilate_mask(orig_mask_b, radius=10)
            if not np.any(mask_b):
                continue
            overlap_mask = mask_a & mask_b
            if not np.any(overlap_mask):
                continue
            # Heights at overlap
            pts_a = depth_to_points(orig_mask_a)
            pts_b = depth_to_points(orig_mask_b)
            if pts_a.shape[0] == 0 or pts_b.shape[0] == 0:
                continue
            h_a = pts_a @ g_normal
            h_b = pts_b @ g_normal
            h_a_val = np.median(h_a)
            h_b_val = np.median(h_b)

            # Stacked-on rule:
            # If cam_ground_sign < 0 (camera is above), higher objects (along normal) are in front, thus "on top"
            # If cam_ground_sign > 0 (camera is below), lower (along normal) are in front, thus "on top"
            obj_a_name = obj_a["name"]
            obj_b_name = obj_b["name"]
            if np.isfinite(h_a_val) and np.isfinite(h_b_val):
                if cam_ground_sign < 0:
                    # camera above ground & objects: higher-along-normal overlays lower
                    print("Camera is above ground")
                    if h_a_val > h_b_val + 0.01:
                        stacking[oid_b].append(oid_a)
                        print(f"Object {obj_a_name} is stacked on {obj_b_name}")
                        print(f"Object {obj_a_name} occludes {obj_b_name}")
                else:
                    # camera below: lower-along-normal overlays higher
                    print("Camera is below ground")
                    if h_a_val < h_b_val - 0.01:
                        stacking[oid_b].append(oid_a)
                        print(f"Object {obj_b_name} is stacked on {obj_a_name}")
                        print(f"Object {obj_a_name} occludes {obj_b_name}")

    # Remove duplicates, sort
    for oid in stacking:
        stacking[oid] = list(sorted(set(stacking[oid])))

    return stacking





def clear_object(orig_img, orig_mask, whole_mask):
    # Compute bbox from orig_mask
    ys, xs = np.where(orig_mask)
    if ys.size == 0 or xs.size == 0:
        raise ValueError("orig_mask contains no foreground pixels")
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    margin = int(0.1 * max(x2 - x1, y2 - y1))
    # expand bbox by margin, clamp to image boundaries
    h, w = orig_img.shape[0], orig_img.shape[1]
    x1 = max(x1 - margin, 0)
    y1 = max(y1 - margin, 0)
    x2 = min(x2 + margin, w - 1)
    y2 = min(y2 + margin, h - 1)
    crop_img = orig_img[y1:y2+1, x1:x2+1]
    orig_mask = orig_mask[y1:y2+1, x1:x2+1]
    new_mask = whole_mask[y1:y2+1, x1:x2+1]
    new_mask = ~orig_mask & new_mask

    return crop_img, new_mask, orig_mask

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

    sam_weights_path = "third_party/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
    sam_cfg_path = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam_predictor = SAM2ImagePredictor(build_sam2(sam_cfg_path, sam_weights_path))
    sam_predictor.model.to(device)

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
        stacking = object_stacking(scene_dict)
        for idx, item in enumerate(objs):
            mask  = item['mask'].astype(bool)
            name = item['name']
            stem  = f"{item['oid']}_{name}"
            png_path = out_dir / f"{stem}.png"
            regenerated_path = out_dir / f"{stem}_regenerated.png"
            cut_mask_path = out_dir / f"{stem}_cut_mask.png"
            # 1) save transparent PNG
            if len(stacking[item['oid']]) > 0:
                print(f"[Info] processing stacking for object {name}")
                stack_mask = np.zeros_like(mask)
                for oid in stacking[item['oid']]:
                    for obj in objs:
                        if obj["oid"] == oid:
                            break
                    stack_mask |= np.array(obj["mask"]).astype(bool)
                crop_img, new_mask, orig_mask = clear_object(np.asarray(orig_img), mask, stack_mask)
                crop_img = Image.fromarray(crop_img,mode="RGB")
                crop_img = resize_by_short_side(crop_img, 512, resample=Image.BICUBIC)
                new_mask = resize_by_short_side(Image.fromarray(new_mask.astype(np.uint8) * 255), 512, resample=Image.NEAREST)
                new_mask.save(cut_mask_path)
                orig_mask = resize_by_short_side(Image.fromarray(orig_mask.astype(np.uint8) * 255), 512, resample=Image.NEAREST)
                
                new_img = pipe(
                    prompt="remove the instance of object",
                    image=crop_img,
                    mask_image=new_mask,
                    generator=generator,
                    num_inference_steps=NUM_STEPS,
                    strength=STRENGTH,
                    guidance_scale=GUIDANCE_SCALE,
                    height=crop_img.height,
                    width=crop_img.width,
                )
                crop_img = new_img.images[0]
                sam_predictor.set_image(np.asarray(crop_img))
                ## may sample from orig mask.
                orig_mask_np = np.array(orig_mask).astype(bool)
                yx_locs = np.argwhere(orig_mask_np)
                num_points = min(8, len(yx_locs))
                if num_points > 0:
                    selected_indices = np.random.choice(len(yx_locs), num_points, replace=False)
                    sampled_points = yx_locs[selected_indices]
                    point_coords = np.fliplr(sampled_points).copy()
                    point_labels = np.ones(num_points, dtype=int)
                else:
                    point_coords = None
                    point_labels = None

                new_mask, scores, logits = sam_predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=None,
                    multimask_output=False
                )
                new_mask = new_mask[0].astype(bool)
                crop_img.save(regenerated_path)
                save_object_png(crop_img, new_mask, png_path)
                print(f"[Info] [{key}] saved crop → {png_path}")
            else:
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


