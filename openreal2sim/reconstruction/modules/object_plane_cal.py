'''
This file is used to get the plane each object is placed on.
'''
import os
import sys
import numpy as np
import torch
import yaml
import cv2
import random
import pickle
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../third_party/Grounded-SAM-2'))
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor    
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

ROOT = Path.cwd()
THIRD = ROOT / "third_party/Grounded-SAM-2"
sys.path.append(str(THIRD))

TEXT = "ground. plane. table top. desk top"
DEV  = "cuda" if torch.cuda.is_available() else "cpu"
CFG  =  "configs/sam2.1/sam2.1_hiera_l.yaml"
CKPT = ROOT / "third_party/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
img_predictor = SAM2ImagePredictor(build_sam2(CFG, CKPT))
dino_proc  = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                 "IDEA-Research/grounding-dino-base").to(DEV)

def get_object_plane(keys, key_scene_dicts, key_cfgs):
    for key in keys:
        plane_masks = []
        scene_dict = key_scene_dicts[key]
        first_img = scene_dict["recon"]["background"]
        batch = dino_proc(images=[first_img], text = TEXT, return_tensors="pt")
        batch={k:v.to(DEV) for k,v in batch.items()}
        with torch.no_grad():
            out=dino_model(pixel_values=batch["pixel_values"],
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"))
        boxes = dino_proc.post_process_grounded_object_detection(out,batch["input_ids"],
                                                            .25,.3,target_sizes=[first_img.shape[:2]])[0]["boxes"].cpu().numpy()
        for box in boxes:
            img_predictor.set_image(first_img)
            m,*_=img_predictor.predict(box=box,multimask_output=False)
            m = m[0] >.5
            if np.sum(m) > 0:
                plane_masks.append(m)
        if scene_dict.get("recon", None) is None:
            scene_dict["recon"] = {}
        scene_dict["recon"]["plane_masks"] = plane_masks
        objects = scene_dict["mask"][0]
        for oid, obj in objects.items():
            obj_mask = obj["mask"]
            max_overlap = 0
            selected_idx = None
            for i, plane_mask in enumerate(plane_masks):
                overlap = np.logical_and(obj_mask, plane_mask).sum()
                if overlap > max_overlap:
                    max_overlap = overlap
                    selected_idx = i
            if scene_dict.get("info", None) is None:
                scene_dict["info"] = {}
            if scene_dict["info"].get("objects", None) is None:
                scene_dict["info"]["objects"] = {}
            scene_dict["info"]["objects"][oid] = {
                "oid": oid,
                "name": obj["name"],
                "plane_mask": plane_masks[selected_idx] if selected_idx is not None else None,
                "mask": obj_mask
            }

        vis_dir = ROOT / f"outputs/{key}/reconstruction/"
        os.makedirs(vis_dir, exist_ok=True)

        vis_img = scene_dict["images"][0].copy()
        oids = list(objects.keys())
        random.seed(12345)
        def random_color():
            return tuple([random.randint(80,220) for _ in range(3)])

        mask_vis =  vis_img.copy()
        for oid, obj in objects.items():
            obj_mask = obj["mask"]
            plane_mask = obj.get("plane_mask", None)
            color = random_color()
            mask_vis[obj_mask > 0] = color
            if plane_mask is not None:
                mask_vis[plane_mask > 0] = color
        
        plane_mask_vis = np.zeros_like(obj_mask)
        for plane_mask in plane_masks:
            plane_mask_vis[plane_mask > 0] = 1
       
        out_path = vis_dir / f"object_plane_mask_overlay.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(mask_vis, cv2.COLOR_RGB2BGR))
        out_path = vis_dir / f"plane_mask.jpg"
        cv2.imwrite(str(out_path), (plane_mask_vis.astype(np.uint8) * 255))
        print(f"[Visualization] Saved {out_path}")
        key_scene_dicts[key] = scene_dict
        with open(base_dir / f'outputs/{key}/scene/scene.pkl', 'wb') as f:
            pickle.dump(scene_dict, f)
        print(f"[Info] [{key}] Object plane extraction completed.\n")
    return key_scene_dicts
       



if __name__ == "__main__":
    base_dir = Path.cwd()
    cfg_path = base_dir / "config" / "config.yaml"
    cfg = yaml.safe_load(cfg_path.open("r"))
    keys = cfg["keys"]

    from utils.compose_config import compose_configs
    key_cfgs = {key: compose_configs(key, cfg) for key in keys} 
    print(f"Key cfgs: {key_cfgs}")
    key_scene_dicts = {}
    for key in keys:
        scene_pkl = base_dir / f'outputs/{key}/scene/scene.pkl'
        with open(scene_pkl, 'rb') as f:
            scene_dict = pickle.load(f)
        key_scene_dicts[key] = scene_dict
    get_object_plane(keys, key_scene_dicts, key_cfgs)
