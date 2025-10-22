## In this file: we use hand-object detector to decide the starting and ending frame of contact.
## Then we use wilor to extract the hand info.


import os
import sys
import cv2
import numpy as np
import torch
import yaml
from typing import List, Tuple, Dict, Optional
import logging
from pathlib import Path
import pickle
import logging


sys.path.append(os.path.join(os.path.dirname(__file__), '../../../third_party/WiLoR'))
from wilor.models import load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.utils.renderer import cam_crop_to_full
from ultralytics import YOLO

class WiLoRExtractor:
    def __init__(self,
                 model_path: str,
                 cfg_path: str,
                 yolo_weights_path: str,
                 device: str):
        self._wilor_model, self._wilor_cfg = load_wilor(model_path, cfg_path)
        self._wilor_model.eval()
        self._yolo_detector = YOLO(yolo_weights_path)
        
    def process(self, images: np.ndarray, batch_size: int = 16):
        boxes = []
        right = []
        self._wilor_model.to(torch.device(device))
        self._yolo_detector.to(torch.device(device))
        self._wilor_model.eval()
        self._yolo_detector.eval()
        for i in range(0, len(images), batch_size):
            batch = np.array(images)[i:i+batch_size]
            detections = self._yolo_detector(batch, conf=0.3, verbose=False)
            for single_detection in detections:
                bboxes = []
                is_right = []
                # If no detection (ie, no hand), append empty
                if len(single_detection) == 0:
                    boxes.append(np.empty((0, 4), dtype=np.float32))
                    right.append(np.empty((0,), dtype=np.float32))
                    continue
                for det in single_detection:
                    Bbox = det.boxes.data.cpu().detach().squeeze().numpy()
                    cls_flag = det.boxes.cls.cpu().detach().squeeze().item()
                    is_right.append(cls_flag)
                    bboxes.append(Bbox[:4].tolist())
                boxes.append(np.stack(bboxes))
                right.append(np.stack(is_right))

        dataset = ViTDetDataset(self._wilor_cfg, images, boxes, right, rescale_factor=rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
        all_kpts = []
        all_global_orient = []
        for batch in dataloader:
            batch = recursive_to(batch, torch.device(device))
            with torch.no_grad():
                out = self._wilor_model(batch)
            multiplier = (2 * batch['right'] - 1)
            pred_cam = out['pred_cam']
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            box_center = batch['box_center'].float()
            box_size = batch['box_size'].float()
            img_size = batch['img_size'].float()
            scaled_focal_length = self._wilor_cfg.EXTRA.FOCAL_LENGTH / self._wilor_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                # Check if this sample had no hand: right[n] is empty (from above) or batch['right'][n] is empty
                if batch['right'][n].numel() == 0:
                    all_kpts.append(np.empty((0,2)))
                    all_global_orient.append(np.empty((0,)))
                    continue
                joints = out['pred_keypoints_3d'][n].detach().cpu().numpy()
                is_right_flag = batch['right'][n].cpu().numpy()
                joints[:, 0] = (2 * is_right_flag - 1) * joints[:, 0]
                cam_t = pred_cam_t_full[n]
                kpts_2d = self.project_full_img(joints, cam_t, float(scaled_focal_length), img_size[n])
                all_kpts.append(kpts_2d)
                all_global_orient.append(out['pred_global_orient'][n].detach().cpu().numpy())
        return all_kpts, all_global_orient





    def project_full_img(points, cam_trans, focal_length, img_res):
        ''' we use simple K here. It works.'''
        if not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32)
        if not isinstance(cam_trans, torch.Tensor):
            cam_trans = torch.tensor(cam_trans, dtype=torch.float32)
        # Ensure numeric image resolution
        try:
            img_w = float(img_res[0])
            img_h = float(img_res[1])
        except Exception:
            # Fallback for unexpected types
            img_w, img_h =  float(img_res[0].item()), float(img_res[1].item())
        K = torch.eye(3, dtype=torch.float32)
        K[0, 0] = float(focal_length)
        K[1, 1] = float(focal_length)
        K[0, 2] = img_w / 2.0
        K[1, 2] = img_h / 2.0
        pts = points + cam_trans
        pts = pts / pts[..., -1:]
        V_2d = (K @ pts.T).T
        return V_2d[..., :-1].detach().cpu().numpy()






def hand_extraction(keys, key_scene_dicts, key_cfgs):
    base_dir = Path.cwd()
    model_path = base_dir / "third_party" / "WiLoR" / "pretrained_models" / "wilor_final.ckpt"
    cfg_path = base_dir / "third_party" / "WiLoR" / "pretrained_models" / "model_config.yaml"
    yolo_weights_path = base_dir / "third_party" / "WiLoR" / "pretrained_models" / "detector.pt"

    
    for key in keys:
      
        scene_dict = key_scene_dicts[key]
        config = key_cfgs[key]
        gpu_id = config['gpu']
        device = f"cuda:{gpu_id}"
        wilor_extractor = WiLoRExtractor(model_path=model_path, cfg_path=cfg_path, yolo_weights_path=yolo_weights_path, device=device)
        images = scene_dict["images"]
        kpts, global_orient = wilor_extractor.process(images)
        scene_dict["simulation"]["hand_kpts"] = kpts
        scene_dict["simulation"]["hand_global_orient"] = global_orient
        key_scene_dicts[key] = scene_dict
        with open(base_dir / f'outputs/{key}/scene/scene.pkl', 'wb') as f:
            pickle.dump(scene_dict, f)

        import cv2
        first_img = scene_dict["images"][0]
        first_kpts = kpts[0] 
        pts2d = np.asarray(first_kpts).astype(np.int32)
        overlay_img = first_img.copy()
        for x, y in pts2d:
            cv2.circle(overlay_img, (int(x), int(y)), 4, (0, 255, 0), -1)
        save_dir = base_dir / f'outputs/{key}/simulation/'
        save_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = save_dir / "first_frame_hand_kpts_overlay.jpg"
        cv2.imwrite(str(overlay_path), overlay_img)
        print(f"Saved hand keypoints overlay for {key} to {overlay_path}")
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
    hand_extraction(keys, key_scene_dicts, key_cfgs)

    
