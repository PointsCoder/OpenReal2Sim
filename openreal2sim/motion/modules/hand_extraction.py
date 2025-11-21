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
import tqdm
base_dir = Path.cwd()
sys.path.append(str(base_dir / "openreal2sim" / "motion" / "modules"))
sys.path.append(str(base_dir / "third_party" / "WiLoR"))
sys.path.append(str(base_dir / "third_party" / "Grounded-SAM-2"))


from wilor.utils.renderer import cam_crop_to_full
from wilor.models import load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset    
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor    
from ultralytics import YOLO

class WiLoRExtractor:
    def __init__(self,
                 model_path: str,
                 cfg_path: str,
                 yolo_weights_path: str,
                 sam_weights_path: str,
                 sam_cfg_path: str,
                 device: str):
        self._wilor_model, self._wilor_cfg = load_wilor(model_path, cfg_path)
        self._wilor_model.eval()
        self._yolo_detector = YOLO(yolo_weights_path)
        self.device = torch.device(device)
        self._sam_predictor = SAM2ImagePredictor(build_sam2(sam_cfg_path, sam_weights_path))
        
    def process(self, images: np.ndarray, batch_size: int = 16, rescale_factor: float = 1.0):
        boxes = []
        right = []
        masks = []
        self._wilor_model.to(self.device)
        self._yolo_detector.to(self.device)
        self._sam_predictor.model.to(self.device)
        self._wilor_model.eval()

        all_global_orient = []
        all_kpts = []
        all_masks = []
        
        for i in tqdm.tqdm(range(0, len(images)), desc="Detecting hands"):
            batch = np.array(images)[i]
            with torch.no_grad():
                detections = self._yolo_detector.predict(batch, conf=0.3, verbose=False, save=False, show=False)[0]
                
                if len(detections.boxes.cls.cpu().detach().numpy()) == 0:
                    all_global_orient.append(None)
                    all_kpts.append(None)
                    all_masks.append(None)
                else:
                    det = max(detections, key=lambda d: d.boxes.conf.cpu().detach().item())
                    Bbox = det.boxes.data.cpu().detach().squeeze().numpy()
                    cls_flag = det.boxes.cls.cpu().detach().squeeze().item()
                    
                    self._sam_predictor.set_image(batch)
                    hand_masks, scores, logits = self._sam_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=np.array([[Bbox[0], Bbox[1], Bbox[2], Bbox[3]]]),
                        multimask_output=False
                    )
                    hand_masks = hand_masks[0]
                    hand_masks = hand_masks.astype(bool)
                    dataset = ViTDetDataset(self._wilor_cfg, batch, np.array([Bbox[:4]]), np.array([cls_flag]), rescale_factor=rescale_factor)
                    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
                    sign = False
                    for batch in dataloader:
                        assert sign == False
                        batch = recursive_to(batch, torch.device(self.device))
                        with torch.no_grad():
                            out = self._wilor_model(batch)
                        multiplier = 2 * batch['right'][0].cpu().numpy() - 1
                        pred_cam = out['pred_cam'][0] 
                        pred_cam[1] = multiplier * pred_cam[1]
                        box_center =  batch['box_center'][0].float()
                        box_size      = batch['box_size'][0].float()
                        img_size      = batch['img_size'][0].float()
                        scaled_focal_length = self._wilor_cfg.EXTRA.FOCAL_LENGTH / self._wilor_cfg.MODEL.IMAGE_SIZE * img_size.max()
                        #import pdb; pdb.set_trace()
                        pred_cam_t_full     = cam_crop_to_full(pred_cam.reshape(1, 3), box_center.reshape(1, 2), box_size.reshape(1, 1), img_size.reshape(1, 2), scaled_focal_length).detach().cpu().numpy()
                        batch_size = batch['img'].shape[0]                    
                        joints = out['pred_keypoints_3d'][0].detach().cpu().numpy()
                        joints[:, 0] = multiplier * joints[:, 0]
                        cam_t = pred_cam_t_full[0]
                        kpts_2d = self.project_full_img(joints, cam_t, float(scaled_focal_length), img_size)
                        all_kpts.append(kpts_2d)
                        all_global_orient.append(out['pred_mano_params']['global_orient'][0,0].detach().cpu().numpy())
                        all_masks.append(hand_masks)        
                        sign = True
        return all_kpts, all_global_orient, all_masks





    def project_full_img(self, points, cam_trans, focal_length, img_res):
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
    sam_cfg_path = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam_weights_path = "third_party/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
    
    for key in keys:
      
        scene_dict = key_scene_dicts[key]
        config = key_cfgs[key]
        gpu_id = config['gpu']
        device = f"cuda:{gpu_id}"
        wilor_extractor = WiLoRExtractor(model_path=model_path, cfg_path=cfg_path, yolo_weights_path=yolo_weights_path, sam_weights_path=sam_weights_path, sam_cfg_path=sam_cfg_path, device=device)
        #images = np.transpose(scene_dict["images"], (0, 3, 1, 2))
        images = scene_dict["images"].astype(np.float32)
        kpts, global_orient, masks = wilor_extractor.process(images)
        if scene_dict.get("motion") is None:
            scene_dict["motion"] = {}
        scene_dict["motion"]["hand_kpts"] = kpts
        scene_dict["motion"]["hand_global_orient"] = global_orient
        scene_dict["motion"]["hand_masks"] = masks
        key_scene_dicts[key] = scene_dict
        with open(base_dir / f'outputs/{key}/scene/scene.pkl', 'wb') as f:
            pickle.dump(scene_dict, f)


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

    
