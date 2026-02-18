import os
import sys

import cv2
import numpy as np
import torch
import yaml
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import pickle
import tqdm

base_dir = Path.cwd()
sys.path.append(str(base_dir / "openreal2sim" / "motion" / "modules"))
sys.path.append(str(base_dir / "openreal2sim" / "motion" / "utils"))
sys.path.append(str(base_dir / "third_party" / "WiLoR"))

from wilor.utils.renderer import cam_crop_to_full
from wilor.models import load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset    
from ultralytics import YOLO

torch.set_num_threads(8)
torch.set_num_interop_threads(4)

class WiLoRExtractor:
    def __init__(self,
                 model_path: str,
                 cfg_path: str,
                 yolo_weights_path: str,
                 device: str):
        self._wilor_model, self._wilor_cfg = load_wilor(model_path, cfg_path)
        self._wilor_model.eval()
        self._yolo_detector = YOLO(yolo_weights_path)
        self.device = torch.device(device)
        
    def process(self, images: np.ndarray, batch_size: int = 16, rescale_factor: float = 1.0, visualize: bool = False, vis_output_dir: Optional[str] = None):
        self._wilor_model.to(self.device)
        self._yolo_detector.to(self.device)
        self._wilor_model.eval()

        all_global_orient = []
        all_kpts = []
        all_masks = []
        has_hand = False
        
        # Create output directory for visualization if needed
        if visualize and vis_output_dir:
            vis_output_path = Path(vis_output_dir)
            vis_output_path.mkdir(parents=True, exist_ok=True)
        
        for i in tqdm.tqdm(range(0, len(images)), desc="Detecting hands"):
            # Get single frame and convert to float32 only when needed (much faster than bulk conversion)
            frame_raw = np.array(images[i])
            if frame_raw.dtype != np.float32:
                batch = frame_raw.astype(np.float32)
            else:
                batch = frame_raw
            
            try:
                with torch.no_grad():
                    detections = self._yolo_detector.predict(batch, conf=0.3, verbose=False, save=False, show=False)[0]
            except Exception as e:
                print(f"[ERROR] Frame {i}: YOLO detection failed: {e}")
                raise
            
            if len(detections.boxes.cls.cpu().detach().numpy()) == 0:
                all_global_orient.append(None)
                all_kpts.append(None)
                all_masks.append(None)
            else:
                has_hand = True
                det = max(detections, key=lambda d: d.boxes.conf.cpu().detach().item())
                Bbox = det.boxes.data.cpu().detach().squeeze().numpy()
                cls_flag = det.boxes.cls.cpu().detach().squeeze().item()
                
                # Create rectangular mask from YOLO bounding box
                img_h, img_w = batch.shape[:2]
                hand_masks = np.zeros((img_h, img_w), dtype=bool)
                x1, y1, x2, y2 = int(Bbox[0]), int(Bbox[1]), int(Bbox[2]), int(Bbox[3])
                # Ensure coordinates are within image bounds
                x1 = max(0, min(x1, img_w - 1))
                y1 = max(0, min(y1, img_h - 1))
                x2 = max(0, min(x2, img_w - 1))
                y2 = max(0, min(y2, img_h - 1))
                hand_masks[y1:y2, x1:x2] = True
                
                # Visualize mask if requested
                if visualize:
                    self._visualize_mask(batch, hand_masks, Bbox, i, vis_output_dir)
                
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
                    pred_cam_t_full     = cam_crop_to_full(pred_cam.reshape(1, 3), box_center.reshape(1, 2), box_size.reshape(1, 1), img_size.reshape(1, 2), scaled_focal_length).detach().cpu().numpy()
                    joints = out['pred_keypoints_3d'][0].detach().cpu().numpy()
                    joints[:, 0] = multiplier * joints[:, 0]
                    cam_t = pred_cam_t_full[0]
                    kpts_2d = self.project_full_img(joints, cam_t, float(scaled_focal_length), img_size)
                    all_kpts.append(kpts_2d)
                    all_global_orient.append(out['pred_mano_params']['global_orient'][0,0].detach().cpu().numpy())
                    all_masks.append(hand_masks)        
                    sign = True
        return all_kpts, all_global_orient, all_masks, has_hand

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

    def _visualize_mask(self, image: np.ndarray, mask: np.ndarray, bbox: np.ndarray, frame_idx: int, output_dir: Optional[str] = None):
        """Visualize the mask overlay on the image for debugging."""
        # Ensure image is in uint8 format
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                vis_image = (image * 255).astype(np.uint8)
            else:
                vis_image = image.astype(np.uint8)
        else:
            vis_image = image.copy()
        
        # Convert to BGR for OpenCV
        if len(vis_image.shape) == 3 and vis_image.shape[2] == 3:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        elif len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        
        # Create colored mask overlay (semi-transparent red)
        mask_colored = np.zeros_like(vis_image)
        mask_colored[mask] = [0, 0, 255]  # Red in BGR
        vis_image = cv2.addWeighted(vis_image, 0.7, mask_colored, 0.3, 0)
        
        # Draw bounding box
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add text info
        cv2.putText(vis_image, f"Frame {frame_idx}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(vis_image, f"BBox: [{x1}, {y1}, {x2}, {y2}]", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save or display
        if output_dir:
            output_path = Path(output_dir) / f"hand_mask_frame_{frame_idx:04d}.jpg"
            cv2.imwrite(str(output_path), vis_image)
        else:
            # Display using OpenCV (for debugging)
            cv2.imshow(f"Hand Mask Debug - Frame {frame_idx}", vis_image)
            cv2.waitKey(1)  # Non-blocking wait


def hand_extraction(keys, key_scene_dicts, key_cfgs, visualize: bool = False):
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
        images_raw = scene_dict["images"]
        
        print(f"[Info] Extracting hands for key: {key}")
        
        # Set up visualization output directory
        vis_output_dir = None
        if visualize:
            vis_output_dir = base_dir / f'outputs/{key}/hand_masks_debug'
            vis_output_dir.mkdir(parents=True, exist_ok=True)
            print(f"[Info] Saving mask visualizations to: {vis_output_dir}")
        
        kpts, global_orient, masks, has_hand = wilor_extractor.process(images_raw, visualize=visualize, vis_output_dir=str(vis_output_dir) if vis_output_dir else None)
        
        if scene_dict.get("motion") is None:
            scene_dict["motion"] = {}
        scene_dict["motion"]["hand_kpts"] = kpts
        scene_dict["motion"]["hand_global_orient"] = global_orient
        scene_dict["motion"]["hand_masks"] = masks
        scene_dict["motion"]["has_hand"] = has_hand
        print(f"[Info] Hand extraction completed for key: {key}, has_hand: {has_hand}")
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
    # Set visualize=True to enable mask visualization for debugging
    hand_extraction(keys, key_scene_dicts, key_cfgs, visualize=True)
