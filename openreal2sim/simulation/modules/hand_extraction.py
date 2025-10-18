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

# Add hand_object_detector to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../third_party/hand_object_detector'))
from demo import parse_args, _get_image_blob
from model.utils.config import cfg, cfg_from_file
from model.faster_rcnn.resnet import resnet
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from model.utils.net_utils import vis_detections_filtered_objects_PIL

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../third_party/WiLoR'))
from wilor.models import load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.utils.renderer import cam_crop_to_full
from ultralytics import YOLO

class HandObjectDetector:
    """Hand-Object Detector wrapper for contact detection"""
    
    def __init__(self, 
                 model_path: str,
                 config_path: str = "cfgs/res101.yml",
                 device: str = "cuda",
                 thresh_hand: float = 0.5,
                 thresh_obj: float = 0.5):
        """
        Initialize the hand-object detector
        
        Args:
            model_path: Path to the trained model
            config_path: Path to the config file
            device: Device to run inference on
            thresh_hand: Threshold for hand detection
            thresh_obj: Threshold for object detection
        """
        self.device = device
        self.thresh_hand = thresh_hand
        self.thresh_obj = thresh_obj
        
        # Load config
        if os.path.exists(config_path):
            cfg_from_file(config_path)
        
        # Initialize model
        self.pascal_classes = np.asarray(['__background__', 'targetobject', 'hand'])
        self.fasterRCNN = resnet(self.pascal_classes, 101, pretrained=False, class_agnostic=False)
        self.fasterRCNN.create_architecture()
        
        # Load checkpoint
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            self.fasterRCNN.load_state_dict(checkpoint['model'])
            if 'pooling_mode' in checkpoint.keys():
                cfg.POOLING_MODE = checkpoint['pooling_mode']
        
        self.fasterRCNN.to(device)
        self.fasterRCNN.eval()
        
        # Initialize tensors
        self.im_data = torch.FloatTensor(1).to(device)
        self.im_info = torch.FloatTensor(1).to(device)
        self.num_boxes = torch.LongTensor(1).to(device)
        self.gt_boxes = torch.FloatTensor(1).to(device)
        self.box_info = torch.FloatTensor(1).to(device)
        
        logging.info("Hand-Object Detector initialized successfully")
    
    def detect_hands_and_objects(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect hands and objects in an image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Tuple of (hand_detections, object_detections)
            Each detection contains [bbox(4), score(1), state(1), offset_vector(3), left/right(1)]
        """
        with torch.no_grad():
            # Preprocess image
            blobs, im_scales = _get_image_blob(image)
            im_blob = blobs
            im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)
            
            im_data_pt = torch.from_numpy(im_blob).permute(0, 3, 1, 2).to(self.device)
            im_info_pt = torch.from_numpy(im_info_np).to(self.device)
            
            # Prepare input tensors
            self.im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
            self.im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
            self.gt_boxes.resize_(1, 1, 5).zero_()
            self.num_boxes.resize_(1).zero_()
            self.box_info.resize_(1, 1, 5).zero_()
            
            # Run inference
            rois, cls_prob, bbox_pred, _, _, _, _, _, loss_list = self.fasterRCNN(
                self.im_data, self.im_info, self.gt_boxes, self.num_boxes, self.box_info
            )
            
            scores = cls_prob.data
            boxes = rois.data[:, :, 1:5]
            
            # Extract predicted parameters
            contact_vector = loss_list[0][0]  # hand contact state info
            offset_vector = loss_list[1][0].detach()  # offset vector
            lr_vector = loss_list[2][0].detach()  # hand side info (left/right)
            
            # Get hand contact
            _, contact_indices = torch.max(contact_vector, 2)
            contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()
            
            # Get hand side
            lr = torch.sigmoid(lr_vector) > 0.5
            lr = lr.squeeze(0).float()
            
            # Apply bounding box regression
            if cfg.TEST.BBOX_REG:
                box_deltas = bbox_pred.data
                if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                    if self.device == "cuda":
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                    + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                    + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                    box_deltas = box_deltas.view(1, -1, 4)
                
                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                pred_boxes = clip_boxes(pred_boxes, self.im_info.data, 1)
            else:
                pred_boxes = np.tile(boxes, (1, scores.shape[1]))
            
            pred_boxes /= im_scales[0]
            scores = scores.squeeze()
            pred_boxes = pred_boxes.squeeze()
            
            # Process detections
            obj_dets, hand_dets = None, None
            
            for j in range(1, len(self.pascal_classes)):
                if self.pascal_classes[j] == 'hand':
                    inds = torch.nonzero(scores[:, j] > self.thresh_hand).view(-1)
                elif self.pascal_classes[j] == 'targetobject':
                    inds = torch.nonzero(scores[:, j] > self.thresh_obj).view(-1)
                else:
                    continue
                
                if inds.numel() > 0:
                    cls_scores = scores[:, j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
                    
                    cls_dets = torch.cat((
                        cls_boxes, 
                        cls_scores.unsqueeze(1), 
                        contact_indices[inds], 
                        offset_vector.squeeze(0)[inds], 
                        lr[inds]
                    ), 1)
                    cls_dets = cls_dets[order]
                    keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                    cls_dets = cls_dets[keep.view(-1).long()]
                    
                    if self.pascal_classes[j] == 'targetobject':
                        obj_dets = cls_dets.cpu().numpy()
                    elif self.pascal_classes[j] == 'hand':
                        hand_dets = cls_dets.cpu().numpy()
            
            return hand_dets, obj_dets


class ContactFrameDetector:
    """Detects starting and ending frames of hand-object contact"""
    
    # Contact state definitions from hand-object detector
    CONTACT_STATES = {
        0: "N",  # No contact
        1: "S",  # Self contact
        2: "O",  # Other person contact  
        3: "P",  # Portable object contact
        4: "F"   # Stationary object contact (furniture)
    }
    
    def __init__(self, 
                 model_path: str,
                 contact_types: List[str] = ["P", "F"],  # Types of contact to detect
                 min_contact_frames: int = 5,
                 smoothing_window: int = 3):
        """
        Initialize contact frame detector
        
        Args:
            model_path: Path to hand-object detector model
            contact_types: List of contact types to detect (N, S, O, P, F)
            min_contact_frames: Minimum frames to consider as valid contact
            smoothing_window: Window size for smoothing contact detection
        """
        self.detector = HandObjectDetector(model_path)
        self.contact_types = contact_types
        self.min_contact_frames = min_contact_frames
        self.smoothing_window = smoothing_window
        
    def detect_contact_frames(self, images: List[np.ndarray]) -> Dict[str, List[int]]:
        """
        Detect starting and ending frames of hand-object contact in a video
        
        Args:
            video_path: Path to input video
            
        Returns:
            Dictionary containing contact information for each hand
        """
        contact_frames = []
        frame_count = 0
        
        for frame in images:
            # Detect hands and objects
            hand_dets, obj_dets = self.detector.detect_hands_and_objects(frame)
            
            # Check for contact
            contact_result = self._check_contact(hand_dets, obj_dets)
            contact_frames.append(contact_result)
            
            frame_count += 1
            if frame_count % 100 == 0:
                logging.info(f"Processed {frame_count} frames")
        
        cap.release()
        
        # Extract boolean contact flags for smoothing
        contact_flags = [frame["has_contact"] for frame in contact_frames]
        
        # Smooth contact detection
        contact_flags = self._smooth_contact_detection(contact_flags)
        
        # Find contact segments
        contact_segments = self._find_contact_segments(contact_flags, contact_frames)
        
        return contact_segments
    
    def _check_contact(self, hand_dets: Optional[np.ndarray], obj_dets: Optional[np.ndarray]) -> Dict[str, any]:
        """Check if there's hand-object contact in current frame"""
        if hand_dets is None:
            return {"has_contact": False, "contact_info": []}
        
        contact_info = []
        
        # Check each hand for contact
        for i, hand in enumerate(hand_dets):
            # hand format: [bbox(4), score(1), state(1), offset_vector(3), left/right(1)]
            contact_state_idx = int(hand[5])  # state index
            contact_state = self.CONTACT_STATES.get(contact_state_idx, "N")
            hand_side = "L" if hand[9] < 0.5 else "R"  # left/right hand
            confidence = hand[4]  # detection confidence
            
            # Check if this is a contact type we want to detect
            if contact_state in self.contact_types:
                contact_info.append({
                    "hand_id": i,
                    "hand_side": hand_side,
                    "contact_state": contact_state,
                    "confidence": confidence,
                    "bbox": hand[:4].tolist(),
                    "offset_vector": hand[6:9].tolist()
                })
        
        return {
            "has_contact": len(contact_info) > 0,
            "contact_info": contact_info
        }
    
    def _smooth_contact_detection(self, contact_frames: List[bool]) -> List[bool]:
        """Apply smoothing to contact detection"""
        if len(contact_frames) < self.smoothing_window:
            return contact_frames
        
        smoothed = []
        for i in range(len(contact_frames)):
            start_idx = max(0, i - self.smoothing_window // 2)
            end_idx = min(len(contact_frames), i + self.smoothing_window // 2 + 1)
            window = contact_frames[start_idx:end_idx]
            smoothed.append(sum(window) > len(window) / 2)
        
        return smoothed
    
    def _find_contact_segments(self, contact_flags: List[bool], contact_frames: List[Dict]) -> Dict[str, any]:
        """Find continuous segments of contact"""
        segments = []
        in_contact = False
        start_frame = 0
        
        for i, has_contact in enumerate(contact_flags):
            if has_contact and not in_contact:
                # Start of contact
                start_frame = i
                in_contact = True
            elif not has_contact and in_contact:
                # End of contact
                if i - start_frame >= self.min_contact_frames:
                    # Get detailed contact info for this segment
                    segment_info = {
                        "start_frame": start_frame,
                        "end_frame": i - 1,
                        "duration": i - start_frame,
                        "contact_types": self._get_segment_contact_types(contact_frames[start_frame:i])
                    }
                    segments.append(segment_info)
                in_contact = False
        
        # Handle case where contact continues to end of video
        if in_contact and len(contact_flags) - start_frame >= self.min_contact_frames:
            segment_info = {
                "start_frame": start_frame,
                "end_frame": len(contact_flags) - 1,
                "duration": len(contact_flags) - start_frame,
                "contact_types": self._get_segment_contact_types(contact_frames[start_frame:])
            }
            segments.append(segment_info)
        
        return {
            "contact_segments": segments,
            "total_frames": len(contact_flags),
            "contact_frames": contact_frames,
            "contact_types_detected": self.contact_types
        }
    
    def _get_segment_contact_types(self, segment_frames: List[Dict]) -> Dict[str, int]:
        """Get contact type statistics for a segment"""
        contact_type_counts = {}
        for frame in segment_frames:
            for contact in frame.get("contact_info", []):
                contact_type = contact["contact_state"]
                contact_type_counts[contact_type] = contact_type_counts.get(contact_type, 0) + 1
        return contact_type_counts





class WiLoRExtractor:
    def __init__(self,
                 model_path: str,
                checkpoint_path: str,
                yolo_weights_path: str,
    )
    model, model_cfg = load_wilor(checkpoint_path=str(model_path), cfg_path=str(config_path))
    self._wilor_model = model.to(torch.device(DEVICE))
    self._wilor_model.eval()
    self._wilor_cfg = model_cfg
    self._yolo_detector = YOLO(str(yolo_weights_path))
    
    def process(self, images: np.ndarray, batch_size: int = 16):
        boxes = []
        right = []
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
            batch = recursive_to(batch, torch.device(DEVICE))
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
    for key in keys:
        scene_dict = key_scene_dicts[key]
        images = scene_dict["images"]
        hand_detector_config = {
        "model_path": "../../../third_party/hand_object_detector/models/res101_handobj_100K/pascal_voc/faster_rcnn_1_8_89999.pth",
        "contact_types": ["P"], 
        "min_contact_frames": 5,
        "smoothing_window": 3
    }
        hand_detector = HandObjectDetector(hand_detector_config)
        contact_frames = hand_detector.detect_contact_frames(images)
        scene_dict["hand"]["contact_frames"] = contact_frames
        wilor_config = {
            "model_path": "../../../third_party/WiLoR/pretrained_models/wilor_final.ckpt",
            "checkpoint_path": "../../../third_party/WiLoR/pretrained_models/model_config.yaml",
            "yolo_weights_path": "../../../third_party/WiLoR/pretrained_models/detector.pt"
        }
        wilor_extractor = WiLoRExtractor(wilor_config)
        kpts, global_orient = wilor_extractor.process(images)
        scene_dict["hand"]["kpts"] = kpts
        scene_dict["hand"]["global_orient"] = global_orient
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
    key_scene_dicts = {}
    for key in keys:
        scene_pkl = base_dir / f'outputs/{key}/scene/scene.pkl'
        with open(scene_pkl, 'rb') as f:
            scene_dict = pickle.load(f)
        key_scene_dicts[key] = scene_dict

    
