import sys
import numpy as np
import cv2, pickle
from pathlib import Path
from typing import Dict, Any
import supervision as sv
import yaml

ROOT  = Path.cwd()
THIRD = ROOT / "third_party/Grounded-SAM-2"
sys.path.append(str(THIRD))

from sam2.build_sam import build_sam2_video_predictor

OUT_ROOT = ROOT / "outputs"; OUT_ROOT.mkdir(exist_ok=True)

CFG  = "configs/sam2.1/sam2.1_hiera_l.yaml"
CKPT = "third_party/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"

video_pred = build_sam2_video_predictor(CFG, CKPT)

MASK_ANN  = sv.MaskAnnotator()
BOX_ANN   = sv.BoxAnnotator()
LABEL_ANN = sv.LabelAnnotator()

def draw_objects(img: np.ndarray, objs: Dict[int, Dict]) -> np.ndarray:
    """Draw all masks/bboxes/text in objs onto img at once"""
    if not objs:
        return img

    masks = np.stack([o["mask"] for o in objs.values()])
    xyxy  = np.stack([o["bbox"] for o in objs.values()])
    class_ids = np.array(list(objs.keys()))
    det = sv.Detections(xyxy=xyxy, mask=masks, class_id=class_ids)

    img = MASK_ANN.annotate(img, det)
    img = BOX_ANN .annotate(img, det)
    labels = [f"{oid}_{objs[oid]['name']}" for oid in det.class_id]
    img = LABEL_ANN.annotate(img, det, labels=labels)
    return img

def overlay(img, mask, color, a=.45):
    out = img.copy()
    out[mask] = out[mask]*(1-a)+np.array(color)*a
    return out.astype(np.uint8)


def render(segmented_video: object, idx:int):
    if "frames" not in segmented_video or idx>=len(segmented_video["frames"]): return np.zeros((60,60,3),np.uint8)
    img=draw_objects(segmented_video["frames"][idx].copy(),segmented_video["mask_dict"].get(idx,{}))
    return img

def mask_iou(a: np.ndarray, b: np.ndarray)->float:
    inter=np.logical_and(a,b).sum(); union=np.logical_or(a,b).sum()
    return 0. if union==0 else inter/union

def add_mask(segmented_video: object, frame_idx:int, name:str, bound_box, mask, iou_thr, object_id:int):
    segmented_video["mask_dict"].setdefault(frame_idx,{})
    for object_dict in segmented_video["mask_dict"][frame_idx].values():
        if object_dict["name"] == name and mask_iou(object_dict["mask"], mask)>iou_thr:
            object_dict["bbox"] = bound_box
            object_dict["mask"] = mask
            return
        
    segmented_video["mask_dict"][frame_idx][object_id]={"name": name, "bbox": bound_box, "mask": mask}


def save_mask_dict(segmented_video: object, out_dir: Path, mask_dict: Dict[int,Dict]):
    scene_path = out_dir / "scene/scene.pkl"
    with open(scene_path, "rb") as f:
        scene_dict = pickle.load(f)
    scene_dict["mask"] = mask_dict
    with open(scene_path, "wb") as f:
        pickle.dump(scene_dict, f)
    
    vis_dir=OUT_ROOT/segmented_video["key"]/ "annotated_images"; vis_dir.mkdir(parents=True,exist_ok=True)
    for frame_idx in range( len(segmented_video["frames"]) ):
        cv2.imwrite(str(vis_dir/f"{frame_idx:06d}.jpg"), cv2.cvtColor(render(segmented_video, frame_idx),cv2.COLOR_RGB2BGR))


def propagate_maks(segmented_video: object):
    cur_idx=segmented_video.get("cur",0);     
    mask_dict = segmented_video["mask_dict"]
    objects=mask_dict.get(0,{}) 

    if not objects: return "⚠️ No confirmed objects in current frame"
    print("Propagation Started")

    object_pairs = [(object_id ,object_dict["mask"]) for object_id, object_dict in objects.items()]

    state=video_pred.init_state(video_path=str(segmented_video["resized_dir"]))
    for object_id, object_mask in object_pairs: 
        video_pred.add_new_mask(state, cur_idx, object_id, object_mask)

    frames = {}
    for frame_idx, object_ids, masks in video_pred.propagate_in_video(state):
        frames[frame_idx] = {
            object_id: (mask > 0).cpu().numpy() for object_id, mask in zip(object_ids, masks)
        }

    for frame_idx, frame_objects in frames.items():
        print("frame", frame_idx, " of ", len(frames))
        for object_id, object_mask in frame_objects.items():
            bound_box=sv.mask_to_xyxy(np.squeeze(object_mask)[None])[0]
            name=segmented_video["mask_dict"][cur_idx][object_id]["name"]
            add_mask(segmented_video, frame_idx, name, bound_box, np.squeeze(object_mask), 0.99, object_id)
    save_mask_dict(segmented_video, OUT_ROOT/segmented_video["key"], segmented_video["mask_dict"])

    print("✅ Propagation finished and saved")

def mask_propagation(keys):
    for key in keys:
        print("propagating for", key)
        with open(OUT_ROOT/key/"scene/first_scene.pkl", "rb") as f:
            segmented_video = pickle.load(f)
            propagate_maks(segmented_video)


if __name__ == "__main__":
    base_dir = Path.cwd()
    cfg_path = base_dir / "config" / "config.yaml"
    cfg = yaml.safe_load(cfg_path.open("r"))
    keys = cfg["keys"]
    mask_propagation(keys)
    