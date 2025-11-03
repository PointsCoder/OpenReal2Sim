import sys
import numpy as np
import cv2, pickle, re
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


#========================================================================================
# Saving
#========================================================================================

def load_frames(frames_directory: Path):
        def natural(p: Path):
            return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', p.stem)]

        pics = sorted(sum((list(frames_directory.glob(f"*.{e}")) for e in ["jpg","jpeg","png"]), []), key=natural)
        frames = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in pics]
        return np.stack(frames, 0) if pics else ([], np.empty((0,)))


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


def render(frames, mask_dict: object, idx:int):
    if len(frames) <= idx: return np.zeros((60,60,3),np.uint8)

    return draw_objects(frames[idx].copy(),mask_dict.get(idx,{}))

def save_masks(output_directory: object, mask_dict: Dict[int, Dict]):
    scene_path = output_directory / "scene/scene.pkl"
    with open(scene_path, "rb") as f:
        scene_dict = pickle.load(f)

    scene_dict["mask"] = mask_dict.copy()

    with open(scene_path, "wb") as f:
        pickle.dump(scene_dict, f)

def save_frames(output_directory, mask_dict):
    annotated_frames_directory= output_directory/ "annotated_images"
    annotated_frames_directory.mkdir(parents=True,exist_ok=True)
    frames = load_frames(output_directory/"resized_images")

    for frame_idx in range( len(frames) ):
        cv2.imwrite(str(annotated_frames_directory/f"{frame_idx:06d}.jpg"), cv2.cvtColor(render(frames, mask_dict,frame_idx),cv2.COLOR_RGB2BGR))


def save_propagation(output_directory: Path, mask_dict: Dict[int,Dict]):
    save_masks(output_directory, mask_dict)
    save_frames(output_directory, mask_dict)

    

#========================================================================================
# Propagating
#========================================================================================
    
def mask_iou(a: np.ndarray, b: np.ndarray)->float:
    inter=np.logical_and(a,b).sum(); union=np.logical_or(a,b).sum()
    return 0. if union==0 else inter/union

def add_mask(segmented_video: object, mask_dict: object, frame_idx:int, name:str, bound_box, mask, iou_thr, object_id:int):
    mask_dict.setdefault(frame_idx,{})
    for object_dict in mask_dict[frame_idx].values():
        if object_dict["name"] == name and mask_iou(object_dict["mask"], mask)>iou_thr:
            object_dict["bbox"] = bound_box
            object_dict["mask"] = mask
            return
        
    mask_dict[frame_idx][object_id]={"name": name, "bbox": bound_box, "mask": mask}

def propagate_maks(segmented_video: object, output_directory: Path):
    cur_idx=segmented_video.get("cur",0);     
    mask_dict = segmented_video["mask"]
    objects=mask_dict.get(0,{}) 

    if not objects: return "⚠️ No confirmed objects in current frame"
    print("Propagation Started")

    object_pairs = [(object_id ,object_dict["mask"]) for object_id, object_dict in objects.items()]

    state=video_pred.init_state(video_path=str(output_directory/"resized_images"))
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
            name=mask_dict[cur_idx][object_id]["name"]
            add_mask(segmented_video, mask_dict, frame_idx, name, bound_box, np.squeeze(object_mask), 0.99, object_id)

    save_propagation(output_directory, mask_dict)

    print("✅ Propagation finished and saved")


#========================================================================================
# Main
#========================================================================================
def mask_propagation(keys):
    for key in keys:
        print("propagating for", key)
        with open(OUT_ROOT/key/"scene/scene.pkl", "rb") as f:
            segmented_video = pickle.load(f)
            propagate_maks(segmented_video, OUT_ROOT/key)


if __name__ == "__main__":
    base_dir = Path.cwd()
    cfg_path = base_dir / "config" / "config.yaml"
    cfg = yaml.safe_load(cfg_path.open("r"))
    keys = cfg["keys"]
    mask_propagation(keys)
    