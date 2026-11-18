"""
Automatic frame-0 segmentation: Grounding DINO boxes + SAM 2.1 masks.

Drop-in replacement for the manual step of the segmentation GUI
(`openreal2sim/gui/segmentation_annotator.py`): instead of a human typing object
names, Grounding DINO is prompted with "all objects." and every box it is
confident about becomes one object. The masks go into `scene.pkl` in the
annotator's format, so `segmentation_mask_propagation.py` and every later
reconstruction step are unchanged.

Method (validated on 300 frames against human masks, 97% of frames fully
correct):
  1. Grounding DINO base, prompt "all objects." (the trailing period matters),
     box_threshold 0.05, text_threshold 0.3, then keep the boxes scoring above
     SCORE_CUTOFF (0.4). No NMS.
  2. SAM 2.1 hiera large image predictor, predict(box=..., multimask_output=True)
     per box, and pick among the three candidates by box fill:
         inside_area / box_area - 0.5 * outside_area / mask_area
     Plain multimask rank 1 returns a part of the object (a plate rim, a candle
     glass); the fill score prefers the candidate that fills the DINO box.
  3. Objects are named object_1..object_N in box-score order and get object ids
     1..N, the ids a fresh annotator session would allocate.
  4. A second Grounding DINO pass with the prompt "table. desk. floor." gives the
     ground: the highest scoring box, masked the same way, minus the object
     masks. It is stored last in mask[0] as an object named "ground", the name
     the reconstruction steps look for, and propagates with the rest.

Inputs:
    - outputs/{key}/scene/scene.pkl (from the preprocess steps)
Outputs:
    - outputs/{key}/resized_images/*.jpg (propagation input, as the annotator writes)
    - outputs/{key}/scene/scene.pkl, updated with "mask" (frame 0 only; run the
      mask_propagation step afterwards to fill the remaining frames). Any
      "objects" and "recon" built from older masks are dropped.

Note:
    - "mask" : {frame_idx: {oid: {"name": str, "bbox": [x1,y1,x2,y2], "mask": [H,W] bool}}}
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import yaml

from openreal2sim.steps.reconstruction.steps.segmentation_mask_propagation import save_masks
from openreal2sim.utils.grounded_sam2_loader import detect_boxes, load_sam2_image_predictor

ROOT = Path.cwd()
OUT_ROOT = ROOT / "outputs"

# Defaults validated on the 300 frame benchmark; override from the CLI.
PROMPT = "all objects."
GROUND_PROMPT = "table. desk. floor."
BOX_THRESHOLD = 0.05
TEXT_THRESHOLD = 0.3
SCORE_CUTOFF = 0.4
MAX_BOXES = 100
OUTSIDE_PENALTY = 0.5
GROUND_NAME = "ground"          # the name the reconstruction steps look for
STALE_KEYS = ("objects", "recon")  # rebuilt from the masks, so never kept


# ========================================================================================
# Boxes
# ========================================================================================

def dino_boxes(image: np.ndarray, prompt: str, box_threshold: float,
               text_threshold: float, score_cutoff: float,
               max_boxes: int = MAX_BOXES) -> Tuple[np.ndarray, np.ndarray]:
    """Grounding DINO boxes sorted by descending score, cut at score_cutoff."""
    boxes, scores = detect_boxes(image, prompt, box_threshold, text_threshold)
    order = np.argsort(-scores)[:max_boxes]
    boxes, scores = boxes[order], scores[order]
    keep = scores > score_cutoff
    return boxes[keep].astype(np.float32), scores[keep].astype(np.float32)


# ========================================================================================
# Masks
# ========================================================================================

def box_pixels(box, height: int, width: int) -> Tuple[int, int, int, int]:
    """Integer pixel bounds of a float xyxy box, clamped to the image."""
    x0, y0, x1, y1 = [float(v) for v in box]
    bx0 = int(max(0, np.floor(x0)))
    by0 = int(max(0, np.floor(y0)))
    bx1 = int(min(width, np.ceil(x1)))
    by1 = int(min(height, np.ceil(y1)))
    return bx0, by0, max(bx1, bx0 + 1), max(by1, by0 + 1)


def fill_score(mask: np.ndarray, box, height: int, width: int,
               outside_penalty: float = OUTSIDE_PENALTY) -> float:
    """How well a mask fills its box, penalizing the part that spills out."""
    bx0, by0, bx1, by1 = box_pixels(box, height, width)
    box_area = float((bx1 - bx0) * (by1 - by0))
    inside = float(mask[by0:by1, bx0:bx1].sum())
    total = float(mask.sum())
    outside = total - inside
    return inside / box_area - outside_penalty * (outside / max(total, 1.0))


def masks_by_fill(image: np.ndarray, boxes: np.ndarray,
                  outside_penalty: float = OUTSIDE_PENALTY) -> np.ndarray:
    """One SAM 2 mask per box: the best filling of the three multimask candidates."""
    height, width = image.shape[:2]
    if len(boxes) == 0:
        return np.zeros((0, height, width), bool)

    predictor = load_sam2_image_predictor()
    predictor.set_image(image)
    candidates, _, _ = predictor.predict(box=boxes, multimask_output=True)
    candidates = np.asarray(candidates)
    if candidates.ndim == 3:  # a single box gives (3, H, W)
        candidates = candidates[None]
    candidates = candidates > 0.5

    picked = []
    for i in range(len(boxes)):
        scores = [
            fill_score(candidates[i, k], boxes[i], height, width, outside_penalty)
            for k in range(candidates.shape[1])
        ]
        picked.append(candidates[i, int(np.argmax(scores))])
    return np.stack(picked)


def detect_ground(image: np.ndarray, object_masks: np.ndarray,
                  prompt: str = GROUND_PROMPT,
                  box_threshold: float = BOX_THRESHOLD,
                  text_threshold: float = TEXT_THRESHOLD,
                  outside_penalty: float = OUTSIDE_PENALTY):
    """The ground: the best table, desk or floor box, masked, minus the objects."""
    boxes, scores = detect_boxes(image, prompt, box_threshold, text_threshold)
    if len(boxes) == 0:
        return None, None, 0.0
    top = int(np.argmax(scores))
    box = boxes[top:top + 1].astype(np.float32)
    mask = masks_by_fill(image, box, outside_penalty)[0]
    for object_mask in object_masks:
        mask &= ~object_mask
    return mask, box[0], float(scores[top])


# ========================================================================================
# scene.pkl
# ========================================================================================

def write_resized_images(output_directory: Path, images: np.ndarray) -> int:
    """Write the frames the SAM 2 video predictor propagates over."""
    resized = output_directory / "resized_images"
    resized.mkdir(parents=True, exist_ok=True)
    for i in range(len(images)):
        cv2.imwrite(str(resized / f"{i:06d}.jpg"),
                    cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR))
    return len(images)


def build_frame_objects(boxes: np.ndarray, masks: np.ndarray) -> Dict[int, Dict[str, Any]]:
    """Frame-0 objects in the annotator's format, oids 1..N in box-score order."""
    objects: Dict[int, Dict[str, Any]] = {}
    for index, (box, mask) in enumerate(zip(boxes, masks)):
        oid = index + 1
        objects[oid] = {
            "name": f"object_{oid}",
            "bbox": np.asarray(box, dtype=np.float32),
            "mask": np.asarray(mask, dtype=bool),
        }
    return objects


# ========================================================================================
# Stage
# ========================================================================================

def auto_segment_scene(scene_dict: Dict[str, Any], output_directory: Path,
                       prompt: str = PROMPT,
                       box_threshold: float = BOX_THRESHOLD,
                       text_threshold: float = TEXT_THRESHOLD,
                       score_cutoff: float = SCORE_CUTOFF,
                       outside_penalty: float = OUTSIDE_PENALTY,
                       ground_prompt: str = GROUND_PROMPT) -> Dict[str, Any]:
    """Detect and mask the frame-0 objects of one scene, then save scene.pkl."""
    images = scene_dict["images"]
    n_frames = write_resized_images(output_directory, images)

    frame = np.ascontiguousarray(images[0])
    boxes, scores = dino_boxes(frame, prompt, box_threshold, text_threshold, score_cutoff)
    print(f"[Info] {len(boxes)} boxes above {score_cutoff} from prompt '{prompt}'")
    if len(boxes) == 0:
        print("[Warning] no object detected, nothing written")
        return scene_dict

    masks = masks_by_fill(frame, boxes, outside_penalty)
    if scene_dict.get("mask"):
        print("[Warning] overwriting the masks already in scene.pkl")
    frame_objects = build_frame_objects(boxes, masks)
    for oid, obj in frame_objects.items():
        print(f"[Info]   oid {oid} {obj['name']} score {scores[oid - 1]:.3f} "
              f"area {int(obj['mask'].sum())} px")

    ground, ground_box, ground_score = detect_ground(
        frame, masks, ground_prompt, box_threshold, text_threshold, outside_penalty)
    if ground is None or not ground.any():
        print(f"[Warning] prompt '{ground_prompt}' found no ground, the "
              "reconstruction steps need one: annotate it with "
              "openreal2sim/gui/segmentation_annotator.py")
    else:
        oid = len(frame_objects) + 1
        frame_objects[oid] = {
            "name": GROUND_NAME,
            "bbox": np.asarray(ground_box, dtype=np.float32),
            "mask": ground,
        }
        print(f"[Info]   oid {oid} {GROUND_NAME} score {ground_score:.3f} "
              f"area {int(ground.sum())} px")

    scene_dict["mask"] = {0: frame_objects}
    for stale in STALE_KEYS:
        if scene_dict.pop(stale, None) is not None:
            print(f"[Info] dropped '{stale}', it was built from the old masks")
    save_masks(output_directory, scene_dict["mask"], drop_keys=STALE_KEYS)
    print(f"[Info] frame-0 masks written to {output_directory.name}/scene/scene.pkl, "
          f"{n_frames} resized frames ready to propagate")
    return scene_dict


def auto_segmentation(keys: List[str], key_scene_dicts: Dict[str, Any], **kwargs):
    """Reconstruction step entry point, same shape as the other steps."""
    for key in keys:
        print("auto segmenting", key)
        key_scene_dicts[key] = auto_segment_scene(
            key_scene_dicts[key], OUT_ROOT / key, **kwargs)
    return key_scene_dicts


# ========================================================================================
# Main
# ========================================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--key", type=str, default=None,
                        help="process a single key instead of all keys from config")
    parser.add_argument("--prompt", type=str, default=PROMPT)
    parser.add_argument("--ground-prompt", type=str, default=GROUND_PROMPT)
    parser.add_argument("--box-threshold", type=float, default=BOX_THRESHOLD)
    parser.add_argument("--text-threshold", type=float, default=TEXT_THRESHOLD)
    parser.add_argument("--score-cutoff", type=float, default=SCORE_CUTOFF,
                        help="keep Grounding DINO boxes scoring above this")
    parser.add_argument("--outside-penalty", type=float, default=OUTSIDE_PENALTY,
                        help="weight of the spill term in the box fill score")
    parser.add_argument("--skip-propagation", action="store_true",
                        help="only write the frame-0 masks")
    args = parser.parse_args()

    if args.key is not None:
        keys = [args.key]
    else:
        cfg = yaml.safe_load((ROOT / "config" / "config.yaml").open("r"))
        keys = cfg["keys"]

    key_scene_dicts = {}
    for key in keys:
        with open(OUT_ROOT / key / "scene/scene.pkl", "rb") as f:
            key_scene_dicts[key] = pickle.load(f)

    key_scene_dicts = auto_segmentation(
        keys, key_scene_dicts,
        prompt=args.prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        score_cutoff=args.score_cutoff,
        outside_penalty=args.outside_penalty,
        ground_prompt=args.ground_prompt,
    )

    if args.skip_propagation:
        return
    from openreal2sim.steps.reconstruction.steps.segmentation_mask_propagation import mask_propagation
    mask_propagation(keys, key_scene_dicts)


if __name__ == "__main__":
    main()
