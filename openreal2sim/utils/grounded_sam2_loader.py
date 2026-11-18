"""
Shared Grounded-SAM-2 model loading for the segmentation tools.

The manual annotator (`openreal2sim/gui/segmentation_annotator.py`), the
automatic segmentation step and the mask propagation step need the same models,
so the paths, ids and the Grounding DINO call live here instead of being
duplicated.

Models are built lazily and cached per process, so importing this module is
cheap.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

ROOT = Path.cwd()
THIRD = ROOT / "third_party/Grounded-SAM-2"
if str(THIRD) not in sys.path:
    sys.path.append(str(THIRD))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CKPT = "third_party/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
DINO_ID = "IDEA-Research/grounding-dino-base"

_CACHE: Dict[str, Any] = {}


def load_sam2_image_predictor():
    """SAM 2.1 hiera large image predictor."""
    if "image" not in _CACHE:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        _CACHE["image"] = SAM2ImagePredictor(build_sam2(SAM2_CFG, SAM2_CKPT))
    return _CACHE["image"]


def load_sam2_video_predictor():
    """SAM 2.1 hiera large video predictor, used for mask propagation."""
    if "video" not in _CACHE:
        from sam2.build_sam import build_sam2_video_predictor

        _CACHE["video"] = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT)
    return _CACHE["video"]


def load_dino():
    """Grounding DINO base processor and model, on DEVICE."""
    if "dino" not in _CACHE:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        processor = AutoProcessor.from_pretrained(DINO_ID)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_ID).to(DEVICE)
        _CACHE["dino"] = (processor, model)
    return _CACHE["dino"]


def detect_boxes(
    image: np.ndarray,
    prompt: str,
    box_threshold: float,
    text_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run Grounding DINO on one RGB image.

    Returns (boxes, scores): boxes are xyxy in pixels, scores are the detection
    confidences, both in the order the model returns them.
    """
    processor, model = load_dino()
    batch = processor(images=[image], text=prompt, return_tensors="pt")
    batch = {k: v.to(DEVICE) for k, v in batch.items()}
    with torch.no_grad():
        out = model(
            pixel_values=batch["pixel_values"],
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
        )
    res = processor.post_process_grounded_object_detection(
        out,
        batch["input_ids"],
        box_threshold,
        text_threshold,
        target_sizes=[image.shape[:2]],
    )
    boxes = res[0]["boxes"].float().cpu().numpy()
    scores = res[0]["scores"].float().cpu().numpy()
    return boxes, scores
