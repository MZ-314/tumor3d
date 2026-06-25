"""MedSAM organ segmentation on 2D MRI slices (GPU, Phase 2 Model 1)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from config_medical import SegmentationError
from config_pipeline import MEDSAM_CHECKPOINT

logger = logging.getLogger(__name__)

_predictor = None


def medsam_available() -> bool:
    return MEDSAM_CHECKPOINT.is_file()


def _require_cuda():
    import torch

    if not torch.cuda.is_available():
        raise SegmentationError(
            "MedSAM requires a CUDA GPU. Deploy on RunPod with NVIDIA GPU and install GPU deps:\n"
            "  pip install -e backend/.[gpu]\n"
            "  python backend/scripts/setup_medsam.py"
        )
    return torch.device("cuda")


def _load_predictor():
    global _predictor
    if _predictor is not None:
        return _predictor

    if not medsam_available():
        raise SegmentationError(
            f"MedSAM checkpoint not found at {MEDSAM_CHECKPOINT}.\n"
            "On RunPod run: python backend/scripts/setup_medsam.py"
        )

    try:
        from segment_anything import sam_model_registry, SamPredictor
    except ImportError as exc:
        raise SegmentationError(
            "segment_anything not installed. On RunPod run:\n"
            "  pip install git+https://github.com/facebookresearch/segment-anything.git"
        ) from exc

    device = _require_cuda()
    sam = sam_model_registry["vit_b"](checkpoint=str(MEDSAM_CHECKPOINT))
    sam.to(device=device)
    sam.eval()
    _predictor = SamPredictor(sam)
    return _predictor


def _slice_to_rgb(slice_img: np.ndarray) -> np.ndarray:
    """MedSAM expects HxWx3 uint8."""
    u8 = (np.clip(slice_img, 0, 1) * 255).astype(np.uint8)
    return np.stack([u8, u8, u8], axis=-1)


def _brain_bbox(h: int, w: int) -> np.ndarray:
    """Center crop bbox prompt for intracranial MRI."""
    margin_y, margin_x = int(h * 0.12), int(w * 0.12)
    return np.array([margin_x, margin_y, w - margin_x, h - margin_y])


def segment_organ_2d(slice_img: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Segment organ ROI on a single 2D grayscale slice.

    Returns (bool mask HxW, confidence).
    """
    if slice_img.ndim != 2:
        raise SegmentationError(f"MedSAM expects 2D slice, got shape {slice_img.shape}")

    predictor = _load_predictor()
    rgb = _slice_to_rgb(slice_img)
    predictor.set_image(rgb)

    h, w = slice_img.shape
    box = _brain_bbox(h, w)
    masks, scores, _ = predictor.predict(box=box, multimask_output=True)

    if masks is None or len(masks) == 0:
        raise SegmentationError("MedSAM returned no organ mask for this slice.")

    best_idx = int(np.argmax(scores))
    mask = masks[best_idx].astype(bool)
    confidence = float(scores[best_idx])
    return mask, confidence


def save_mask_png(mask: np.ndarray, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)
