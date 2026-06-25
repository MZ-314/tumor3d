"""Segmentation backends: MONAI BraTS (GPU) per architecture plan."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config_medical import SegmentationError


@dataclass
class LesionMask:
    lesion_id: int
    mask: np.ndarray  # bool, same shape as volume
    in_plane_confidence: float


@dataclass
class SegmentationResult:
    lesions: list[LesionMask]
    global_confidence: float


VOLUME_ONLY_MODALITIES = frozenset({"knee_mri", "volume_mri", "other_mri", "volume_only"})


def segment_volume(
    volume: np.ndarray,
    backend: str,
    *,
    modality: str = "brain_mri",
) -> SegmentationResult:
    modality = modality.lower()
    if modality in VOLUME_ONLY_MODALITIES:
        return SegmentationResult(lesions=[], global_confidence=0.0)

    backend = backend.lower()
    if backend == "monai":
        return _segment_monai(volume)
    raise SegmentationError(
        f"Unknown segmentation backend: {backend}. Use SEGMENTATION_BACKEND=monai on RunPod GPU."
    )


def _segment_monai(volume: np.ndarray) -> SegmentationResult:
    from pipeline.segment.monai_brats import segment_brats

    return segment_brats(volume)
