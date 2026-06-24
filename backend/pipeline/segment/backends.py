"""Segmentation backends: stub (CPU demo) and MONAI (GPU)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

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
    if backend == "stub":
        return _segment_stub(volume, modality=modality)
    if backend == "monai":
        return _segment_monai(volume)
    raise SegmentationError(f"Unknown segmentation backend: {backend}")


def _brain_roi_mask(slice_img: np.ndarray) -> np.ndarray:
    """Elliptical mask approximating intracranial region."""
    h, w = slice_img.shape
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    ry, rx = h * 0.43, w * 0.43
    return ((yy - cy) ** 2 / ry**2 + (xx - cx) ** 2 / rx**2) <= 1.0


def _segment_stub_brain(slice_img: np.ndarray, volume: np.ndarray, z_mid: int) -> SegmentationResult:
    """
    Conservative single-candidate heuristic for UI wiring tests only.
    Not a tumor detector — real MRI requires MONAI on GPU.
    """
    roi = _brain_roi_mask(slice_img)
    roi_vals = slice_img[roi]
    if roi_vals.size < 100:
        raise SegmentationError("Could not estimate brain region on this image.")

    # Hyperintense focal regions within brain (rough T1c-style heuristic).
    thresh = float(np.percentile(roi_vals, 97.5))
    candidate = (slice_img >= thresh) & roi
    candidate = ndimage.binary_opening(candidate, iterations=2)
    candidate = ndimage.binary_closing(candidate, iterations=2)

    labeled, n = ndimage.label(candidate)
    roi_area = int(roi.sum())
    min_area = max(40, int(roi_area * 0.001))
    max_area = int(roi_area * 0.12)

    best_mask: np.ndarray | None = None
    best_area = 0
    for label_id in range(1, n + 1):
        component = labeled == label_id
        area = int(component.sum())
        if area < min_area or area > max_area:
            continue
        if area > best_area:
            best_area = area
            best_mask = component

    if best_mask is None:
        raise SegmentationError(
            "Stub demo mode cannot find a plausible focal region on this brain MRI. "
            "For real tumor segmentation on RunPod run: "
            "export SEGMENTATION_BACKEND=monai and configure MONAI_BUNDLE_DIR "
            "(see docs/runpod-setup.md)."
        )

    if volume.shape[0] == 1:
        mask3d = best_mask[np.newaxis, ...]
    else:
        mask3d = np.zeros_like(volume, dtype=bool)
        mask3d[z_mid] = best_mask

    lesion = LesionMask(lesion_id=1, mask=mask3d, in_plane_confidence=0.22)
    return SegmentationResult(lesions=[lesion], global_confidence=0.22)


def _segment_stub_generic(slice_img: np.ndarray, volume: np.ndarray, z_mid: int) -> SegmentationResult:
    """Fallback for non-brain modalities in stub mode — single largest bright blob."""
    smoothed = ndimage.gaussian_filter(slice_img, sigma=2.0)
    thresh = float(np.percentile(smoothed, 96))
    binary = smoothed >= thresh
    binary = ndimage.binary_opening(binary, iterations=2)
    binary = ndimage.binary_closing(binary, iterations=3)

    labeled, n = ndimage.label(binary)
    if n == 0:
        raise SegmentationError("Stub segmentation found no bright regions.")

    areas = [(labeled == i).sum() for i in range(1, n + 1)]
    label_id = int(np.argmax(areas)) + 1
    component = labeled == label_id

    if volume.shape[0] == 1:
        mask3d = component[np.newaxis, ...]
    else:
        mask3d = np.zeros_like(volume, dtype=bool)
        mask3d[z_mid] = component

    lesion = LesionMask(lesion_id=1, mask=mask3d, in_plane_confidence=0.2)
    return SegmentationResult(lesions=[lesion], global_confidence=0.2)


def _segment_stub(volume: np.ndarray, *, modality: str) -> SegmentationResult:
    z_mid = volume.shape[0] // 2
    slice_img = volume[z_mid]
    if modality.startswith("brain"):
        return _segment_stub_brain(slice_img, volume, z_mid)
    return _segment_stub_generic(slice_img, volume, z_mid)


def _segment_monai(volume: np.ndarray) -> SegmentationResult:
    from pipeline.segment.monai_brats import segment_brats

    return segment_brats(volume)
