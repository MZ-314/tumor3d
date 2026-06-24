"""Segmentation backends: stub (CPU) and MONAI (GPU)."""

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


def segment_volume(volume: np.ndarray, backend: str) -> SegmentationResult:
    backend = backend.lower()
    if backend == "stub":
        return _segment_stub(volume)
    if backend == "monai":
        return _segment_monai(volume)
    raise SegmentationError(f"Unknown segmentation backend: {backend}")


def _segment_stub(volume: np.ndarray) -> SegmentationResult:
    """Heuristic bright-region segmentation for demo without GPU."""
    z_mid = volume.shape[0] // 2
    slice_img = volume[z_mid]

    smoothed = ndimage.gaussian_filter(slice_img, sigma=2.0)
    thresh = float(np.percentile(smoothed, 92))
    binary = smoothed >= thresh
    binary = ndimage.binary_opening(binary, iterations=2)
    binary = ndimage.binary_closing(binary, iterations=3)

    labeled, n = ndimage.label(binary)
    if n == 0:
        # Fallback: small central blob so UI always has something in demo mode
        h, w = slice_img.shape
        fallback = np.zeros_like(slice_img, dtype=bool)
        cy, cx = h // 2, w // 2
        fallback[cy - 8 : cy + 8, cx - 8 : cx + 8] = True
        labeled = fallback.astype(np.int32)
        n = 1

    lesions: list[LesionMask] = []
    for label_id in range(1, n + 1):
        component = labeled == label_id
        area = int(component.sum())
        if area < 20:
            continue

        if volume.shape[0] == 1:
            mask3d = component[np.newaxis, ...]
        else:
            mask3d = np.zeros_like(volume, dtype=bool)
            mask3d[z_mid] = component

        confidence = min(0.85, 0.45 + area / (slice_img.size * 0.05))
        lesions.append(
            LesionMask(lesion_id=len(lesions) + 1, mask=mask3d, in_plane_confidence=confidence)
        )

    if not lesions:
        raise SegmentationError("Stub segmentation found no lesion candidates")

    # Keep top 5 by volume
    lesions.sort(key=lambda l: l.mask.sum(), reverse=True)
    lesions = lesions[:5]
    for i, lesion in enumerate(lesions, start=1):
        lesion.lesion_id = i

    global_conf = float(np.mean([l.in_plane_confidence for l in lesions]))
    return SegmentationResult(lesions=lesions, global_confidence=global_conf)


def _segment_monai(volume: np.ndarray) -> SegmentationResult:
    """MONAI bundle inference when GPU stack is available."""
    try:
        import torch
        from monai.bundle import ConfigParser
    except ImportError as exc:
        raise SegmentationError(
            "MONAI backend requires torch and monai. "
            "Use SEGMENTATION_BACKEND=stub for local demo."
        ) from exc

    from config_medical import MONAI_BUNDLE_DIR

    if not MONAI_BUNDLE_DIR:
        raise SegmentationError(
            "MONAI_BUNDLE_DIR is not set. Download a BraTS-style bundle and set the path."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = ConfigParser()
    parser.read_config(MONAI_BUNDLE_DIR)

    # Bundle-specific; users configure inference on RunPod per docs.
    inferer = parser.get_parsed_content("inferer")
    network = parser.get_parsed_content("network").to(device)

    tensor = torch.from_numpy(volume[np.newaxis, np.newaxis, ...]).float().to(device)
    with torch.no_grad():
        logits = inferer(tensor, network)
    pred = (torch.sigmoid(logits) > 0.5).cpu().numpy()[0, 0]

    labeled, n = ndimage.label(pred)
    lesions: list[LesionMask] = []
    for label_id in range(1, n + 1):
        mask = labeled == label_id
        if mask.sum() < 10:
            continue
        lesions.append(
            LesionMask(lesion_id=len(lesions) + 1, mask=mask, in_plane_confidence=0.75)
        )

    if not lesions:
        raise SegmentationError("MONAI segmentation produced no lesions")

    return SegmentationResult(lesions=lesions, global_confidence=0.75)
