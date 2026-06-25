"""Atlas-anchored 3D volume synthesis from a single patient slice (Phase 6a)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

from config_pipeline import ATLAS_BRAIN_TEMPLATE
from pipeline.ingest.images import SliceVolume

logger = logging.getLogger(__name__)


def _normalize_volume(vol: np.ndarray) -> np.ndarray:
    out = vol.astype(np.float32)
    out = out - out.min()
    denom = out.max() or 1.0
    return out / denom


def _load_atlas_volume() -> np.ndarray | None:
    if not ATLAS_BRAIN_TEMPLATE.is_file():
        return None
    try:
        import nibabel as nib
    except ImportError:
        logger.warning("nibabel not installed — atlas synthesis skipped")
        return None

    data = nib.load(str(ATLAS_BRAIN_TEMPLATE)).get_fdata().astype(np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        logger.warning("Unexpected atlas shape %s", data.shape)
        return None
    return _normalize_volume(data)


def _resize_mask(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    if mask.shape == (h, w):
        return mask.astype(np.float32)
    factors = (h / mask.shape[0], w / mask.shape[1])
    return (zoom(mask.astype(np.float32), factors, order=0) > 0.5).astype(np.float32)


def synthesize_atlas_anchored_volume(
    volume: SliceVolume,
    *,
    target_z: int,
    organ_mask_2d: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    """
    Build a full-depth brain volume from one patient slice + population atlas.

    The anchor plane preserves the uploaded slice (measured). Other planes are
    atlas-guided AI estimates blended toward the patient appearance near the anchor.
    """
    atlas = _load_atlas_volume()
    patient_slice = volume.data[0].astype(np.float32)
    h, w = patient_slice.shape

    if atlas is None:
        return _fallback_z_expansion(patient_slice, target_z), "single_slice_z_expansion"

    az, ah, aw = atlas.shape
    atlas_resampled = zoom(
        atlas,
        (target_z / az, h / ah, w / aw),
        order=1,
    ).astype(np.float32)

    anchor_z = target_z // 2
    patient_norm = _normalize_volume(patient_slice)
    synth = atlas_resampled.copy()

    if organ_mask_2d is not None:
        blend = _resize_mask(np.asarray(organ_mask_2d), (h, w))
        # Soft boundary for smoother transitions
        blend = gaussian_filter(blend, sigma=2.0)
        blend = np.clip(blend, 0.0, 1.0)
    else:
        blend = np.ones((h, w), dtype=np.float32)

    # Lock anchor plane: patient intensity inside brain ROI, atlas outside skull
    atlas_plane = synth[anchor_z]
    synth[anchor_z] = patient_norm * blend + atlas_plane * (1.0 - blend)

    # Pull neighboring slices toward patient appearance (atlas-shaped 3D estimate)
    for dz in range(1, min(6, anchor_z + 1)):
        falloff = 0.55 * (1.0 - dz / 6.0)
        for z in (anchor_z - dz, anchor_z + dz):
            if 0 <= z < target_z:
                wgt = blend * falloff
                synth[z] = synth[z] * (1.0 - wgt) + patient_norm * wgt

    # Light smoothing along z except anchor
    smoothed = gaussian_filter(synth, sigma=(0.6, 0.0, 0.0))
    for z in range(target_z):
        if z != anchor_z:
            synth[z] = 0.7 * synth[z] + 0.3 * smoothed[z]
    synth[anchor_z] = patient_norm * blend + atlas_plane * (1.0 - blend)

    return np.clip(synth, 0.0, 1.0).astype(np.float32), "single_slice_atlas_anchored"


def _fallback_z_expansion(patient_slice: np.ndarray, target_z: int) -> np.ndarray:
    """Used only when atlas template is missing."""
    h, w = patient_slice.shape
    anchor = target_z // 2
    synth = np.zeros((target_z, h, w), dtype=np.float32)
    mid = anchor
    synth[mid] = patient_slice
    for z in range(target_z):
        if z == mid:
            continue
        weight = 1.0 - abs(z - mid) / max(mid, 1)
        synth[z] = patient_slice * (0.85 + 0.15 * weight)
    return synth
