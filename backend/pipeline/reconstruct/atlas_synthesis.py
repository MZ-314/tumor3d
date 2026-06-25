"""Atlas-anchored 3D volume synthesis from a single patient slice (Phase 6a)."""

from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

from config_pipeline import ATLAS_BRAIN_TEMPLATE
from pipeline.ingest.images import SliceVolume
from pipeline.reconstruct.view_orient import (
    build_brain_envelope,
    fit_atlas_plane_to_patient,
    orient_atlas_volume,
)
from shared.schemas.pydantic.pipeline import MriView

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
    mri_view: MriView = MriView.UNKNOWN,
) -> tuple[np.ndarray, str]:
    """
    Build a full-depth brain volume from one patient slice + population atlas.

    - Anchor plane: 100% patient DICOM (no atlas overlay).
    - Other planes: view-oriented atlas, scaled to patient brain ROI, inside envelope only.
    """
    atlas = _load_atlas_volume()
    patient_slice = volume.data[0].astype(np.float32)
    h, w = patient_slice.shape
    patient_norm = _normalize_volume(patient_slice)

    if atlas is None:
        return _fallback_z_expansion(patient_slice, target_z), "single_slice_z_expansion"

    mask = _resize_mask(np.asarray(organ_mask_2d), (h, w)) if organ_mask_2d is not None else None

    oriented = orient_atlas_volume(atlas, mri_view)
    az, ah, aw = oriented.shape
    atlas_resampled = zoom(
        oriented,
        (target_z / az, h / ah, w / aw),
        order=1,
    ).astype(np.float32)

    anchor_z = target_z // 2
    synth = np.zeros((target_z, h, w), dtype=np.float32)

    # Fit each atlas slice to patient brain scale (in-plane), keep off-slice anatomy consistent
    ref_atlas_plane = fit_atlas_plane_to_patient(
        atlas_resampled[anchor_z],
        patient_norm,
        mask,
    )
    for z in range(target_z):
        if z == anchor_z:
            continue
        plane = fit_atlas_plane_to_patient(atlas_resampled[z], patient_norm, mask)
        # Blend toward reference so stack does not jump slice-to-slice
        synth[z] = 0.65 * plane + 0.35 * ref_atlas_plane

    # Anchor is exactly the uploaded slice — never superimpose atlas here
    synth[anchor_z] = patient_norm

    if mask is not None:
        envelope = build_brain_envelope((target_z, h, w), mask, anchor_z)
        bg = np.median(patient_norm[~mask.astype(bool)]) if mask.any() else 0.0
        for z in range(target_z):
            if z == anchor_z:
                continue
            env = envelope[z]
            synth[z] = synth[z] * env + bg * (1.0 - env)

    # Mild through-plane smooth away from anchor
    smoothed = gaussian_filter(synth, sigma=(0.5, 0.8, 0.8))
    for z in range(target_z):
        if z == anchor_z:
            continue
        dz = abs(z - anchor_z)
        wgt = min(0.25, 0.08 * dz)
        synth[z] = (1.0 - wgt) * synth[z] + wgt * smoothed[z]
    synth[anchor_z] = patient_norm

    strategy = f"single_slice_atlas_anchored_{mri_view.value}"
    return np.clip(synth, 0.0, 1.0).astype(np.float32), strategy


def _fallback_z_expansion(patient_slice: np.ndarray, target_z: int) -> np.ndarray:
    h, w = patient_slice.shape
    anchor = target_z // 2
    synth = np.zeros((target_z, h, w), dtype=np.float32)
    synth[anchor] = patient_slice
    for z in range(target_z):
        if z == anchor:
            continue
        weight = 1.0 - abs(z - anchor) / max(anchor, 1)
        synth[z] = patient_slice * (0.85 + 0.15 * weight)
    return synth
