"""3D brain volume synthesis: atlas slice search, 2D registration warp, patient anchor lock."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from skimage import exposure

from config_pipeline import ATLAS_BRAIN_TEMPLATE
from pipeline.ingest.images import SliceVolume
from pipeline.reconstruct.view_orient import (
    build_brain_envelope,
    orient_atlas_volume,
)
from shared.schemas.pydantic.pipeline import AtlasWarpResult, MriView

logger = logging.getLogger(__name__)


def _normalize(vol: np.ndarray) -> np.ndarray:
    v = vol.astype(np.float32)
    v = v - v.min()
    return v / (v.max() or 1.0)


def load_oriented_atlas(mri_view: MriView) -> np.ndarray | None:
    if not ATLAS_BRAIN_TEMPLATE.is_file():
        return None
    import nibabel as nib

    data = nib.load(str(ATLAS_BRAIN_TEMPLATE)).get_fdata().astype(np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        return None
    return orient_atlas_volume(_normalize(data), mri_view)


def find_best_atlas_slice_index(
    patient_slice: np.ndarray,
    atlas_oriented: np.ndarray,
    organ_mask: np.ndarray | None,
) -> int:
    """Pick atlas slice index with highest masked normalized cross-correlation."""
    patient = _normalize(patient_slice)
    az = atlas_oriented.shape[0]
    best_i = az // 2
    best_score = -1.0

    mask = None
    if organ_mask is not None and organ_mask.any():
        mask = organ_mask.astype(bool)

    for i in range(az):
        plane = atlas_oriented[i]
        ph, pw = patient.shape
        if plane.shape != (ph, pw):
            plane = zoom(plane, (ph / plane.shape[0], pw / plane.shape[1]), order=1)

        atlas_n = _normalize(plane)
        if mask is not None:
            p = patient[mask]
            a = atlas_n[mask]
            if p.size < 64:
                continue
            p = p - p.mean()
            a = a - a.mean()
            denom = float(np.linalg.norm(p) * np.linalg.norm(a)) or 1.0
            score = float(np.dot(p, a) / denom)
        else:
            score = float(np.corrcoef(patient.ravel(), atlas_n.ravel())[0, 1])

        if score > best_score:
            best_score = score
            best_i = i

    logger.info("Atlas slice match: index=%d score=%.3f (of %d)", best_i, best_score, az)
    return best_i


def _load_sitk_transform(transform_path: Path | None):
    if transform_path is None or not transform_path.is_file():
        return None
    try:
        import SimpleITK as sitk

        return sitk.ReadTransform(str(transform_path))
    except Exception as exc:
        logger.warning("Could not load atlas transform: %s", exc)
        return None


def _warp_slice_to_patient_plane(
    atlas_slice: np.ndarray,
    patient_slice: np.ndarray,
    spacing_xy: tuple[float, float],
    transform,
) -> np.ndarray:
    import SimpleITK as sitk

    ph, pw = patient_slice.shape
    if atlas_slice.shape != (ph, pw):
        atlas_slice = zoom(
            atlas_slice.astype(np.float32),
            (ph / atlas_slice.shape[0], pw / atlas_slice.shape[1]),
            order=1,
        )

    ref = sitk.GetImageFromArray(patient_slice.astype(np.float32))
    ref.SetSpacing((spacing_xy[1], spacing_xy[0]))

    moving = sitk.GetImageFromArray(atlas_slice.astype(np.float32))
    moving.SetSpacing((spacing_xy[1], spacing_xy[0]))

    if transform is None:
        return _normalize(atlas_slice)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    resampler.SetTransform(transform)
    warped = sitk.GetArrayFromImage(resampler.Execute(moving))
    return _normalize(warped)


def build_registered_atlas_volume(
    volume: SliceVolume,
    *,
    target_z: int,
    organ_mask_2d: np.ndarray | None,
    mri_view: MriView,
    atlas_warp: AtlasWarpResult | None,
    work_dir: Path | None,
) -> tuple[np.ndarray, str, int]:
    """
    Construct patient-specific 3D brain volume from oriented atlas + 2D registration.

    Returns (volume_zyx, strategy_name, matched_atlas_index).
    """
    patient_slice = volume.data[0].astype(np.float32)
    h, w = patient_slice.shape
    patient_norm = _normalize(patient_slice)
    row_sp, col_sp = volume.pixel_spacing_mm
    spacing_xy = (row_sp, col_sp)

    mask = None
    if organ_mask_2d is not None:
        mask = np.asarray(organ_mask_2d, dtype=bool)
        if mask.shape != (h, w):
            mask = zoom(mask.astype(np.float32), (h / mask.shape[0], w / mask.shape[1]), order=0) > 0.5

    atlas = load_oriented_atlas(mri_view)
    if atlas is None:
        return _fallback_expansion(patient_norm, target_z), "single_slice_fallback", 0

    best_i = (
        atlas_warp.estimated_slice_index
        if atlas_warp is not None and atlas_warp.estimated_slice_index is not None
        else find_best_atlas_slice_index(patient_norm, atlas, mask)
    )

    transform_path = None
    if atlas_warp is not None and atlas_warp.transform_path and work_dir is not None:
        transform_path = work_dir / atlas_warp.transform_path
    transform = _load_sitk_transform(transform_path)

    az = atlas.shape[0]
    warped_stack = np.zeros((az, h, w), dtype=np.float32)
    for i in range(az):
        warped_stack[i] = _warp_slice_to_patient_plane(
            atlas[i], patient_norm, spacing_xy, transform
        )

    anchor_z = target_z // 2
    synth = np.zeros((target_z, h, w), dtype=np.float32)

    for out_z in range(target_z):
        delta = out_z - anchor_z
        src_i = best_i + delta
        if 0 <= src_i < az:
            synth[out_z] = warped_stack[src_i]
        elif src_i < 0:
            synth[out_z] = warped_stack[0]
        else:
            synth[out_z] = warped_stack[az - 1]

    # Lock measured anchor — patient slice is ground truth on this plane
    synth[anchor_z] = patient_norm

    # Harmonize off-slice intensity to patient (inside brain ROI)
    if mask is not None and mask.any():
        bg = float(np.median(patient_norm[~mask]))
        for z in range(target_z):
            if z == anchor_z:
                continue
            matched = exposure.match_histograms(synth[z], patient_norm, channel_axis=None)
            blend = mask.astype(np.float32)
            synth[z] = matched * blend + bg * (1.0 - blend)
    else:
        for z in range(target_z):
            if z == anchor_z:
                continue
            synth[z] = exposure.match_histograms(synth[z], patient_norm, channel_axis=None)

    synth[anchor_z] = patient_norm

    if mask is not None:
        envelope = build_brain_envelope((target_z, h, w), mask.astype(np.float32), anchor_z)
        bg = float(np.median(patient_norm[~mask])) if mask.any() else 0.0
        for z in range(target_z):
            if z == anchor_z:
                continue
            env = envelope[z]
            synth[z] = synth[z] * env + bg * (1.0 - env)

    # Through-plane continuity (preserve anchor)
    smoothed = gaussian_filter(synth, sigma=(0.45, 0.6, 0.6))
    for z in range(target_z):
        if z == anchor_z:
            continue
        dz = abs(z - anchor_z)
        wgt = min(0.2, 0.05 * dz)
        synth[z] = (1.0 - wgt) * synth[z] + wgt * smoothed[z]
    synth[anchor_z] = patient_norm

    strategy = f"registered_atlas_3d_{mri_view.value}"
    return np.clip(synth, 0.0, 1.0).astype(np.float32), strategy, best_i


def _fallback_expansion(patient: np.ndarray, target_z: int) -> np.ndarray:
    h, w = patient.shape
    anchor = target_z // 2
    synth = np.zeros((target_z, h, w), dtype=np.float32)
    synth[anchor] = patient
    for z in range(target_z):
        if z == anchor:
            continue
        t = 1.0 - abs(z - anchor) / max(anchor, 1)
        synth[z] = patient * (0.8 + 0.2 * t)
    return synth
