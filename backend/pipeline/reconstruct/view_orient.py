"""MRI view detection and view-aware atlas orientation."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom

from shared.schemas.pydantic.pipeline import MriView


def detect_mri_view_from_pixels(
    slice_img: np.ndarray,
    organ_mask: np.ndarray | None = None,
) -> MriView:
    """
    Estimate axial / sagittal / coronal when DICOM orientation tags are missing.

    Uses brain ROI eccentricity: sagittal slices are usually tall-narrow;
    axial slices are rounder in-plane.
    """
    img = slice_img.astype(np.float32)
    mask = organ_mask.astype(bool) if organ_mask is not None else img > np.percentile(img, 60)

    if not mask.any():
        h, w = img.shape
        return MriView.SAGITTAL if h > w * 1.15 else MriView.AXIAL if w >= h else MriView.UNKNOWN

    rows, cols = np.where(mask)
    h_span = float(rows.max() - rows.min() + 1)
    w_span = float(cols.max() - cols.min() + 1)
    aspect = h_span / max(w_span, 1.0)

    if aspect > 1.35:
        return MriView.SAGITTAL
    if aspect < 0.78:
        return MriView.AXIAL
    return MriView.CORONAL


def orient_atlas_volume(atlas: np.ndarray, view: MriView) -> np.ndarray:
    """
    Reorder atlas axes so axis 0 is the through-plane stack direction for `view`.

    Assumes atlas on disk is stored with axial slices along axis 0 (Z, Y, X).
    """
    if atlas.ndim != 3:
        raise ValueError(f"Expected 3D atlas, got {atlas.shape}")

    if view in (MriView.AXIAL, MriView.UNKNOWN):
        return atlas
    if view == MriView.SAGITTAL:
        return np.transpose(atlas, (2, 1, 0))
    if view == MriView.CORONAL:
        return np.transpose(atlas, (1, 2, 0))
    return atlas


def atlas_reference_slice(atlas: np.ndarray, view: MriView) -> np.ndarray:
    """Single 2D atlas slice matching the patient's imaging plane."""
    oriented = orient_atlas_volume(atlas, view)
    return oriented[oriented.shape[0] // 2].astype(np.float32)


def fit_atlas_plane_to_patient(
    atlas_plane: np.ndarray,
    patient_plane: np.ndarray,
    organ_mask: np.ndarray | None,
) -> np.ndarray:
    """Scale and place atlas in-plane to align with patient brain ROI."""
    ph, pw = patient_plane.shape
    plane = atlas_plane.astype(np.float32)
    plane = plane - plane.min()
    plane = plane / (plane.max() or 1.0)

    if organ_mask is not None and organ_mask.any():
        rows, cols = np.where(organ_mask)
        cy = (rows.min() + rows.max()) / 2.0
        cx = (cols.min() + cols.max()) / 2.0
        brain_h = rows.max() - rows.min() + 1
        brain_w = cols.max() - cols.min() + 1
    else:
        cy, cx = ph / 2.0, pw / 2.0
        brain_h, brain_w = ph * 0.7, pw * 0.7

    ath, aw = plane.shape
    atlas_mask = plane > np.percentile(plane, 55)
    if atlas_mask.any():
        ar, ac = np.where(atlas_mask)
        ah = ar.max() - ar.min() + 1
        aw_ = ac.max() - ac.min() + 1
    else:
        ah, aw_ = ath * 0.6, aw * 0.6

    scale_y = brain_h / max(ah, 1.0)
    scale_x = brain_w / max(aw_, 1.0)
    scale = float(min(scale_y, scale_x) * 0.95)

    scaled = zoom(plane, (scale, scale), order=1)
    sh, sw = scaled.shape
    out = np.zeros((ph, pw), dtype=np.float32)

    y0 = int(round(cy - sh / 2.0))
    x0 = int(round(cx - sw / 2.0))
    y1, x1 = y0 + sh, x0 + sw
    sy0, sx0 = max(0, -y0), max(0, -x0)
    dy0, dx0 = max(0, y0), max(0, x0)
    dy1 = min(ph, y1)
    dx1 = min(pw, x1)
    sy1 = sy0 + (dy1 - dy0)
    sx1 = sx0 + (dx1 - dx0)
    if dy1 > dy0 and dx1 > dx0:
        out[dy0:dy1, dx0:dx1] = scaled[sy0:sy1, sx0:sx1]
    return out


def build_brain_envelope(
    shape_zyx: tuple[int, int, int],
    organ_mask_2d: np.ndarray,
    anchor_z: int,
    *,
    z_sigma: float = 8.0,
) -> np.ndarray:
    """Soft 3D mask: brain column through-stack, strongest at anchor."""
    z, h, w = shape_zyx
    mask = organ_mask_2d.astype(np.float32)
    if mask.shape != (h, w):
        mask = zoom(mask, (h / mask.shape[0], w / mask.shape[1]), order=0)
    mask = np.clip(mask, 0.0, 1.0)

    envelope = np.zeros((z, h, w), dtype=np.float32)
    for zi in range(z):
        dz = abs(zi - anchor_z)
        weight = float(np.exp(-0.5 * (dz / z_sigma) ** 2))
        envelope[zi] = np.clip(mask * (0.35 + 0.65 * weight), 0.0, 1.0)
    return envelope
