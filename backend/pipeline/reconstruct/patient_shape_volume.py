"""Patient-primary 3D brain volume: shape from MedSAM, intensity from anchor slice."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter, zoom

from shared.schemas.pydantic.pipeline import MriView


def _normalize(vol: np.ndarray) -> np.ndarray:
    v = vol.astype(np.float32)
    v = v - v.min()
    return v / (v.max() or 1.0)


def _fit_organ_mask(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    m = mask.astype(bool)
    if m.shape != (h, w):
        m = zoom(m.astype(np.float32), (h / m.shape[0], w / m.shape[1]), order=0) > 0.5
    m = binary_fill_holes(m)
    m = binary_closing(m, iterations=2)
    return m


def _ellipsoid_mask_3d(
    organ_mask_2d: np.ndarray,
    target_z: int,
    anchor_z: int,
    *,
    taper: float = 0.42,
) -> np.ndarray:
    """Smooth 3D brain envelope — tapers away from anchor like a head."""
    h, w = organ_mask_2d.shape
    rows, cols = np.where(organ_mask_2d)
    if rows.size == 0:
        yy, xx = np.ogrid[:h, :w]
        cy, cx = h / 2.0, w / 2.0
        ry, rx = h * 0.35, w * 0.35
    else:
        cy = (rows.min() + rows.max()) / 2.0
        cx = (cols.min() + cols.max()) / 2.0
        ry = (rows.max() - rows.min() + 1) / 2.0 * 1.08
        rx = (cols.max() - cols.min() + 1) / 2.0 * 1.08

    yy, xx = np.ogrid[:h, :w]
    envelope = np.zeros((target_z, h, w), dtype=np.float32)

    for z in range(target_z):
        dz = abs(z - anchor_z) / max(anchor_z, 1)
        scale = max(0.18, 1.0 - taper * dz)
        ellipse = ((yy - cy) / (ry * scale + 1e-6)) ** 2 + ((xx - cx) / (rx * scale + 1e-6)) ** 2 <= 1.0
        slice_mask = ellipse & organ_mask_2d
        envelope[z] = slice_mask.astype(np.float32)

    return np.clip(envelope, 0.0, 1.0)


def build_patient_primary_volume(
    patient_slice: np.ndarray,
    *,
    target_z: int,
    organ_mask_2d: np.ndarray | None,
    mri_view: MriView,
    atlas_hint: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    """
    Build 3D volume dominated by the patient's slice, not atlas texture.

    - Anchor plane = exact patient intensities.
    - Off-slice = intensity propagated inside an ellipsoid brain envelope.
    - Optional atlas_hint blended at low weight for depth shading only.
    """
    patient = _normalize(patient_slice)
    h, w = patient.shape
    anchor_z = target_z // 2

    if organ_mask_2d is not None and organ_mask_2d.any():
        mask = _fit_organ_mask(organ_mask_2d, (h, w))
    else:
        mask = patient > np.percentile(patient, 35)

    envelope = _ellipsoid_mask_3d(mask, target_z, anchor_z)
    bg = float(np.median(patient[~mask])) if mask.any() else 0.0

    vol = np.zeros((target_z, h, w), dtype=np.float32)
    vol[anchor_z] = patient

    # Propagate anchor intensity along Z inside brain envelope (patient-looking tissue)
    for _ in range(4):
        blurred = gaussian_filter(vol, sigma=(2.8, 1.0, 1.0))
        vol = blurred * envelope + bg * (1.0 - envelope)
        vol[anchor_z] = patient

    # Sagittal/coronal: slightly stronger in-plane smooth (anatomy varies along depth axis)
    if mri_view in (MriView.SAGITTAL, MriView.CORONAL):
        vol = gaussian_filter(vol, sigma=(1.2, 0.6, 0.6))
        vol = vol * envelope + bg * (1.0 - envelope)
        vol[anchor_z] = patient

    if atlas_hint is not None and atlas_hint.shape == vol.shape:
        # Atlas only modulates depth shading (15%), not replace patient appearance
        hint = atlas_hint * envelope
        vol = 0.88 * vol + 0.12 * hint
        vol[anchor_z] = patient

    vol = vol * envelope + bg * (1.0 - envelope)
    vol[anchor_z] = patient

    return np.clip(vol, 0.0, 1.0).astype(np.float32), f"patient_primary_{mri_view.value}"
