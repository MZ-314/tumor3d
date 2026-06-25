"""3D brain-shaped envelope (ellipsoid) for single-slice volume synthesis."""

from __future__ import annotations

import numpy as np


def build_brain_envelope_3d(
    shape_zyx: tuple[int, int, int],
    organ_mask_2d: np.ndarray,
    anchor_z: int,
    *,
    z_taper: float = 0.92,
) -> np.ndarray:
    """
    Soft 3D ellipsoid mask aligned to the anchor slice brain ROI.

    Removes rectangular-slab corners and tapers superior/inferior like a head.
    """
    z_count, h, w = shape_zyx
    mask = organ_mask_2d.astype(bool)
    if not mask.any():
        yy, xx = np.ogrid[:h, :w]
        cy, cx = h / 2.0, w / 2.0
        ry = h * 0.38
        rx = w * 0.38
    else:
        rows, cols = np.where(mask)
        cy = (rows.min() + rows.max()) / 2.0
        cx = (cols.min() + cols.max()) / 2.0
        ry = (rows.max() - rows.min() + 1) / 2.0 * 1.05
        rx = (cols.max() - cols.min() + 1) / 2.0 * 1.05

    rz = max(anchor_z, z_count - anchor_z - 1, 1) * 1.08
    zz, yy, xx = np.ogrid[:z_count, :h, :w]
    dz = (zz - anchor_z) / (rz + 1e-6)
    dy = (yy - cy) / (ry + 1e-6)
    dx = (xx - cx) / (rx + 1e-6)
    ellipsoid = (dz * dz + dy * dy + dx * dx) <= 1.0

    envelope = ellipsoid.astype(np.float32)
    # Softer falloff at surface
    dist = np.sqrt(dz * dz + dy * dy + dx * dx)
    envelope = np.clip(1.0 - (dist - 0.75) / 0.35, 0.0, 1.0) * ellipsoid.astype(np.float32)
    for zi in range(z_count):
        taper = max(0.12, 1.0 - z_taper * abs(zi - anchor_z) / max(anchor_z, 1))
        envelope[zi] *= taper
    return np.clip(envelope, 0.0, 1.0).astype(np.float32)


def apply_brain_envelope(
    volume: np.ndarray,
    envelope: np.ndarray,
    *,
    anchor_z: int,
    anchor_plane: np.ndarray,
    background: float = 0.0,
) -> np.ndarray:
    """Mask volume to brain ellipsoid; lock anchor plane."""
    out = volume * envelope + background * (1.0 - envelope)
    out[anchor_z] = anchor_plane
    return out.astype(np.float32)
