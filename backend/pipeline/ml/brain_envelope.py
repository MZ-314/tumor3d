"""Through-plane masks for single-slice volumes — extrude 2D ROI, no 3D sphere."""

from __future__ import annotations

import numpy as np


def build_extruded_mask_3d(
    shape_zyx: tuple[int, int, int],
    organ_mask_2d: np.ndarray,
    anchor_z: int,
) -> np.ndarray:
    """
    Extrude the in-plane brain mask along the stack axis.

    Cross-section stays the shape of the real slice (MedSAM ROI), not an ellipsoid dome.
    Intensity fades toward stack ends only — does not invent a spherical scalp.
    """
    z_count, h, w = shape_zyx
    mask2d = organ_mask_2d.astype(np.float32)
    if mask2d.shape != (h, w):
        from scipy.ndimage import zoom

        mask2d = zoom(mask2d, (h / mask2d.shape[0], w / mask2d.shape[1]), order=0)

    out = np.zeros((z_count, h, w), dtype=np.float32)
    half = max(anchor_z, z_count - anchor_z - 1, 1)
    for zi in range(z_count):
        dz = abs(zi - anchor_z) / half
        # Soft fade at stack ends; in-plane shape unchanged
        weight = float(max(0.0, 1.0 - dz**1.2))
        out[zi] = mask2d * weight
    out[anchor_z] = np.clip(mask2d, 0.0, 1.0)
    return out


def apply_volume_mask(
    volume: np.ndarray,
    mask_3d: np.ndarray,
    *,
    anchor_z: int,
    anchor_plane: np.ndarray,
    background: float = 0.0,
) -> np.ndarray:
    """Mask volume to extruded ROI; anchor plane stays exact."""
    out = volume * mask_3d + background * (1.0 - mask_3d)
    out[anchor_z] = anchor_plane
    return out.astype(np.float32)
