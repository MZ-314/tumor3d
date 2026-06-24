"""Export imaging volumes and tumor masks as NIfTI for the web viewer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pipeline.ingest.images import SliceVolume


def _affine_from_volume(volume: SliceVolume) -> np.ndarray:
    """RAS-like affine from slice spacing (mm)."""
    sy, sx = volume.pixel_spacing_mm
    sz = volume.slice_thickness_mm
    return np.array(
        [
            [sx, 0.0, 0.0, 0.0],
            [0.0, sy, 0.0, 0.0],
            [0.0, 0.0, sz, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def save_volume_nifti(volume: SliceVolume, out_path: Path) -> Path:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError("nibabel required: pip install nibabel") from exc

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # NIfTI expects (X, Y, Z); our data is (Z, H, W) = (Z, Y, X)
    data = volume.data.astype(np.float32)
    data_xyz = np.transpose(data, (2, 1, 0))
    img = nib.Nifti1Image(data_xyz, _affine_from_volume(volume))
    nib.save(img, str(out_path))
    return out_path


def save_mask_nifti(mask: np.ndarray, volume: SliceVolume, out_path: Path) -> Path:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError("nibabel required: pip install nibabel") from exc

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if mask.shape != volume.data.shape:
        raise ValueError(f"Mask shape {mask.shape} != volume {volume.data.shape}")
    data_xyz = np.transpose(mask.astype(np.uint8), (2, 1, 0))
    img = nib.Nifti1Image(data_xyz, _affine_from_volume(volume))
    nib.save(img, str(out_path))
    return out_path


def combined_lesion_mask(lesions) -> np.ndarray | None:
    if not lesions:
        return None
    mask = np.zeros_like(lesions[0].mask, dtype=np.uint8)
    for i, lesion in enumerate(lesions, start=1):
        mask[lesion.mask] = np.maximum(mask[lesion.mask], i)
    return mask
