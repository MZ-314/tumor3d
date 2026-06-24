"""Load one or more imaging slices into a normalized volume."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from config_medical import DEFAULT_PIXEL_SPACING_MM, DEFAULT_SLICE_THICKNESS_MM


@dataclass
class SliceVolume:
    """Grayscale volume in shape (Z, H, W), values 0–1."""

    data: np.ndarray
    pixel_spacing_mm: tuple[float, float]
    slice_thickness_mm: float
    source_paths: list[Path]


def _load_single_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in {".dcm", ".dicom"}:
        try:
            import pydicom
        except ImportError as exc:
            raise ValueError(
                "DICOM upload requires pydicom. Install with: pip install pydicom"
            ) from exc
        ds = pydicom.dcmread(str(path))
        arr = ds.pixel_array.astype(np.float32)
        if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
            arr = arr * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
        arr = arr - arr.min()
        denom = arr.max() or 1.0
        arr = arr / denom
        return arr

    with Image.open(path) as img:
        gray = img.convert("L")
        arr = np.asarray(gray, dtype=np.float32) / 255.0
    return arr


def load_slice_volume(paths: list[Path]) -> SliceVolume:
    if not paths:
        raise ValueError("At least one image path is required")

    arrays = [_load_single_array(p) for p in paths]
    h, w = arrays[0].shape
    for i, arr in enumerate(arrays[1:], start=1):
        if arr.shape != (h, w):
            raise ValueError(
                f"Slice {i} shape {arr.shape} does not match first slice {(h, w)}"
            )

    if len(arrays) == 1:
        volume = arrays[0][np.newaxis, ...]
    else:
        volume = np.stack(arrays, axis=0)

    return SliceVolume(
        data=volume,
        pixel_spacing_mm=(DEFAULT_PIXEL_SPACING_MM, DEFAULT_PIXEL_SPACING_MM),
        slice_thickness_mm=DEFAULT_SLICE_THICKNESS_MM,
        source_paths=list(paths),
    )


def save_png_overlay(base_slice: np.ndarray, mask: np.ndarray, out_path: Path) -> None:
    """Save RGB overlay of mask on grayscale slice."""
    base_u8 = (np.clip(base_slice, 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([base_u8, base_u8, base_u8], axis=-1)
    overlay = rgb.copy()
    overlay[mask > 0, 0] = np.minimum(255, overlay[mask > 0, 0] + 120)
    overlay[mask > 0, 1] = np.maximum(0, overlay[mask > 0, 1] - 40)
    Image.fromarray(overlay).save(out_path)
