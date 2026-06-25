"""Load one or more imaging slices into a normalized volume."""

from __future__ import annotations

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


@dataclass
class _DicomSlice:
    array: np.ndarray
    sort_key: float
    row_spacing: float
    col_spacing: float
    thickness: float


def _is_dicom(path: Path) -> bool:
    return path.suffix.lower() in {".dcm", ".dicom"}


def _load_png_array(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        gray = img.convert("L")
        return np.asarray(gray, dtype=np.float32) / 255.0


def _rgb_to_grayscale(plane: np.ndarray) -> np.ndarray:
    """Convert (H, W, 3+) RGB plane to grayscale."""
    rgb = plane[..., :3].astype(np.float32)
    return np.tensordot(rgb, [0.299, 0.587, 0.114], axes=([-1], [0]))


def _dicom_to_grayscale_2d(arr: np.ndarray, ds) -> np.ndarray:
    """Reduce pydicom pixel_array to 2D float32 (H, W)."""
    if arr.ndim == 2:
        return arr

    if arr.ndim != 3:
        raise ValueError(f"Unsupported DICOM pixel array shape {arr.shape}")

    frames = int(getattr(ds, "NumberOfFrames", 0) or 0)
    if frames > 1 and arr.shape[0] == frames:
        return arr[frames // 2]

    # RGB / RGBA stored as (rows, cols, channels)
    if arr.shape[-1] in (3, 4) and arr.shape[-1] < arr.shape[0]:
        return _rgb_to_grayscale(arr)

    # Planar RGB: (channels, rows, cols)
    if arr.shape[0] in (3, 4) and arr.shape[0] < min(arr.shape[1], arr.shape[2]):
        plane = arr[:3, ...].transpose(1, 2, 0)
        return _rgb_to_grayscale(plane)

    # Heuristic: (frames, H, W) without NumberOfFrames metadata
    if arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
        return arr[arr.shape[0] // 2]

    return arr.mean(axis=-1)


def _load_dicom_slice(path: Path) -> _DicomSlice:
    import pydicom

    try:
        ds = pydicom.dcmread(str(path))
        arr = ds.pixel_array.astype(np.float32)
    except Exception as exc:
        msg = str(exc)
        if "Unable to decompress" in msg or "plugins are missing" in msg:
            raise ValueError(
                "DICOM is JPEG-compressed. On RunPod install codecs, then restart uvicorn:\n"
                "  pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg\n"
                "Or: pip install python-gdcm"
            ) from exc
        raise

    arr = _dicom_to_grayscale_2d(arr, ds)

    if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
        arr = arr * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    arr = arr - arr.min()
    denom = arr.max() or 1.0
    arr = arr / denom

    row_sp = col_sp = DEFAULT_PIXEL_SPACING_MM
    thickness = DEFAULT_SLICE_THICKNESS_MM
    sort_key = 0.0

    if hasattr(ds, "PixelSpacing") and ds.PixelSpacing is not None:
        row_sp = float(ds.PixelSpacing[0])
        col_sp = float(ds.PixelSpacing[1])
    if hasattr(ds, "SliceThickness") and ds.SliceThickness:
        thickness = float(ds.SliceThickness)
    if hasattr(ds, "InstanceNumber"):
        sort_key = float(ds.InstanceNumber)
    elif hasattr(ds, "ImagePositionPatient") and ds.ImagePositionPatient is not None:
        sort_key = float(ds.ImagePositionPatient[2])

    return _DicomSlice(
        array=arr,
        sort_key=sort_key,
        row_spacing=row_sp,
        col_spacing=col_sp,
        thickness=thickness,
    )


def load_slice_volume(paths: list[Path]) -> SliceVolume:
    if not paths:
        raise ValueError("At least one image path is required")

    if all(_is_dicom(p) for p in paths):
        try:
            import pydicom  # noqa: F401
        except ImportError as exc:
            raise ValueError(
                "DICOM upload requires pydicom. Install with: pip install pydicom"
            ) from exc

        dicom_slices = [_load_dicom_slice(p) for p in paths]
        dicom_slices.sort(key=lambda s: s.sort_key)
        arrays = [s.array for s in dicom_slices]
        h, w = arrays[0].shape
        for i, arr in enumerate(arrays[1:], start=1):
            if arr.shape != (h, w):
                raise ValueError(f"DICOM slice {i} shape {arr.shape} != {(h, w)}")

        volume = arrays[0][np.newaxis, ...] if len(arrays) == 1 else np.stack(arrays, axis=0)
        ref = dicom_slices[0]
        return SliceVolume(
            data=volume,
            pixel_spacing_mm=(ref.row_spacing, ref.col_spacing),
            slice_thickness_mm=ref.thickness,
            source_paths=list(paths),
        )

    arrays = [_load_png_array(p) for p in paths]
    h, w = arrays[0].shape
    for i, arr in enumerate(arrays[1:], start=1):
        if arr.shape != (h, w):
            raise ValueError(f"Slice {i} shape {arr.shape} does not match first slice {(h, w)}")

    volume = arrays[0][np.newaxis, ...] if len(arrays) == 1 else np.stack(arrays, axis=0)
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
