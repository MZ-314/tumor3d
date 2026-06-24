"""Reject inputs TripoSR cannot meaningfully reconstruct."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

from config_reconstruction import Image3DError

_MEDICAL_NAME_HINTS = ("mri", "brain", "ct_", "_ct", "scan", "dicom", "slice", "nifti", "xray", "x-ray")


def _separator_centers(profile: np.ndarray, *, size: int) -> list[int]:
    """Dark gutter lines between montage panels (not image borders)."""
    if profile.size < 8:
        return []

    lo = float(np.percentile(profile, 8))
    dark_thresh = min(0.14, lo + 0.04)
    margin = max(4, int(size * 0.035))
    max_run = max(3, int(size * 0.06))

    centers: list[int] = []
    i = 0
    while i < profile.size:
        if profile[i] < dark_thresh:
            j = i
            while j < profile.size and profile[j] < dark_thresh:
                j += 1
            run_len = j - i
            center = (i + j) // 2
            if 2 <= run_len <= max_run and margin < center < size - margin:
                centers.append(center)
            i = j
        else:
            i += 1
    return centers


def detect_slice_montage(image_path: Path) -> tuple[int, int] | None:
    """
    Return (rows, cols) if the image looks like several scan slices on one canvas.

    TripoSR will extrude the whole collage as one lump — not individual anatomy.
    """
    with Image.open(image_path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0

    gray = rgb.mean(axis=2)
    h, w = gray.shape
    row_prof = gray.mean(axis=1)
    col_prof = gray.mean(axis=0)

    h_seps = _separator_centers(row_prof, size=h)
    v_seps = _separator_centers(col_prof, size=w)
    rows = len(h_seps) + 1
    cols = len(v_seps) + 1
    panels = rows * cols

    if panels >= 3 and (len(h_seps) > 0 or len(v_seps) >= 2):
        return rows, cols
    if cols >= 4 and len(v_seps) >= 3:
        return rows, cols
    return None


def _mostly_grayscale(rgb: np.ndarray) -> bool:
    spread = np.mean(
        np.abs(rgb[..., 0] - rgb[..., 1])
        + np.abs(rgb[..., 1] - rgb[..., 2])
        + np.abs(rgb[..., 0] - rgb[..., 2])
    )
    return spread < 0.09


def _medical_filename(name: str) -> bool:
    lower = name.lower()
    return any(h in lower for h in _MEDICAL_NAME_HINTS)


def validate_ai_3d_input(image_path: Path) -> None:
    """Raise Image3DError when the upload is wrong for single-image 3D inference."""
    if os.environ.get("IMAGE3D_ALLOW_MONTAGE", "").lower() in ("1", "true", "yes"):
        return

    grid = detect_slice_montage(image_path)
    if grid is not None:
        rows, cols = grid
        raise Image3DError(
            f"This image looks like a {rows}×{cols} slice montage ({rows * cols} panels on one picture). "
            "TripoSR does not read MRI/CT slices — it extrudes the entire image as one 3D block.\n\n"
            "For real scan data: switch to **DICOM volume** and upload your `.dcm` files (📁 folder). "
            "For AI 3D: upload one everyday photo with a single subject (not a scan sheet)."
        )

    with Image.open(image_path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0

    if _mostly_grayscale(rgb) and _medical_filename(image_path.name):
        raise Image3DError(
            "This looks like a medical scan image, not a regular photo. "
            "TripoSR guesses 3D shape from photos — it does not reconstruct anatomy from MRI/CT.\n\n"
            "Upload your DICOM slice series in **DICOM volume** mode. "
            "Use **AI 3D** only for a single non-medical photo (product, face, object)."
        )
