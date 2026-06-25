"""View-aware atlas orientation tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BACKEND))

from pipeline.reconstruct.view_orient import (  # noqa: E402
    detect_mri_view_from_pixels,
    orient_atlas_volume,
)
from shared.schemas.pydantic.pipeline import MriView  # noqa: E402


def test_orient_atlas_sagittal_changes_stack_axis() -> None:
    atlas = np.zeros((20, 40, 50), dtype=np.float32)
    out = orient_atlas_volume(atlas, MriView.SAGITTAL)
    assert out.shape == (50, 40, 20)


def test_detect_sagittal_from_tall_mask() -> None:
    img = np.zeros((128, 128), dtype=np.float32)
    mask = np.zeros((128, 128), dtype=bool)
    mask[20:110, 50:78] = True
    img[mask] = 1.0
    assert detect_mri_view_from_pixels(img, mask) == MriView.SAGITTAL


def test_detect_axial_from_round_mask() -> None:
    img = np.zeros((128, 128), dtype=np.float32)
    mask = np.zeros((128, 128), dtype=bool)
    mask[40:88, 40:88] = True
    img[mask] = 1.0
    assert detect_mri_view_from_pixels(img, mask) == MriView.AXIAL
