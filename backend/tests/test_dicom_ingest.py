"""DICOM pixel array normalization for load_slice_volume."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pipeline.ingest.images import _dicom_to_grayscale_2d


def test_dicom_rgb_hwc() -> None:
    arr = np.zeros((64, 48, 3), dtype=np.float32)
    arr[..., 0] = 1.0
    ds = SimpleNamespace(NumberOfFrames=0)
    gray = _dicom_to_grayscale_2d(arr, ds)
    assert gray.shape == (64, 48)


def test_dicom_multiframe() -> None:
    arr = np.random.rand(5, 32, 32).astype(np.float32)
    ds = SimpleNamespace(NumberOfFrames=5)
    gray = _dicom_to_grayscale_2d(arr, ds)
    assert gray.shape == (32, 32)


def test_dicom_planar_rgb() -> None:
    arr = np.zeros((3, 40, 50), dtype=np.float32)
    arr[1, :, :] = 1.0
    ds = SimpleNamespace(NumberOfFrames=0)
    gray = _dicom_to_grayscale_2d(arr, ds)
    assert gray.shape == (40, 50)
