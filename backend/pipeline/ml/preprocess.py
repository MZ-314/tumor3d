"""Image preprocessing helpers for ML reconstruction."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, zoom


def fit_organ_mask(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    m = mask.astype(bool)
    if m.shape != (h, w):
        m = zoom(m.astype(np.float32), (h / m.shape[0], w / m.shape[1]), order=0) > 0.5
    m = binary_fill_holes(m)
    m = binary_closing(m, iterations=2)
    return m
