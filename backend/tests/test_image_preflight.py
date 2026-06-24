"""AI 3D input validation tests."""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BACKEND))

from config_reconstruction import Image3DError  # noqa: E402
from pipeline.image_to_3d.image_preflight import (  # noqa: E402
    detect_slice_montage,
    validate_ai_3d_input,
)


def _write_montage(path: Path, rows: int, cols: int) -> None:
    h, w = 120 * rows + 8 * (rows - 1), 100 * cols + 8 * (cols - 1)
    canvas = np.zeros((h, w), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            y0 = r * 128
            x0 = c * 108
            canvas[y0 : y0 + 120, x0 : x0 + 100] = 80 + (r * cols + c) * 20
    Image.fromarray(canvas, mode="L").convert("RGB").save(path)


def test_detects_slice_montage(tmp_path: Path) -> None:
    path = tmp_path / "montage.png"
    _write_montage(path, 2, 3)
    assert detect_slice_montage(path) == (2, 3)


def test_rejects_montage_for_ai_3d(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IMAGE3D_ALLOW_MONTAGE", raising=False)
    path = tmp_path / "brain_mri_montage.jpg"
    _write_montage(path, 2, 3)
    with pytest.raises(Image3DError, match="montage"):
        validate_ai_3d_input(path)


def test_allows_simple_photo(tmp_path: Path) -> None:
    path = tmp_path / "cat.jpg"
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[40:160, 40:160, 0] = 220
    img[40:160, 40:160, 1] = 120
    Image.fromarray(img).save(path)
    validate_ai_3d_input(path)
