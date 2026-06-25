"""Atlas-anchored single-slice synthesis tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BACKEND))

from pipeline.ingest.images import SliceVolume  # noqa: E402
from pipeline.reconstruct.atlas_synthesis import synthesize_atlas_anchored_volume  # noqa: E402


def test_atlas_synthesis_anchor_matches_patient(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    atlas = np.linspace(0, 1, 16 * 32 * 32, dtype=np.float32).reshape(16, 32, 32)
    atlas_path = tmp_path / "template.nii.gz"
    import nibabel as nib

    nib.save(nib.Nifti1Image(atlas, np.eye(4)), str(atlas_path))
    monkeypatch.setattr(
        "pipeline.reconstruct.atlas_synthesis.ATLAS_BRAIN_TEMPLATE",
        atlas_path,
    )

    patient = np.zeros((32, 32), dtype=np.float32)
    patient[10:22, 10:22] = 0.9
    volume = SliceVolume(
        data=patient[np.newaxis, ...],
        pixel_spacing_mm=(1.0, 1.0),
        slice_thickness_mm=3.0,
        source_paths=[],
    )
    mask = np.zeros((32, 32), dtype=bool)
    mask[10:22, 10:22] = True

    synth, strategy = synthesize_atlas_anchored_volume(
        volume,
        target_z=48,
        organ_mask_2d=mask,
    )

    assert strategy.startswith("single_slice_atlas_anchored")
    assert synth.shape == (48, 32, 32)
    anchor = 48 // 2
    assert float(np.mean(np.abs(synth[anchor] - patient))) < 0.05
