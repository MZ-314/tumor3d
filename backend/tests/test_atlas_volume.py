"""Registered atlas volume synthesis tests."""

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
from pipeline.reconstruct.atlas_volume import (  # noqa: E402
    build_registered_atlas_volume,
    find_best_atlas_slice_index,
)
from shared.schemas.pydantic.pipeline import MriView  # noqa: E402


def test_find_best_atlas_slice_picks_offset() -> None:
    az, h, w = 20, 64, 64
    atlas = np.zeros((az, h, w), dtype=np.float32)
    atlas[14] = 1.0
    patient = np.zeros((h, w), dtype=np.float32)
    patient[:, :] = atlas[14]
    assert find_best_atlas_slice_index(patient, atlas, None) == 14


def test_build_volume_locks_anchor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    atlas = np.random.rand(24, 48, 48).astype(np.float32)
    atlas_path = tmp_path / "template.nii.gz"
    import nibabel as nib

    nib.save(nib.Nifti1Image(atlas, np.eye(4)), str(atlas_path))
    monkeypatch.setattr(
        "pipeline.reconstruct.atlas_volume.ATLAS_BRAIN_TEMPLATE",
        atlas_path,
    )

    patient = np.zeros((48, 48), dtype=np.float32)
    patient[16:32, 16:32] = 0.95
    volume = SliceVolume(
        data=patient[np.newaxis, ...],
        pixel_spacing_mm=(1.0, 1.0),
        slice_thickness_mm=3.0,
        source_paths=[],
    )
    mask = np.zeros((48, 48), dtype=bool)
    mask[16:32, 16:32] = True

    synth, strategy, _ = build_registered_atlas_volume(
        volume,
        target_z=40,
        organ_mask_2d=mask,
        mri_view=MriView.AXIAL,
        atlas_warp=None,
        work_dir=None,
    )
    assert "registered_atlas" in strategy
    assert synth.shape == (40, 48, 48)
    assert float(np.mean(np.abs(synth[20] - patient))) < 0.05
