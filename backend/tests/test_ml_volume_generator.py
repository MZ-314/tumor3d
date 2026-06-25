"""Tests for ML single-slice volume synthesis (Phase 6b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BACKEND))

from pipeline.ml.pose import estimate_pose, route_organ  # noqa: E402
from pipeline.reconstruct.synthesis_volume import synthesize_volume  # noqa: E402
from pipeline.ingest.images import SliceVolume  # noqa: E402
from shared.schemas.pydantic.pipeline import OrganType, ReconstructionBlueprint  # noqa: E402


def _synthetic_brain_slice(size: int = 64) -> np.ndarray:
    yy, xx = np.ogrid[:size, :size]
    cy, cx = size // 2, size // 2
    brain = ((yy - cy) ** 2 + (xx - cx) ** 2) < (size * 0.32) ** 2
    img = np.zeros((size, size), dtype=np.float32)
    img[brain] = 0.35 + 0.4 * np.random.rand(brain.sum()).astype(np.float32)
    return img


def test_pose_estimation_heuristic() -> None:
    slice_img = _synthetic_brain_slice()
    pose = estimate_pose(slice_img)
    assert pose.organ_type in (OrganType.BRAIN, OrganType.UNKNOWN)
    assert 0.0 <= pose.slice_index_normalized <= 1.0
    assert pose.through_plane_axis in (0, 1, 2)


def test_organ_router_brain_heuristic() -> None:
    slice_img = _synthetic_brain_slice()
    assert route_organ(slice_img) == OrganType.BRAIN


try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_TORCH, reason="torch required")
def test_ml_volume_generator_anchor_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import nibabel as nib
    import torch

    from pipeline.ml.training.train_volume_generator import train

    size = 32
    vol = np.random.rand(size, size, size).astype(np.float32)
    nii_path = tmp_path / "train_vol.nii.gz"
    nib.save(nib.Nifti1Image(vol, np.eye(4)), str(nii_path))

    ckpt_dir = tmp_path / "models"
    ckpt_dir.mkdir()
    ckpt = ckpt_dir / "volume_generator.pt"
    train([nii_path], output_path=ckpt, epochs=1, batch_size=4, device="cpu")

    monkeypatch.setenv("SYNTHESIS_BACKEND", "ml")
    monkeypatch.setenv("ML_VOLUME_MODEL_DIR", str(ckpt_dir))

    import config_pipeline

    monkeypatch.setattr(config_pipeline, "SYNTHESIS_BACKEND", "ml", raising=False)
    monkeypatch.setattr(config_pipeline, "ML_VOLUME_MODEL_DIR", ckpt_dir, raising=False)

    anchor = _synthetic_brain_slice(64)
    volume = SliceVolume(
        data=anchor[None, ...],
        pixel_spacing_mm=(1.0, 1.0),
        slice_thickness_mm=3.0,
        source_paths=[],
    )
    blueprint = ReconstructionBlueprint(
        volume_shape_zyx=[32, 64, 64],
        anchor_slice_indices=[0],
        synthesis_strategy="ml_volume_generator",
    )

    result, out = synthesize_volume(
        volume,
        lesions=[],
        work_dir=tmp_path,
        reconstruction_id="test_ml",
        anchor_indices=[0],
        organ_mask_2d=anchor > 0.2,
        blueprint=blueprint,
    )

    assert "ml" in result.strategy
    anchor_z = out.data.shape[0] // 2
    anchor_norm = anchor / (anchor.max() or 1.0)
    assert np.allclose(out.data[anchor_z], anchor_norm, atol=0.05)
    assert (tmp_path / "pose_estimate.json").exists()
    assert (tmp_path / "test_ml_volume.nii.gz").exists()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
