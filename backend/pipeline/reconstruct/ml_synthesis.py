"""ML-backed single-slice volume synthesis (Phase 6b)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from config_medical import MedicalPipelineError
from config_pipeline import ML_VOLUME_MODEL_DIR, SYNTHESIS_BACKEND
from pipeline.ingest.images import SliceVolume
from pipeline.ml.pose import estimate_pose
from pipeline.ml.volume_generator import generate_ml_volume
from shared.schemas.pydantic.pipeline import MriView, PoseEstimate as PoseEstimateSchema


def _normalize(vol: np.ndarray) -> np.ndarray:
    v = vol.astype(np.float32)
    v = v - v.min()
    return v / (v.max() or 1.0)


def _dicom_source_path(volume: SliceVolume) -> Path | None:
    for p in volume.source_paths:
        if p.suffix.lower() in {".dcm", ".dicom"}:
            return p
    return volume.source_paths[0] if volume.source_paths else None


def synthesize_ml_volume(
    volume: SliceVolume,
    *,
    target_z: int,
    organ_mask_2d: np.ndarray | None,
    work_dir: Path,
) -> tuple[np.ndarray, str, PoseEstimateSchema]:
    """Generate 3D volume from one slice using learned parallel-slice synthesis."""
    if SYNTHESIS_BACKEND != "ml":
        raise MedicalPipelineError(f"SYNTHESIS_BACKEND={SYNTHESIS_BACKEND!r} is not 'ml'")

    ckpt = ML_VOLUME_MODEL_DIR / "volume_generator.pt"
    if not ckpt.is_file():
        raise MedicalPipelineError(
            f"ML volume generator checkpoint missing at {ckpt}.\n"
            "On RunPod run:\n"
            "  python backend/scripts/setup_ml_brain_recon.py\n"
            "Then restart uvicorn."
        )

    anchor_slice = _normalize(volume.data[0])
    dicom_path = _dicom_source_path(volume)

    pose = estimate_pose(
        anchor_slice,
        organ_mask=organ_mask_2d,
        dicom_path=dicom_path,
        modality="MR",
    )

    synth, strategy, pose_out = generate_ml_volume(
        anchor_slice,
        target_z=target_z,
        organ_mask_2d=organ_mask_2d,
        dicom_path=dicom_path,
        checkpoint=ckpt,
        pose=pose,
    )

    pose_path = work_dir / "pose_estimate.json"
    pose_schema = PoseEstimateSchema(
        organ_type=pose_out.organ_type,
        through_plane_axis=pose_out.through_plane_axis,
        slice_index_normalized=pose_out.slice_index_normalized,
        mri_view=pose_out.mri_view,
        confidence=pose_out.confidence,
        source=pose_out.source,
    )
    pose_path.write_text(pose_schema.model_dump_json(indent=2), encoding="utf-8")

    anchor_z = target_z // 2
    synth[anchor_z] = anchor_slice
    return synth, strategy, pose_schema
