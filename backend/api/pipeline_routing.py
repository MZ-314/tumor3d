"""Choose medical volume vs AI image-to-3D pipeline."""

from __future__ import annotations

from pathlib import Path

PIPELINE_AI_3D = "ai_3d"
PIPELINE_MEDICAL_VOLUME = "medical_volume"
PIPELINE_MEDICAL_TUMOR = "medical_tumor"

VOLUME_MODALITIES = frozenset({"knee_mri", "volume_mri", "other_mri"})


def is_dicom_path(path: Path) -> bool:
    return path.suffix.lower() in {".dcm", ".dicom"}


def resolve_pipeline(modality: str, slice_paths: list[Path]) -> str:
    modality = modality.lower()

    if modality == "ai_3d":
        return PIPELINE_AI_3D

    if modality == "brain_mri":
        return PIPELINE_MEDICAL_TUMOR

    if modality in VOLUME_MODALITIES:
        return PIPELINE_MEDICAL_VOLUME

    # Auto-detect from files
    if slice_paths and all(is_dicom_path(p) for p in slice_paths):
        return PIPELINE_MEDICAL_TUMOR if modality == "brain_mri" else PIPELINE_MEDICAL_VOLUME

    if len(slice_paths) == 1 and not is_dicom_path(slice_paths[0]):
        return PIPELINE_AI_3D

    return PIPELINE_MEDICAL_VOLUME
