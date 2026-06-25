"""Phase 1 — DICOM ingest and ScanContext extraction."""

from __future__ import annotations

from config_pipeline import MULTI_SLICE_MIN_SLICES, PARTIAL_VOLUME_MIN_SLICES
from pipeline.ingest.dicom_meta import (
    detect_input_source,
    detect_mri_view_from_dicom,
    organ_type_from_dicom,
    slice_spacing_from_dicom,
)
from pipeline.ingest.images import load_slice_volume
from pipeline.reconstruct.context import PipelineState
from shared.schemas.pydantic.common import AccuracyTier
from shared.schemas.pydantic.pipeline import InputSource, MriView, OrganType, ScanContext


def _organ_from_modality(modality: str) -> OrganType:
    m = modality.lower()
    if "brain" in m:
        return OrganType.BRAIN
    if "knee" in m:
        return OrganType.KNEE
    return OrganType.OTHER


def _accuracy_tier(slice_count: int) -> AccuracyTier:
    if slice_count == 1:
        return AccuracyTier.SINGLE_SLICE
    if slice_count < PARTIAL_VOLUME_MIN_SLICES:
        return AccuracyTier.PARTIAL_VOLUME
    if slice_count < MULTI_SLICE_MIN_SLICES:
        return AccuracyTier.PARTIAL_VOLUME
    return AccuracyTier.MULTI_SLICE


async def run_input_intelligence(state: PipelineState) -> None:
    volume = load_slice_volume(state.slice_paths)
    state.slice_volume = volume

    slice_count = volume.data.shape[0]
    tier = _accuracy_tier(slice_count)
    anchor_indices = list(range(slice_count))

    input_src = detect_input_source(state.slice_paths)
    mri_view = MriView.UNKNOWN
    organ = _organ_from_modality(state.modality)
    spacing = None
    body_part = None
    series_desc = None
    warnings: list[str] = []

    if input_src == "dicom":
        organ = organ_type_from_dicom(state.slice_paths[0], state.modality)
        mri_view = detect_mri_view_from_dicom(state.slice_paths[0])
        spacing = slice_spacing_from_dicom(state.slice_paths[0])
        import pydicom

        ds = pydicom.dcmread(str(state.slice_paths[0]), stop_before_pixels=True)
        body_part = str(getattr(ds, "BodyPartExamined", "") or "") or None
        series_desc = str(getattr(ds, "SeriesDescription", "") or "") or None
    else:
        warnings.append(
            "Upload is not DICOM — spacing and orientation use defaults. "
            "Doctors should upload DICOM (.dcm) series for measured 3D."
        )

    if slice_count == 1:
        warnings.append(
            "Single-slice upload: unseen depth will be synthesized from atlas and anatomical priors."
        )

    z, h, w = volume.data.shape
    state.scan_context = ScanContext(
        reconstruction_id=state.reconstruction_id,
        input_source=InputSource(input_src),
        organ_type=organ,
        modality=state.modality,
        mri_view=mri_view,
        accuracy_tier=tier,
        slice_count=slice_count,
        anchor_slice_indices=anchor_indices,
        slice_spacing_mm=spacing,
        volume_shape_zyx=[z, h, w],
        body_part_examined=body_part,
        series_description=series_desc,
        quality_score=1.0,
        warnings=warnings,
    )

    artifact_path = state.work_dir / "scan_context.json"
    artifact_path.write_text(state.scan_context.model_dump_json(indent=2), encoding="utf-8")
