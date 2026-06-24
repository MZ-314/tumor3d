"""Orchestrate medical slice upload → segmentation → lesion meshes → response."""

from __future__ import annotations

from pathlib import Path

from config_medical import (
    SEGMENTATION_BACKEND,
    MedicalPipelineError,
    ensure_data_dirs,
)
from pipeline.ingest.images import load_slice_volume, save_png_overlay
from pipeline.mesh.lesion_mesh import build_lesion_geometries, get_scene_path
from pipeline.segment.backends import segment_volume
from services.groq_assistant import build_assistant_summary
from shared.schemas.pydantic.common import AccuracyTier
from shared.schemas.pydantic.reconstruct import LesionResult, ReconstructResponse


def _accuracy_tier(slice_count: int) -> AccuracyTier:
    if slice_count <= 1:
        return AccuracyTier.SINGLE_SLICE
    if slice_count < 10:
        return AccuracyTier.PARTIAL_VOLUME
    return AccuracyTier.MULTI_SLICE


async def process_medical_slices(
    slice_paths: list[Path],
    work_dir: Path,
    *,
    modality: str = "brain_mri",
    chat_id: str | None = None,
    user_text: str | None = None,
    backend: str | None = None,
) -> ReconstructResponse:
    ensure_data_dirs()
    backend = (backend or SEGMENTATION_BACKEND).lower()
    reconstruction_id = work_dir.name

    try:
        volume = load_slice_volume(slice_paths)
        seg = segment_volume(volume.data, backend, modality=modality)
        geometries = build_lesion_geometries(
            volume, seg.lesions, work_dir, reconstruction_id
        )
    except MedicalPipelineError:
        raise
    except Exception as exc:
        raise MedicalPipelineError(str(exc)) from exc

    z_mid = volume.data.shape[0] // 2
    overlay_path = work_dir / "overlay.png"
    combined_mask = seg.lesions[0].mask[z_mid] if seg.lesions else None
    for lesion in seg.lesions[1:]:
        combined_mask = combined_mask | lesion.mask[z_mid]
    if combined_mask is not None:
        save_png_overlay(volume.data[z_mid], combined_mask.astype(float), overlay_path)

    source_path = work_dir / "source.png"
    if not source_path.exists():
        from PIL import Image
        import numpy as np

        arr = (np.clip(volume.data[z_mid], 0, 1) * 255).astype(np.uint8)
        Image.fromarray(arr).save(source_path)

    base = "/static"
    scene_path = get_scene_path(work_dir, reconstruction_id)
    lesion_results: list[LesionResult] = []
    for geo in geometries:
        lesion_results.append(
            LesionResult(
                lesion_id=geo.lesion_id,
                mesh_url=f"{base}/{reconstruction_id}/{geo.mesh_path.name}",
                centroid_mm=geo.centroid_mm,
                bounding_box_2d=geo.bounding_box_2d,
                bounding_box_3d_mm=geo.bounding_box_3d_mm,
                volume_mm3=geo.volume_mm3,
                in_plane_confidence=geo.in_plane_confidence,
                depth_confidence=geo.depth_confidence,
                vertices=geo.vertices,
            )
        )

    stub_disclaimer = (
        "STUB DEMO MODE: this is not real tumor detection. "
        "Bright-region heuristics only — unsuitable for clinical MRI. "
        "On RunPod set SEGMENTATION_BACKEND=monai for trained segmentation."
    )
    disclaimer = (
        stub_disclaimer
        if backend == "stub"
        else (
            "Tumor location on the slice is model-inferred. Depth and volume improve with "
            "more slices. Not for diagnosis."
        )
    )

    tier = _accuracy_tier(volume.data.shape[0])
    response = ReconstructResponse(
        reconstruction_id=reconstruction_id,
        chat_id=chat_id,
        source_image_url=f"{base}/{reconstruction_id}/source.png",
        overlay_image_url=f"{base}/{reconstruction_id}/overlay.png"
        if overlay_path.exists()
        else None,
        scene_mesh_url=f"{base}/{reconstruction_id}/{scene_path.name}",
        slice_count=volume.data.shape[0],
        accuracy_tier=tier,
        modality=modality,
        segmentation_backend=backend,
        lesions=lesion_results,
        assistant_summary="",
        disclaimer=disclaimer,
    )

    response.assistant_summary = await build_assistant_summary(
        response, user_text=user_text
    )
    return response
