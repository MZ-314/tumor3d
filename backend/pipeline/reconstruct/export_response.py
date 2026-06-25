"""Build ReconstructResponse from pipeline state (Phase 7)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from config_medical import SEGMENTATION_BACKEND
from config_pipeline import LEGACY_VOLUME_SYNTHESIS, MODULAR_RECON
from pipeline.export.nifti_export import combined_lesion_mask, save_mask_nifti
from pipeline.ingest.images import save_png_overlay
from pipeline.mesh.lesion_mesh import build_lesion_geometries, get_scene_path
from pipeline.mesh.organ_mesh import build_organ_mesh_scene
from pipeline.mesh.slice_preview import build_slice_preview_scene
from pipeline.reconstruct.context import PipelineState
from pipeline.segment.backends import VOLUME_ONLY_MODALITIES
from services.groq_assistant import build_assistant_summary
from shared.schemas.pydantic.common import AccuracyTier
from shared.schemas.pydantic.reconstruct import LesionResult, ReconstructResponse


def _accuracy_tier(slice_count: int) -> AccuracyTier:
    if slice_count <= 1:
        return AccuracyTier.SINGLE_SLICE
    if slice_count < 10:
        return AccuracyTier.PARTIAL_VOLUME
    return AccuracyTier.MULTI_SLICE


async def build_reconstruct_response(state: PipelineState) -> ReconstructResponse:
    if state.slice_volume is None:
        raise RuntimeError("slice_volume required")
    if state.segmentation is None:
        raise RuntimeError("segmentation required")

    volume = state.output_volume or state.slice_volume
    seg = state.segmentation
    reconstruction_id = state.reconstruction_id
    work_dir = state.work_dir
    modality = state.modality

    modular = (
        MODULAR_RECON
        and not LEGACY_VOLUME_SYNTHESIS
        and modality.lower() == "brain_mri"
        and state.module_assembly is not None
    )

    if not modular:
        if seg.lesions:
            geometries = build_lesion_geometries(volume, seg.lesions, work_dir, reconstruction_id)
        else:
            geometries = []
            organ_mask = state.organ_mask_2d
            if state.scan_context and state.scan_context.slice_count <= 1:
                build_organ_mesh_scene(
                    volume,
                    work_dir,
                    reconstruction_id,
                    organ_mask_2d=organ_mask,
                )
            else:
                build_slice_preview_scene(volume, work_dir, reconstruction_id)
    else:
        geometries = []
        if seg.lesions:
            geometries = build_lesion_geometries(volume, seg.lesions, work_dir, reconstruction_id)

    z_mid = volume.data.shape[0] // 2
    overlay_path = work_dir / "overlay.png"
    combined_mask = seg.lesions[0].mask[z_mid] if seg.lesions else None
    for lesion in seg.lesions[1:]:
        combined_mask = combined_mask | lesion.mask[z_mid]
    if combined_mask is not None:
        save_png_overlay(volume.data[z_mid], combined_mask.astype(float), overlay_path)

    source_path = work_dir / "source.png"
    if not source_path.exists():
        arr = (np.clip(volume.data[z_mid], 0, 1) * 255).astype(np.uint8)
        Image.fromarray(arr).save(source_path)

    base = "/static"
    volume_only = modality.lower() in VOLUME_ONLY_MODALITIES
    no_lesion = len(seg.lesions) == 0
    z_count = state.scan_context.slice_count if state.scan_context else volume.data.shape[0]
    tier = _accuracy_tier(z_count)
    single_slice_ai = z_count <= 1 and not volume_only
    effective_backend = "volume_only" if volume_only else SEGMENTATION_BACKEND

    volume_nii_path = work_dir / f"{reconstruction_id}_volume.nii.gz"
    volume_nifti_url = (
        f"{base}/{reconstruction_id}/{volume_nii_path.name}" if volume_nii_path.exists() else None
    )

    tumor_mask_nifti_url = None
    combined_mask_vol = combined_lesion_mask(seg.lesions)
    if combined_mask_vol is not None:
        mask_nii_path = work_dir / f"{reconstruction_id}_tumor.nii.gz"
        save_mask_nifti(combined_mask_vol, volume, mask_nii_path)
        tumor_mask_nifti_url = f"{base}/{reconstruction_id}/{mask_nii_path.name}"

    scene_path = get_scene_path(work_dir, reconstruction_id)
    modules_out: list = []
    module_manifest_url = None
    explorer_mode = "legacy"
    viewer_mode = "volume" if volume_nifti_url else "mesh"
    geometry_source = "mixed" if single_slice_ai else "measured"

    if modular and state.module_assembly is not None:
        assembly = state.module_assembly
        root = Path(assembly.root_glb_path)
        if root.is_file():
            import shutil

            shutil.copy2(root, scene_path)
        modules_out = assembly.modules
        if assembly.module_manifest_path and Path(assembly.module_manifest_path).is_file():
            manifest_name = Path(assembly.module_manifest_path).name
            module_manifest_url = f"{base}/{reconstruction_id}/assembly/{manifest_name}"
        explorer_mode = "modular"
        viewer_mode = "modular"
        geometry_source = "modular_atlas"
        disclaimer = (
            "Modular 3D brain reconstruction: your DICOM slice and tumor footprint are measured "
            "on the anchor plane; lobe/subcortical modules are registered atlas geometry morphed "
            "near the lesion. Niivue volume (if shown) is AI-completed depth. Not for diagnosis."
        )
    elif volume_only:
        disclaimer = (
            "Volume-only mode: building a 3D stack from your DICOM slices. "
            "No tumor/lesion AI — MONAI BraTS is trained on brain MRI only. "
            "Not for diagnosis."
        )
    elif no_lesion:
        disclaimer = (
            "AI-predicted 3D brain from your single slice (atlas + MedSAM). "
            "The uploaded plane is anchored; other anatomy is an estimate — not measured. "
            "Not for diagnosis."
        )
    else:
        disclaimer = (
            "AI-predicted 3D brain from sparse MRI: your slice is anchored; tumor depth beyond "
            "the upload is estimated by MONAI BraTS. Not for diagnosis."
        )

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

    response = ReconstructResponse(
        reconstruction_id=reconstruction_id,
        chat_id=state.chat_id,
        source_image_url=f"{base}/{reconstruction_id}/source.png",
        overlay_image_url=f"{base}/{reconstruction_id}/overlay.png"
        if overlay_path.exists()
        else None,
        scene_mesh_url=f"{base}/{reconstruction_id}/{scene_path.name}",
        volume_nifti_url=volume_nifti_url,
        tumor_mask_nifti_url=tumor_mask_nifti_url,
        module_manifest_url=module_manifest_url,
        modules=modules_out,
        explorer_mode=explorer_mode,
        viewer_mode=viewer_mode,
        slice_count=z_count,
        accuracy_tier=tier,
        modality=modality,
        pipeline_type="medical",
        geometry_source=geometry_source,
        segmentation_backend=effective_backend,
        lesions=lesion_results,
        assistant_summary="",
        disclaimer=disclaimer,
    )
    response.assistant_summary = await build_assistant_summary(
        response, user_text=state.user_text
    )
    return response
