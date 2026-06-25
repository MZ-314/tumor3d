"""Synthetic slice generation — preserve real slices, fill missing depth (Phase 6)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

from config_pipeline import SYNTHESIS_BACKEND
from pipeline.export.nifti_export import save_mask_nifti, save_volume_nifti
from pipeline.ingest.images import SliceVolume
from pipeline.segment.backends import LesionMask
from shared.schemas.pydantic.pipeline import (
    AtlasWarpResult,
    MriView,
    ReconstructionBlueprint,
    SynthesisResult,
)


def _target_depth(
    slice_count: int,
    slice_thickness_mm: float,
    pixel_spacing_mm: tuple[float, float],
    h: int,
    w: int,
) -> int:
    """Z depth sized for ~isotropic brain extent (avoids thin rectangular slab)."""
    if slice_count >= 10:
        return slice_count
    if slice_count >= 6:
        return max(slice_count, 24)
    row_sp, col_sp = pixel_spacing_mm
    in_plane_mm = max(h * row_sp, w * col_sp)
    # Adult brain ~140–160 mm; match in-plane FOV so 3D looks volumetric not pancake
    brain_extent_mm = float(np.clip(in_plane_mm * 0.75, 120.0, 170.0))
    estimated = int(round(brain_extent_mm / max(slice_thickness_mm, 0.5)))
    return int(np.clip(estimated, 48, 128))


def synthesize_volume(
    volume: SliceVolume,
    *,
    lesions: list[LesionMask],
    work_dir: Path,
    reconstruction_id: str,
    anchor_indices: list[int],
    organ_mask_2d: np.ndarray | None = None,
    mri_view: object | None = None,
    atlas_warp: object | None = None,
    blueprint: ReconstructionBlueprint | None = None,
) -> tuple[SynthesisResult, SliceVolume]:
    """
    Build a 3D intensity volume from uploaded slices.

    Single-slice: ML conditional slice generator (Phase 6b) by default.
    Multi-slice: preserves every uploaded slice at its index (measured).
    """
    data = volume.data
    z_in, h, w = data.shape
    target_z = _target_depth(
        z_in,
        volume.slice_thickness_mm,
        volume.pixel_spacing_mm,
        h,
        w,
    )

    strategy = "measured_stack"
    model_version: str | None = None
    pose_estimate_path: str | None = None
    use_ml = False

    if z_in == target_z:
        synth = data.copy()
        synthetic_count = 0
        strategy = "measured_stack"
    elif z_in == 1:
        use_ml = SYNTHESIS_BACKEND == "ml" and (
            blueprint is None or blueprint.synthesis_strategy == "ml_volume_generator"
        )
        if use_ml:
            from pipeline.reconstruct.ml_synthesis import synthesize_ml_volume

            synth, strategy, pose = synthesize_ml_volume(
                volume,
                target_z=target_z,
                organ_mask_2d=organ_mask_2d,
                work_dir=work_dir,
            )
            model_version = "ml_slice_generator_v1" if "ml_slice" in strategy else "ml"
            pose_estimate_path = "pose_estimate.json"
            if mri_view is not None and isinstance(mri_view, MriView) and pose.mri_view != MriView.UNKNOWN:
                pass  # caller may update scan_context from pose JSON
        else:
            from pipeline.reconstruct.atlas_synthesis import synthesize_atlas_anchored_volume

            view = mri_view if isinstance(mri_view, MriView) else MriView.UNKNOWN
            warp = atlas_warp if isinstance(atlas_warp, AtlasWarpResult) else None
            synth, strategy = synthesize_atlas_anchored_volume(
                volume,
                target_z=target_z,
                organ_mask_2d=organ_mask_2d,
                mri_view=view,
                atlas_warp=warp,
                work_dir=work_dir,
            )
        synthetic_count = target_z - 1
    else:
        zoom_factor = target_z / z_in
        synth = zoom(data, (zoom_factor, 1.0, 1.0), order=1)
        synthetic_count = max(0, synth.shape[0] - z_in)
        strategy = "linear_z_interpolation"
        for idx in anchor_indices:
            if 0 <= idx < z_in:
                out_idx = int(round(idx * zoom_factor))
                out_idx = min(out_idx, synth.shape[0] - 1)
                synth[out_idx] = data[idx]

    out_volume = SliceVolume(
        data=synth.astype(np.float32),
        pixel_spacing_mm=volume.pixel_spacing_mm,
        slice_thickness_mm=volume.slice_thickness_mm,
        source_paths=volume.source_paths,
    )

    intensity_path = work_dir / f"{reconstruction_id}_volume.nii.gz"
    save_volume_nifti(out_volume, intensity_path)

    label_path = work_dir / f"{reconstruction_id}_labels.nii.gz"
    confidence_path = work_dir / f"{reconstruction_id}_confidence.nii.gz"

    if lesions:
        from pipeline.export.nifti_export import combined_lesion_mask

        combined_lesion = combined_lesion_mask(lesions)
        if combined_lesion is not None and combined_lesion.shape[0] != synth.shape[0]:
            lz = synth.shape[0] / combined_lesion.shape[0]
            combined_lesion = zoom(
                combined_lesion.astype(np.float32),
                (lz, 1.0, 1.0),
                order=0,
            ) > 0.5
        if combined_lesion is not None:
            save_mask_nifti(combined_lesion, out_volume, label_path)

    conf = np.full(synth.shape, 0.35, dtype=np.float32)
    if z_in == 1:
        conf[synth.shape[0] // 2] = 1.0
        if use_ml:
            # Graduated confidence away from anchor for ML synthesis
            anchor_z = synth.shape[0] // 2
            for z in range(synth.shape[0]):
                if z == anchor_z:
                    continue
                dz = abs(z - anchor_z) / max(anchor_z, 1)
                conf[z] = float(max(0.2, 0.75 - 0.45 * dz))
    else:
        zoom_factor = synth.shape[0] / z_in
        for idx in anchor_indices:
            out_idx = int(round(idx * zoom_factor)) if z_in > 1 else idx
            out_idx = min(max(out_idx, 0), synth.shape[0] - 1)
            conf[out_idx] = 1.0
    save_volume_nifti(
        SliceVolume(
            data=conf,
            pixel_spacing_mm=volume.pixel_spacing_mm,
            slice_thickness_mm=volume.slice_thickness_mm,
            source_paths=volume.source_paths,
        ),
        confidence_path,
    )

    result = SynthesisResult(
        intensity_volume_path=str(intensity_path.relative_to(work_dir)),
        label_volume_path=str(label_path.relative_to(work_dir)) if label_path.exists() else None,
        confidence_volume_path=str(confidence_path.relative_to(work_dir)),
        real_slices_preserved=z_in,
        synthetic_slices_generated=synthetic_count,
        strategy=strategy,
        model_version=model_version,
        pose_estimate_path=pose_estimate_path,
    )
    return result, out_volume
