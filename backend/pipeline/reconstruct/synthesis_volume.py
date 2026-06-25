"""Synthetic slice generation — preserve real slices, fill missing depth (Phase 6)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

from pipeline.export.nifti_export import save_mask_nifti, save_volume_nifti
from pipeline.ingest.images import SliceVolume
from pipeline.segment.backends import LesionMask
from shared.schemas.pydantic.pipeline import SynthesisResult


def _target_depth(slice_count: int, slice_thickness_mm: float) -> int:
    """Minimum z depth for 3D mesh/volume from sparse stacks."""
    if slice_count >= 10:
        return slice_count
    if slice_count >= 6:
        return max(slice_count, 24)
    # Single-slice USP: expand along z using slice thickness prior (~120mm brain extent).
    extent_mm = 120.0
    estimated = int(round(extent_mm / max(slice_thickness_mm, 0.1)))
    return int(np.clip(estimated, 32, 96))


def synthesize_volume(
    volume: SliceVolume,
    *,
    lesions: list[LesionMask],
    work_dir: Path,
    reconstruction_id: str,
    anchor_indices: list[int],
    organ_mask_2d: np.ndarray | None = None,
) -> tuple[SynthesisResult, SliceVolume]:
    """
    Build a 3D intensity volume from uploaded slices.

    - Multi-slice: preserves every uploaded slice at its index (measured).
    - Single/partial: linearly interpolates along z to target depth; anchor planes unchanged.
    """
    data = volume.data
    z_in, h, w = data.shape
    target_z = _target_depth(z_in, volume.slice_thickness_mm)

    if z_in == target_z:
        synth = data.copy()
        synthetic_count = 0
        strategy = "measured_stack"
    elif z_in == 1:
        from pipeline.reconstruct.atlas_synthesis import synthesize_atlas_anchored_volume

        synth, strategy = synthesize_atlas_anchored_volume(
            volume,
            target_z=target_z,
            organ_mask_2d=organ_mask_2d,
        )
        synthetic_count = target_z - 1
    else:
        zoom_factor = target_z / z_in
        synth = zoom(data, (zoom_factor, 1.0, 1.0), order=1)
        synthetic_count = max(0, synth.shape[0] - z_in)
        strategy = "linear_z_interpolation"
        # Re-anchor original slices
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

    combined_lesion = None
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

    # Confidence: 1.0 on anchor z-planes, lower elsewhere
    conf = np.full(synth.shape, 0.45, dtype=np.float32)
    if z_in == 1:
        conf[synth.shape[0] // 2] = 1.0
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
    )
    return result, out_volume
