"""Volumetric reconstruction — export NIfTI for Niivue."""

from __future__ import annotations

from pathlib import Path

from pipeline.export.nifti_export import save_volume_nifti
from pipeline.modular.types import ModularContext


def run_volumetric(ctx: ModularContext) -> Path | None:
    vol = ctx.output_volume or ctx.slice_volume
    out_path = ctx.work_dir / f"{ctx.reconstruction_id}_volume.nii.gz"
    save_volume_nifti(vol, out_path)
    if ctx.synthesis is None:
        return out_path
    ctx.synthesis = ctx.synthesis.model_copy(update={"intensity_volume_path": str(out_path)})
    return out_path
