"""AI volume completion — BraTS-trained 3D refiner with anchor lock."""

from __future__ import annotations

import asyncio

from pipeline.modular.types import ModularContext
from pipeline.reconstruct.synthesis_volume import synthesize_volume
from shared.schemas.pydantic.pipeline import SynthesisResult


async def run_volume_completion(ctx: ModularContext) -> None:
    lesions = []
    if ctx.lesion_mask_2d is not None:
        from pipeline.segment.backends import LesionMask

        z = ctx.anchor_z
        mask_3d = __import__("numpy").zeros_like(ctx.slice_volume.data, dtype=bool)
        mask_3d[z] = ctx.lesion_mask_2d
        lesions = [
            LesionMask(
                lesion_id=0,
                mask=mask_3d,
                in_plane_confidence=0.85,
            )
        ]

    result, out_volume = await asyncio.to_thread(
        synthesize_volume,
        ctx.slice_volume,
        lesions=lesions,
        work_dir=ctx.work_dir,
        reconstruction_id=ctx.reconstruction_id,
        anchor_indices=[ctx.anchor_z],
        organ_mask_2d=ctx.organ_mask_2d,
        mri_view=ctx.scan_context.mri_view,
        atlas_warp=ctx.atlas_warp,
        blueprint=ctx.blueprint,
    )
    result = result.model_copy(
        update={
            "strategy": "modular_volume_completion",
            "model_version": result.model_version or "volume_refiner_3d",
        }
    )
    ctx.synthesis = result
    ctx.output_volume = out_volume
