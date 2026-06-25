"""Phase 6 — synthetic slice generation."""

from __future__ import annotations

import asyncio

from pipeline.reconstruct.context import PipelineState
from pipeline.reconstruct.synthesis_volume import synthesize_volume


async def run_synthesis(state: PipelineState) -> None:
    if state.slice_volume is None or state.segmentation is None or state.scan_context is None:
        raise RuntimeError("slice_volume and segmentation required")

    lesions = state.segmentation.lesions
    result, out_volume = await asyncio.to_thread(
        synthesize_volume,
        state.slice_volume,
        lesions=lesions,
        work_dir=state.work_dir,
        reconstruction_id=state.reconstruction_id,
        anchor_indices=state.scan_context.anchor_slice_indices,
        organ_mask_2d=state.organ_mask_2d,
        mri_view=state.scan_context.mri_view,
        atlas_warp=state.atlas_warp,
        blueprint=state.blueprint,
    )
    state.synthesis = result
    state.output_volume = out_volume

    pose_path = state.work_dir / "pose_estimate.json"
    if pose_path.is_file() and state.scan_context is not None:
        from shared.schemas.pydantic.pipeline import PoseEstimate

        pose = PoseEstimate.model_validate_json(pose_path.read_text(encoding="utf-8"))
        state.scan_context = state.scan_context.model_copy(update={"mri_view": pose.mri_view})

    path = state.work_dir / "synthesis_result.json"
    path.write_text(state.synthesis.model_dump_json(indent=2), encoding="utf-8")
