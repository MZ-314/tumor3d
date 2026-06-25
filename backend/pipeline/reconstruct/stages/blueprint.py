"""Phase 5 — patient-specific reconstruction blueprint."""

from __future__ import annotations

from config_pipeline import SYNTHESIS_BACKEND
from pipeline.reconstruct.context import PipelineState
from shared.schemas.pydantic.pipeline import ReconstructionBlueprint


async def run_blueprint(state: PipelineState) -> None:
    if state.scan_context is None or state.slice_volume is None:
        raise RuntimeError("scan_context required")

    z, h, w = state.slice_volume.data.shape
    if z == 1:
        strategy = (
            "ml_volume_generator" if SYNTHESIS_BACKEND == "ml" else "single_slice_atlas_anchored"
        )
    elif z < 10:
        strategy = "partial_volume_interpolation"
    else:
        strategy = "measured_stack"

    target_z_hint = max(z, 32) if z == 1 else z

    state.blueprint = ReconstructionBlueprint(
        volume_shape_zyx=[target_z_hint, h, w],
        anchor_slice_indices=state.scan_context.anchor_slice_indices,
        synthesis_strategy=strategy,
        locked_lesion_planes=list(state.scan_context.anchor_slice_indices),
        organ_extent_zyx=[target_z_hint, h, w],
    )
    path = state.work_dir / "reconstruction_blueprint.json"
    path.write_text(state.blueprint.model_dump_json(indent=2), encoding="utf-8")
