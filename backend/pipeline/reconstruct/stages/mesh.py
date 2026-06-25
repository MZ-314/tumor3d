"""Phase 7 — 3D volume and mesh generation."""

from __future__ import annotations

from config_pipeline import LEGACY_VOLUME_SYNTHESIS, MODULAR_RECON
from pipeline.reconstruct.context import PipelineState
from pipeline.reconstruct.export_response import build_reconstruct_response


async def run_mesh_generation(state: PipelineState) -> None:
    if MODULAR_RECON and not LEGACY_VOLUME_SYNTHESIS and state.modality.lower() == "brain_mri":
        from pipeline.modular.orchestrator import (
            apply_modular_context_to_state,
            build_modular_context,
            run_modular_assembly_block,
        )

        ctx = build_modular_context(state)
        ctx.blueprint = state.blueprint
        ctx.atlas_warp = state.atlas_warp
        ctx.synthesis = state.synthesis
        ctx.output_volume = state.output_volume
        run_modular_assembly_block(ctx)
        apply_modular_context_to_state(state, ctx)

    state.response = await build_reconstruct_response(state)
