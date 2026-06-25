"""Phase 7 — 3D volume and mesh generation."""

from __future__ import annotations

from pipeline.reconstruct.context import PipelineState
from pipeline.reconstruct.export_response import build_reconstruct_response


async def run_mesh_generation(state: PipelineState) -> None:
    state.response = await build_reconstruct_response(state)
