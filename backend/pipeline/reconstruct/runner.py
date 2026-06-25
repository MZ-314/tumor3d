"""Orchestrate reconstruction stages with structured logging."""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from config_pipeline import PIPELINE_VERSION
from pipeline.reconstruct.context import PipelineState
from pipeline.reconstruct.stages.atlas import run_atlas_matching
from pipeline.reconstruct.stages.blueprint import run_blueprint
from pipeline.reconstruct.stages.consensus import run_consensus
from pipeline.reconstruct.stages.input_intelligence import run_input_intelligence
from pipeline.reconstruct.stages.medical_analysis import run_medical_analysis
from pipeline.reconstruct.stages.mesh import run_mesh_generation
from pipeline.reconstruct.stages.synthesis import run_synthesis
from pipeline.reconstruct.stages.validation import run_validation
from shared.schemas.pydantic.pipeline import StageTiming
from shared.schemas.pydantic.reconstruct import ReconstructResponse

logger = logging.getLogger(__name__)

StageFn = Callable[[PipelineState], Awaitable[None]]

STAGES: list[tuple[str, StageFn]] = [
    ("input_intelligence", run_input_intelligence),
    ("medical_analysis", run_medical_analysis),
    ("consensus", run_consensus),
    ("atlas_matching", run_atlas_matching),
    ("blueprint", run_blueprint),
    ("synthesis", run_synthesis),
    ("mesh_generation", run_mesh_generation),
    ("validation", run_validation),
]


async def _run_stages(
    state: PipelineState,
    *,
    on_stage: Callable[[str], None] | None = None,
) -> PipelineState:
    for name, fn in STAGES:
        if on_stage:
            on_stage(name)
        started = time.perf_counter()
        try:
            await fn(state)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            state.timings.append(StageTiming(stage=name, duration_ms=elapsed_ms, status="ok"))
            logger.info(
                "Pipeline %s stage=%s duration_ms=%.1f",
                state.reconstruction_id,
                name,
                elapsed_ms,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            state.timings.append(
                StageTiming(
                    stage=name,
                    duration_ms=elapsed_ms,
                    status="error",
                    message=str(exc),
                )
            )
            logger.exception(
                "Pipeline %s failed at stage=%s",
                state.reconstruction_id,
                name,
            )
            raise

    artifacts = state.artifacts(PIPELINE_VERSION)
    artifacts_path = state.work_dir / "pipeline_artifacts.json"
    artifacts_path.write_text(artifacts.model_dump_json(indent=2), encoding="utf-8")

    if state.response is None:
        raise RuntimeError("mesh_generation did not produce ReconstructResponse")

    state.response.pipeline_artifacts = artifacts
    return state


async def run_reconstruction_pipeline(
    slice_paths: list[Path],
    work_dir: Path,
    *,
    modality: str = "brain_mri",
    chat_id: str | None = None,
    user_text: str | None = None,
    on_stage: Callable[[str], None] | None = None,
) -> ReconstructResponse:
    """Run the full multi-stage reconstruction pipeline (GPU required for brain MRI)."""
    state = PipelineState(
        reconstruction_id=work_dir.name,
        work_dir=work_dir,
        slice_paths=slice_paths,
        modality=modality,
        chat_id=chat_id,
        user_text=user_text,
    )
    await _run_stages(state, on_stage=on_stage)
    return state.response
