"""Multi-model AI outputs are produced upstream in medical_analysis."""

from __future__ import annotations

from pipeline.modular.types import ModularContext


def run_multi_model_ai(ctx: ModularContext) -> None:
    """No-op: segmentation already in pipeline state before modular block."""
    _ = ctx
