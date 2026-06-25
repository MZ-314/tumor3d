"""Consensus — anatomical map already fused in consensus stage."""

from __future__ import annotations

from pipeline.modular.types import ModularContext


def run_consensus(ctx: ModularContext) -> None:
    if ctx.anatomical_map is None:
        raise RuntimeError("anatomical_map required for modular reconstruction")
