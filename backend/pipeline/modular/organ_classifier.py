"""Organ detection — route to brain modular pack."""

from __future__ import annotations

from pipeline.modular.types import ModularContext
from shared.schemas.pydantic.pipeline import OrganType


def run_organ_classifier(ctx: ModularContext) -> OrganType:
    return ctx.scan_context.organ_type
