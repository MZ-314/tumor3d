"""Shared types for ML reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

from shared.schemas.pydantic.pipeline import MriView, OrganType


@dataclass(frozen=True)
class PoseEstimate:
    """Continuous slice pose in canonical brain space (no manual view selection)."""

    organ_type: OrganType
    through_plane_axis: int
    """0 = stack along volume Z (axial), 1 = coronal, 2 = sagittal."""
    slice_index_normalized: float
    """0–1 depth along through-plane axis."""
    mri_view: MriView
    confidence: float
    source: str
