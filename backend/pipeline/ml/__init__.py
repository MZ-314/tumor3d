"""ML reconstruction: organ routing, pose estimation, conditional volume synthesis."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["PoseEstimate", "estimate_pose", "generate_ml_volume"]

if TYPE_CHECKING:
    from pipeline.ml.types import PoseEstimate


def __getattr__(name: str):
    if name == "PoseEstimate":
        from pipeline.ml.types import PoseEstimate

        return PoseEstimate
    if name == "estimate_pose":
        from pipeline.ml.pose import estimate_pose

        return estimate_pose
    if name == "generate_ml_volume":
        from pipeline.ml.volume_generator import generate_ml_volume

        return generate_ml_volume
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
