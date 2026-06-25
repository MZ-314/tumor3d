"""Spatial coordinate mapping — DICOM pose + optional ML pose net."""

from __future__ import annotations

import json

from pipeline.ml.pose import estimate_pose
from pipeline.modular.types import ModularContext
from shared.schemas.pydantic.pipeline import MriView, OrganType, PoseEstimate


def run_spatial_mapping(ctx: ModularContext) -> None:
    vol = ctx.slice_volume
    anchor = ctx.anchor_z
    mri_view = ctx.scan_context.mri_view
    try:
        ml_pose = estimate_pose(
            vol.data[anchor],
            organ_mask=ctx.organ_mask_2d,
        )
        pose = PoseEstimate(
            organ_type=ml_pose.organ_type,
            through_plane_axis=ml_pose.through_plane_axis,
            slice_index_normalized=ml_pose.slice_index_normalized,
            mri_view=ml_pose.mri_view,
            confidence=ml_pose.confidence,
            source=ml_pose.source,
        )
    except Exception:
        pose = PoseEstimate(
            organ_type=OrganType.BRAIN,
            through_plane_axis=0,
            slice_index_normalized=0.5,
            mri_view=mri_view if mri_view != MriView.UNKNOWN else MriView.AXIAL,
            confidence=0.5,
            source="heuristic",
        )
    ctx.pose = pose
    pose_path = ctx.work_dir / "pose_estimate.json"
    pose_path.write_text(pose.model_dump_json(indent=2), encoding="utf-8")
