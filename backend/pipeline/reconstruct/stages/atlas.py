"""Phase 4 — atlas matching."""

from __future__ import annotations

import numpy as np

from pipeline.reconstruct.atlas_register import run_atlas_for_organ
from pipeline.reconstruct.context import PipelineState
from pipeline.segment.backends import VOLUME_ONLY_MODALITIES
from pipeline.reconstruct.view_orient import detect_mri_view_from_pixels
from shared.schemas.pydantic.pipeline import MriView, OrganType


async def run_atlas_matching(state: PipelineState) -> None:
    if state.scan_context is None or state.slice_volume is None:
        raise RuntimeError("scan_context and slice_volume required")

    if state.modality.lower() in VOLUME_ONLY_MODALITIES:
        from shared.schemas.pydantic.pipeline import AtlasWarpResult

        state.atlas_warp = AtlasWarpResult(
            atlas_id="none",
            atlas_version="volume_only",
            registration_confidence=1.0,
        )
    elif state.scan_context.organ_type == OrganType.BRAIN:
        if state.organ_mask_2d is None:
            raise RuntimeError("organ_mask_2d required for brain atlas registration")
        anchor = state.scan_context.anchor_slice_indices[0]
        organ_mask = np.asarray(state.organ_mask_2d, dtype=bool)

        mri_view = state.scan_context.mri_view
        if mri_view == MriView.UNKNOWN:
            mri_view = detect_mri_view_from_pixels(state.slice_volume.data[anchor], organ_mask)
            state.scan_context = state.scan_context.model_copy(update={"mri_view": mri_view})

        state.atlas_warp = run_atlas_for_organ(
            state.scan_context.organ_type,
            state.slice_volume,
            organ_mask,
            work_dir=state.work_dir,
            anchor_z=anchor,
            mri_view=mri_view,
        )
    else:
        from config_medical import MedicalPipelineError

        raise MedicalPipelineError(
            f"Atlas pack for organ '{state.scan_context.organ_type.value}' is not installed yet."
        )

    path = state.work_dir / "atlas_warp.json"
    path.write_text(state.atlas_warp.model_dump_json(indent=2), encoding="utf-8")
