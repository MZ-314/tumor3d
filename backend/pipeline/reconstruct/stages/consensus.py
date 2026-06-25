"""Phase 3 — consensus engine."""

from __future__ import annotations

import numpy as np

from pipeline.reconstruct.consensus import build_anatomical_map
from pipeline.reconstruct.context import PipelineState
from pipeline.segment.backends import VOLUME_ONLY_MODALITIES


async def run_consensus(state: PipelineState) -> None:
    if state.slice_volume is None or state.segmentation is None or state.model_outputs is None:
        raise RuntimeError("medical_analysis must complete before consensus")

    modality = state.modality.lower()
    if modality in VOLUME_ONLY_MODALITIES:
        from shared.schemas.pydantic.pipeline import AnatomicalMap

        state.anatomical_map = AnatomicalMap(
            fusion_confidence=1.0,
            consensus_notes=["Volume-only modality — no organ/lesion fusion."],
        )
    else:
        if state.organ_mask_2d is None:
            raise RuntimeError("organ_mask_2d missing from MedSAM")
        organ_mask = np.asarray(state.organ_mask_2d, dtype=bool)
        state.anatomical_map = build_anatomical_map(
            volume=state.slice_volume,
            organ_mask_2d=organ_mask,
            organ_confidence=state.model_outputs.organ_confidence,
            segmentation=state.segmentation,
            landmarks=state.model_outputs.landmarks,
            work_dir=state.work_dir,
        )

    path = state.work_dir / "anatomical_map.json"
    path.write_text(state.anatomical_map.model_dump_json(indent=2), encoding="utf-8")
