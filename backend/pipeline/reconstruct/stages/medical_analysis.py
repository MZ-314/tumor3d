"""Phase 2 — MedSAM + MONAI + structure analysis."""

from __future__ import annotations

import asyncio

import numpy as np

from config_medical import SEGMENTATION_BACKEND
from config_pipeline import PIPELINE_VERSION
from pipeline.reconstruct.context import PipelineState
from pipeline.segment.backends import VOLUME_ONLY_MODALITIES, segment_volume
from pipeline.segment.medsam import save_mask_png, segment_organ_2d
from shared.schemas.pydantic.pipeline import Landmark2D, LesionCandidate2D, ModelOutputs


async def run_medical_analysis(state: PipelineState) -> None:
    if state.slice_volume is None:
        raise RuntimeError("slice_volume required — run input_intelligence first")

    volume = state.slice_volume
    z_mid = volume.data.shape[0] // 2
    slice_img = volume.data[z_mid]
    modality = state.modality.lower()

    organ_mask_path = None
    organ_confidence = 0.0
    landmarks: list[Landmark2D] = []
    lesion_candidates: list[LesionCandidate2D] = []

    if modality not in VOLUME_ONLY_MODALITIES:
        organ_mask, organ_confidence = await asyncio.to_thread(segment_organ_2d, slice_img)
        state.organ_mask_2d = organ_mask
        organ_mask_path = state.work_dir / "medsam_organ_mask.png"
        save_mask_png(organ_mask, organ_mask_path)

        rows, cols = np.where(organ_mask)
        if rows.size:
            landmarks.append(
                Landmark2D(
                    name="organ_centroid",
                    row=float(rows.mean()),
                    col=float(cols.mean()),
                    confidence=organ_confidence,
                )
            )

        seg = await asyncio.to_thread(
            segment_volume,
            volume.data,
            SEGMENTATION_BACKEND,
            modality=state.modality,
        )
        state.segmentation = seg

        for lesion in seg.lesions:
            lesion_candidates.append(
                LesionCandidate2D(
                    lesion_id=str(lesion.lesion_id),
                    in_plane_confidence=lesion.in_plane_confidence,
                    model_source="monai_brats",
                )
            )
    else:
        state.segmentation = await asyncio.to_thread(
            segment_volume,
            volume.data,
            SEGMENTATION_BACKEND,
            modality=state.modality,
        )

    state.model_outputs = ModelOutputs(
        organ_mask_path=str(organ_mask_path.relative_to(state.work_dir)) if organ_mask_path else None,
        organ_confidence=organ_confidence,
        lesion_candidates=lesion_candidates,
        landmarks=landmarks,
        organ_classifier_confidence=1.0 if state.scan_context else 0.0,
        model_versions={
            "pipeline": PIPELINE_VERSION,
            "organ_segmentation": "medsam_vit_b",
            "abnormality_detection": "monai_brats",
            "medical_vision": "landmarks_from_medsam",
        },
    )
    path = state.work_dir / "model_outputs.json"
    path.write_text(state.model_outputs.model_dump_json(indent=2), encoding="utf-8")
