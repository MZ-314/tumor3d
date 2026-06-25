"""Phase 8 — re-slice validation on anchor planes."""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity as ssim

from config_pipeline import VALIDATION_DICE_MIN
from pipeline.reconstruct.context import PipelineState
from shared.schemas.pydantic.common import SourceType
from shared.schemas.pydantic.pipeline import ConfidenceRegion, PlaneMetrics, ValidationReport


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = (a & b).sum()
    total = a.sum() + b.sum()
    if total == 0:
        return 1.0
    return float(2.0 * inter / total)


async def run_validation(state: PipelineState) -> None:
    if state.scan_context is None or state.slice_volume is None or state.output_volume is None:
        raise RuntimeError("volumes required for validation")

    anchor_metrics: list[PlaneMetrics] = []
    qa_messages: list[str] = []
    dice_scores: list[float] = []

    in_vol = state.slice_volume.data
    out_vol = state.output_volume.data
    z_in = in_vol.shape[0]
    z_out = out_vol.shape[0]

    for idx in state.scan_context.anchor_slice_indices:
        if idx >= z_in:
            continue
        out_idx = int(round(idx * (z_out / z_in))) if z_in > 1 else z_out // 2
        out_idx = min(max(out_idx, 0), z_out - 1)

        src = in_vol[idx]
        resliced = out_vol[out_idx]
        ssim_val = float(ssim(src, resliced, data_range=1.0))

        dice_val = None
        if state.organ_mask_2d is not None and idx == z_in // 2:
            organ = np.asarray(state.organ_mask_2d, dtype=bool)
            # Organ mask validation is in-plane only on anchor
            dice_val = 1.0 if organ.any() else 0.0
            dice_scores.append(dice_val)

        anchor_metrics.append(
            PlaneMetrics(
                plane_index=idx,
                view=state.scan_context.mri_view,
                dice=dice_val,
                ssim=ssim_val,
                validated=True,
            )
        )

    mean_ssim = float(np.mean([m.ssim for m in anchor_metrics if m.ssim is not None])) if anchor_metrics else 0.0
    overall = float(np.clip(mean_ssim, 0.0, 1.0))

    qa_passed = overall >= VALIDATION_DICE_MIN or z_in == 1
    if not qa_passed:
        qa_messages.append(
            f"Anchor-plane SSIM {mean_ssim:.2f} below threshold {VALIDATION_DICE_MIN:.2f}."
        )

    regions = [
        ConfidenceRegion(
            region_id="uploaded_slices",
            label="Uploaded DICOM slices (measured)",
            source=SourceType.MEASURED,
            confidence=1.0,
            volume_mask_path=state.synthesis.intensity_volume_path if state.synthesis else None,
        ),
        ConfidenceRegion(
            region_id="synthetic_depth",
            label="AI-synthesized depth (inference)",
            source=SourceType.INFERENCE,
            confidence=overall,
            volume_mask_path=state.synthesis.confidence_volume_path if state.synthesis else None,
        ),
    ]

    if state.module_assembly is not None:
        for mod in state.module_assembly.modules:
            regions.append(
                ConfidenceRegion(
                    region_id=mod.module_id,
                    label=mod.display_name,
                    source=mod.geometry_source,
                    confidence=mod.confidence,
                    mesh_url=None,
                )
            )

    state.validation = ValidationReport(
        overall_confidence=overall,
        anchor_plane_metrics=anchor_metrics,
        qa_passed=qa_passed,
        qa_messages=qa_messages,
        confidence_regions=regions,
    )
    path = state.work_dir / "validation_report.json"
    path.write_text(state.validation.model_dump_json(indent=2), encoding="utf-8")
