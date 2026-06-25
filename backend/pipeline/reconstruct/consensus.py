"""Consensus engine — fuse multi-model 2D outputs (Phase 3)."""

from __future__ import annotations

import numpy as np

from config_medical import MedicalPipelineError
from pipeline.ingest.images import SliceVolume
from pipeline.segment.backends import LesionMask, SegmentationResult
from pipeline.segment.medsam import save_mask_png
from shared.schemas.pydantic.common import ProvenanceField, SourceType, Vec3
from shared.schemas.pydantic.pipeline import AnatomicalMap, Landmark2D


def _mask_centroid_mm(mask: np.ndarray, spacing: tuple[float, float]) -> Vec3:
    rows, cols = np.where(mask)
    if rows.size == 0:
        return Vec3(x=0.0, y=0.0, z=0.0)
    row_sp, col_sp = spacing
    return Vec3(
        x=float(cols.mean() * col_sp),
        y=float(rows.mean() * row_sp),
        z=0.0,
    )


def _organ_area_mm2(mask: np.ndarray, spacing: tuple[float, float]) -> float:
    return float(mask.sum()) * spacing[0] * spacing[1]


def build_anatomical_map(
    *,
    volume: SliceVolume,
    organ_mask_2d: np.ndarray,
    organ_confidence: float,
    segmentation: SegmentationResult,
    landmarks: list[Landmark2D],
    work_dir,
) -> AnatomicalMap:
    organ_path = work_dir / "organ_mask.png"
    save_mask_png(organ_mask_2d, organ_path)

    lesion_paths: list[str] = []
    z_mid = volume.data.shape[0] // 2
    spacing = volume.pixel_spacing_mm

    for i, lesion in enumerate(segmentation.lesions, start=1):
        slice_mask = lesion.mask[z_mid] if lesion.mask.ndim == 3 else lesion.mask
        lp = work_dir / f"lesion_mask_{i}.png"
        save_mask_png(slice_mask, lp)
        lesion_paths.append(str(lp.relative_to(work_dir)))

    lesion_centroids: list[Vec3] = []
    for lesion in segmentation.lesions:
        slice_mask = lesion.mask[z_mid] if lesion.mask.ndim == 3 else lesion.mask
        c = _mask_centroid_mm(slice_mask, spacing)
        lesion_centroids.append(c)

    midline_col = organ_mask_2d.shape[1] / 2.0
    organ_centroid = _mask_centroid_mm(organ_mask_2d, spacing)
    distances: dict[str, ProvenanceField] = {}
    if lesion_centroids:
        dist = abs(lesion_centroids[0].x - midline_col * spacing[1])
        distances["lesion_to_midline_mm"] = ProvenanceField(
            value=dist,
            confidence=organ_confidence,
            source=SourceType.MEASURED,
        )

    notes: list[str] = []
    if organ_confidence < 0.5:
        notes.append("Low MedSAM organ confidence — verify ROI on uploaded slice.")
    if not segmentation.lesions:
        notes.append("MONAI found no whole-tumor region on this volume.")

    return AnatomicalMap(
        organ_mask_path=str(organ_path.relative_to(work_dir)),
        lesion_mask_paths=lesion_paths,
        organ_area_mm2=ProvenanceField(
            value=_organ_area_mm2(organ_mask_2d, spacing),
            confidence=organ_confidence,
            source=SourceType.MEASURED,
        ),
        lesion_centroids_mm=lesion_centroids,
        landmarks=landmarks,
        distances_mm=distances,
        fusion_confidence=float(
            np.clip((organ_confidence + segmentation.global_confidence) / 2.0, 0.0, 1.0)
        ),
        consensus_notes=notes,
    )
