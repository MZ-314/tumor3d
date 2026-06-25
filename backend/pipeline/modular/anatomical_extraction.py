"""Anatomical extraction — measurements from masks + DICOM spacing."""

from __future__ import annotations

import numpy as np

from pipeline.modular.types import ModularContext
from shared.schemas.pydantic.common import ProvenanceField, SourceType, Vec3


def run_anatomical_extraction(ctx: ModularContext) -> None:
    if ctx.anatomical_map is None:
        return
    row_sp, col_sp = ctx.slice_volume.pixel_spacing_mm
    organ = ctx.organ_mask_2d
    area_px = float(organ.sum())
    area_mm2 = area_px * row_sp * col_sp
    ctx.anatomical_map = ctx.anatomical_map.model_copy(
        update={
            "organ_area_mm2": ProvenanceField(
                value=area_mm2,
                confidence=0.9,
                source=SourceType.MEASURED,
            )
        }
    )
    if ctx.lesion_mask_2d is not None and ctx.lesion_mask_2d.any():
        rows, cols = np.where(ctx.lesion_mask_2d)
        cx = float(cols.mean()) * col_sp
        cy = float(rows.mean()) * row_sp
        ctx.anatomical_map.lesion_centroids_mm = [
            Vec3(x=cx, y=cy, z=float(ctx.anchor_z) * ctx.slice_volume.slice_spacing_mm)
        ]
