"""Atlas-anchored 3D volume synthesis from a single patient slice."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pipeline.ingest.images import SliceVolume
from pipeline.reconstruct.atlas_volume import build_registered_atlas_volume
from shared.schemas.pydantic.pipeline import AtlasWarpResult, MriView


def synthesize_atlas_anchored_volume(
    volume: SliceVolume,
    *,
    target_z: int,
    organ_mask_2d: np.ndarray | None = None,
    mri_view: MriView = MriView.UNKNOWN,
    atlas_warp: AtlasWarpResult | None = None,
    work_dir: Path | None = None,
) -> tuple[np.ndarray, str]:
    synth, strategy, _ = build_registered_atlas_volume(
        volume,
        target_z=target_z,
        organ_mask_2d=organ_mask_2d,
        mri_view=mri_view,
        atlas_warp=atlas_warp,
        work_dir=work_dir,
    )
    return synth, strategy


def _fallback_z_expansion(patient_slice: np.ndarray, target_z: int) -> np.ndarray:
    h, w = patient_slice.shape
    anchor = target_z // 2
    synth = np.zeros((target_z, h, w), dtype=np.float32)
    synth[anchor] = patient_slice
    for z in range(target_z):
        if z == anchor:
            continue
        weight = 1.0 - abs(z - anchor) / max(anchor, 1)
        synth[z] = patient_slice * (0.85 + 0.15 * weight)
    return synth
