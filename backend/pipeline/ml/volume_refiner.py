"""Apply 3D volumetric refiner to coarse ML slab."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

from config_pipeline import ML_VOLUME_MODEL_DIR
from pipeline.ml.brain_envelope import apply_brain_envelope, build_brain_envelope_3d

logger = logging.getLogger(__name__)

REFINER_CUBE = 64


def _resolve_refiner_checkpoint() -> Path | None:
    path = ML_VOLUME_MODEL_DIR / "volume_refiner_3d.pt"
    return path if path.is_file() else None


def refine_volume_3d(
    volume: np.ndarray,
    organ_mask_2d: np.ndarray,
    *,
    anchor_z: int,
    anchor_plane: np.ndarray,
    background: float,
) -> tuple[np.ndarray, bool]:
    """
    Run 3D U-Net refiner if checkpoint exists; always apply ellipsoid brain mask.

    Returns (volume, refiner_applied).
    """
    z_count, h, w = volume.shape
    envelope = build_brain_envelope_3d((z_count, h, w), organ_mask_2d, anchor_z)
    masked = apply_brain_envelope(
        volume,
        envelope,
        anchor_z=anchor_z,
        anchor_plane=anchor_plane,
        background=background,
    )

    ckpt = _resolve_refiner_checkpoint()
    if ckpt is None:
        return masked, False

    import torch

    from pipeline.ml.models.volume_refiner_3d import load_volume_refiner_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _version = load_volume_refiner_checkpoint(ckpt, device=device)

    coarse_cube = zoom(
        masked.astype(np.float32),
        (REFINER_CUBE / z_count, REFINER_CUBE / h, REFINER_CUBE / w),
        order=1,
    )
    env_cube = zoom(
        envelope,
        (REFINER_CUBE / z_count, REFINER_CUBE / h, REFINER_CUBE / w),
        order=1,
    )
    inp = np.stack([coarse_cube, env_cube], axis=0).astype(np.float32)
    tensor = torch.from_numpy(inp).unsqueeze(0).to(device)

    with torch.no_grad():
        refined_cube = model(tensor).squeeze().cpu().numpy()

    refined = zoom(
        refined_cube,
        (z_count / REFINER_CUBE, h / REFINER_CUBE, w / REFINER_CUBE),
        order=1,
    ).astype(np.float32)
    refined = apply_brain_envelope(
        refined,
        envelope,
        anchor_z=anchor_z,
        anchor_plane=anchor_plane,
        background=background,
    )
    logger.info("3D volume refiner applied (%s)", ckpt.name)
    return refined, True
