"""Configuration and exceptions for the 3D reconstruction pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass


class Reconstruction3DError(Exception):
    """Base error for the reconstruction pipeline."""


class Stage1IsolationError(Reconstruction3DError):
    """SAM 2 foreground isolation failed."""


class Stage2ReconstructionError(Reconstruction3DError):
    """TRELLIS.2 image-to-3D inference failed."""


class Stage3ExportError(Reconstruction3DError):
    """Blender mesh processing or export failed."""


@dataclass(frozen=True)
class ReconstructionConfig:
    """Runtime configuration loaded from environment variables."""

    sam2_checkpoint: str
    sam2_config: str
    sam2_device: str
    trellis2_model_id: str
    trellis2_device: str
    trellis2_seed: int
    trellis2_decimation_target: int
    trellis2_texture_size: int
    blender_bin: str
    blender_decimate_ratio: float
    blender_max_triangles: int
    stage1_timeout_s: float
    stage2_timeout_s: float
    stage3_timeout_s: float
    mask_min_area_ratio: float
    mask_max_area_ratio: float
    mask_min_score: float

    @classmethod
    def from_env(cls) -> ReconstructionConfig:
        return cls(
            sam2_checkpoint=os.getenv(
                "SAM2_CHECKPOINT", "checkpoints/sam2.1_hiera_large.pt"
            ),
            sam2_config=os.getenv("SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_l.yaml"),
            sam2_device=os.getenv("SAM2_DEVICE", "cuda"),
            trellis2_model_id=os.getenv(
                "TRELLIS2_MODEL_ID", "microsoft/TRELLIS.2-4B"
            ),
            trellis2_device=os.getenv("TRELLIS2_DEVICE", "cuda"),
            trellis2_seed=int(os.getenv("TRELLIS2_SEED", "42")),
            trellis2_decimation_target=int(
                os.getenv("TRELLIS2_DECIMATION_TARGET", "500000")
            ),
            trellis2_texture_size=int(os.getenv("TRELLIS2_TEXTURE_SIZE", "2048")),
            blender_bin=os.getenv("BLENDER_BIN", "blender"),
            blender_decimate_ratio=float(os.getenv("BLENDER_DECIMATE_RATIO", "0.5")),
            blender_max_triangles=int(os.getenv("BLENDER_MAX_TRIANGLES", "100000")),
            stage1_timeout_s=float(os.getenv("RECON_STAGE1_TIMEOUT_S", "300")),
            stage2_timeout_s=float(os.getenv("RECON_STAGE2_TIMEOUT_S", "600")),
            stage3_timeout_s=float(os.getenv("RECON_STAGE3_TIMEOUT_S", "300")),
            mask_min_area_ratio=float(os.getenv("SAM2_MASK_MIN_AREA_RATIO", "0.01")),
            mask_max_area_ratio=float(os.getenv("SAM2_MASK_MAX_AREA_RATIO", "0.95")),
            mask_min_score=float(os.getenv("SAM2_MASK_MIN_SCORE", "0.05")),
        )


DEFAULT_CONFIG = ReconstructionConfig.from_env()
