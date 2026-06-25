"""Medical imaging pipeline configuration."""

from __future__ import annotations

import os
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_ROOT.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", str(REPO_ROOT / "data")))
UPLOADS_DIR = DATA_DIR / "uploads"
OUTPUTS_DIR = DATA_DIR / "outputs"
DB_PATH = Path(os.environ.get("CHAT_DB_PATH", str(DATA_DIR / "chat.db")))

SEGMENTATION_BACKEND = os.environ.get("SEGMENTATION_BACKEND", "monai").lower()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# Physical spacing assumptions when DICOM metadata is missing (mm).
DEFAULT_PIXEL_SPACING_MM = float(os.environ.get("DEFAULT_PIXEL_SPACING_MM", "1.0"))
DEFAULT_SLICE_THICKNESS_MM = float(os.environ.get("DEFAULT_SLICE_THICKNESS_MM", "5.0"))

# Single-slice depth estimate: thin slab (not a thick fake 3D tumor).
SINGLE_SLICE_DEPTH_VOXELS = int(os.environ.get("SINGLE_SLICE_DEPTH_VOXELS", "2"))

MONAI_BUNDLE_DIR = os.environ.get("MONAI_BUNDLE_DIR", "")
MONAI_WT_THRESHOLD = float(os.environ.get("MONAI_WT_THRESHOLD", "0.35"))

RECONSTRUCT_TIMEOUT_SEC = int(os.environ.get("RECONSTRUCT_TIMEOUT_SEC", "300"))


class MedicalPipelineError(Exception):
    """Base error for medical pipeline failures."""


class SegmentationError(MedicalPipelineError):
    """Segmentation step failed."""


class MeshExportError(MedicalPipelineError):
    """Mesh generation or export failed."""


def ensure_data_dirs() -> None:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
