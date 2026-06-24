"""Image-to-3D (TripoSR) configuration."""

from __future__ import annotations

import os
from pathlib import Path

from config_medical import REPO_ROOT

IMAGE3D_BACKEND = os.environ.get("IMAGE3D_BACKEND", "triposr").lower()
TRIPOSR_DIR = Path(os.environ.get("TRIPOSR_DIR", str(REPO_ROOT / "vendor" / "TripoSR")))
TRIPOSR_MC_RESOLUTION = int(os.environ.get("TRIPOSR_MC_RESOLUTION", "256"))


from config_medical import MedicalPipelineError


class Image3DError(MedicalPipelineError):
    """Image-to-3D pipeline failure."""
