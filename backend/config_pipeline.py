"""Reconstruction pipeline configuration (Phases 0–10)."""

from __future__ import annotations

import os
from pathlib import Path

from config_medical import REPO_ROOT

PIPELINE_VERSION = os.environ.get("RECONSTRUCT_PIPELINE_VERSION", "0.1.0")

PARTIAL_VOLUME_MIN_SLICES = int(os.environ.get("PARTIAL_VOLUME_MIN_SLICES", "6"))
MULTI_SLICE_MIN_SLICES = int(os.environ.get("MULTI_SLICE_MIN_SLICES", "10"))

MEDSAM_CHECKPOINT = Path(
    os.environ.get("MEDSAM_CHECKPOINT", str(REPO_ROOT / "models" / "medsam" / "medsam_vit_b.pth"))
)

ATLAS_BRAIN_DIR = Path(os.environ.get("ATLAS_BRAIN_DIR", str(REPO_ROOT / "data" / "atlases" / "brain")))
ATLAS_KNEE_DIR = Path(os.environ.get("ATLAS_KNEE_DIR", str(REPO_ROOT / "data" / "atlases" / "knee")))
ATLAS_BRAIN_TEMPLATE = ATLAS_BRAIN_DIR / "template.nii.gz"

NNU_NET_MODEL_DIR = Path(os.environ.get("NNU_NET_MODEL_DIR", str(REPO_ROOT / "models" / "nnunet")))

VALIDATION_DICE_MIN = float(os.environ.get("VALIDATION_DICE_MIN", "0.5"))

# Phase 6b — ML volume synthesis (default for single-slice USP)
SYNTHESIS_BACKEND = os.environ.get("SYNTHESIS_BACKEND", "ml").strip().lower()
ML_VOLUME_MODEL_DIR = Path(
    os.environ.get("ML_VOLUME_MODEL_DIR", str(REPO_ROOT / "models" / "brain_recon"))
)
# Modular brain reconstruction (default primary path)
MODULAR_RECON = os.environ.get("MODULAR_RECON", "1").strip().lower() in ("1", "true", "yes")
LEGACY_VOLUME_SYNTHESIS = os.environ.get("LEGACY_VOLUME_SYNTHESIS", "0").strip().lower() in (
    "1",
    "true",
    "yes",
)
MODULAR_BRAIN_DIR = Path(
    os.environ.get("MODULAR_BRAIN_DIR", str(ATLAS_BRAIN_DIR / "modules"))
)
