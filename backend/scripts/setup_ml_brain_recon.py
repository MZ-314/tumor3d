#!/usr/bin/env python3
"""Bootstrap ML brain volume generator checkpoint (train on atlas template)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BACKEND))

_INSTALL_HINT = (
    "Python dependencies missing. On RunPod run:\n"
    "  cd /workspace/tumor3d/backend\n"
    '  pip install -e ".[dev,gpu,dicom]"\n'
    "  pip install huggingface_hub pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg\n"
    "  pip install git+https://github.com/facebookresearch/segment-anything.git\n"
    "  pip install onnxruntime xatlas==0.0.9 moderngl SimpleITK"
)


def _require_deps() -> None:
    try:
        import pydantic  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:
        raise SystemExit(_INSTALL_HINT) from exc


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    _require_deps()

    import torch

    from config_pipeline import ATLAS_BRAIN_TEMPLATE, ML_VOLUME_MODEL_DIR
    from pipeline.ml.training.train_volume_generator import train

    if not ATLAS_BRAIN_TEMPLATE.is_file():
        raise SystemExit(
            f"Brain atlas template missing at {ATLAS_BRAIN_TEMPLATE}\n"
            "Run first: python backend/scripts/setup_brain_atlas.py"
        )

    out = ML_VOLUME_MODEL_DIR / "volume_generator.pt"
    if out.is_file():
        logging.info("Checkpoint already exists at %s — retraining bootstrap model", out)

    train(
        [ATLAS_BRAIN_TEMPLATE],
        output_path=out,
        epochs=50,
        batch_size=8,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    logging.info("ML brain volume generator ready: %s", out)


if __name__ == "__main__":
    main()
