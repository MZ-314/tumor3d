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

from config_pipeline import ATLAS_BRAIN_TEMPLATE, ML_VOLUME_MODEL_DIR  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch required for ML reconstruction. Install with:\n"
            '  pip install -e ".[gpu]"'
        ) from exc

    if not ATLAS_BRAIN_TEMPLATE.is_file():
        raise SystemExit(
            f"Brain atlas template missing at {ATLAS_BRAIN_TEMPLATE}\n"
            "Run first: python backend/scripts/setup_brain_atlas.py"
        )

    from pipeline.ml.training.train_volume_generator import train

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
