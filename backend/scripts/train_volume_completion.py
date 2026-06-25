#!/usr/bin/env python3
"""Train 3D volume completion model on BraTS bootstrap (anchor-locked Niivue export)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BACKEND))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Train modular volume completion (3D refiner)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print(
            "Missing torch. On RunPod:\n"
            '  cd /workspace/tumor3d/backend && pip install -e ".[dev,gpu,dicom]"'
        )
        return 1

    from config_pipeline import ATLAS_BRAIN_TEMPLATE, ML_VOLUME_MODEL_DIR
    from pipeline.ml.training.brats_bootstrap import ensure_brats_bootstrap_volumes
    from pipeline.ml.training.train_volume_refiner_3d import train as train_refiner

    if not ATLAS_BRAIN_TEMPLATE.is_file():
        print("Run first: python backend/scripts/setup_brain_atlas.py")
        return 1

    out = ML_VOLUME_MODEL_DIR / "volume_refiner_3d.pt"
    if out.is_file() and not args.force:
        print(f"OK: volume completion model at {out}")
        return 0

    volumes = ensure_brats_bootstrap_volumes(min_volumes=4)
    print(f"Training volume completion on {len(volumes)} bootstrap volumes …")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ML_VOLUME_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train_refiner(
        volume_paths=volumes,
        output_path=out,
        epochs=args.epochs,
        device=device,
    )
    print(f"OK: volume completion model at {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
