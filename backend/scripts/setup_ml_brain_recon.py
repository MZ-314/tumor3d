#!/usr/bin/env python3
"""Bootstrap ML brain volume generator + 3D refiner checkpoints."""

from __future__ import annotations

import argparse
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
)


def _require_deps() -> None:
    try:
        import pydantic  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:
        raise SystemExit(_INSTALL_HINT) from exc


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Bootstrap ML brain reconstruction models")
    parser.add_argument("--force", action="store_true", help="Retrain even if checkpoints exist")
    parser.add_argument("--epochs", type=int, default=20, help="Slice generator epochs")
    parser.add_argument("--refiner-epochs", type=int, default=25, help="3D refiner epochs")
    args = parser.parse_args()

    _require_deps()

    import torch

    from config_pipeline import ATLAS_BRAIN_TEMPLATE, ML_VOLUME_MODEL_DIR
    from pipeline.ml.training.train_volume_generator import train as train_slices
    from pipeline.ml.training.train_volume_refiner_3d import train as train_refiner

    if not ATLAS_BRAIN_TEMPLATE.is_file():
        raise SystemExit(
            f"Brain atlas template missing at {ATLAS_BRAIN_TEMPLATE}\n"
            "Run first: python backend/scripts/setup_brain_atlas.py"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    volumes = [ATLAS_BRAIN_TEMPLATE]
    slice_out = ML_VOLUME_MODEL_DIR / "volume_generator.pt"
    refiner_out = ML_VOLUME_MODEL_DIR / "volume_refiner_3d.pt"

    if slice_out.is_file() and refiner_out.is_file() and not args.force:
        print(f"OK: slice model {slice_out}")
        print(f"OK: 3D refiner {refiner_out}")
        return

    if not slice_out.is_file() or args.force:
        print(f"[1/2] Training parallel-slice generator on {device} …", flush=True)
        train_slices(
            volumes,
            output_path=slice_out,
            epochs=args.epochs,
            batch_size=16,
            samples_per_volume=192,
            device=device,
        )
        print(f"Slice generator ready: {slice_out}", flush=True)

    if not refiner_out.is_file() or args.force:
        print(f"[2/2] Training 3D volume refiner on {device} …", flush=True)
        train_refiner(
            volumes,
            output_path=refiner_out,
            epochs=args.refiner_epochs,
            batch_size=4,
            device=device,
        )
        print(f"3D refiner ready: {refiner_out}", flush=True)

    print("ML brain reconstruction models ready.", flush=True)


if __name__ == "__main__":
    main()
