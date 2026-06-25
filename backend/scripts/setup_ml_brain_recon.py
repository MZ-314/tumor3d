#!/usr/bin/env python3
"""Bootstrap ML brain volume generator checkpoint (train on atlas template)."""

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
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Bootstrap ML brain volume generator")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even if checkpoint already exists",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Bootstrap epochs (default 20)")
    args = parser.parse_args()

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
    if out.is_file() and not args.force:
        print(f"OK: checkpoint already at {out} (use --force to retrain)")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"Bootstrap ML training on {device} using {ATLAS_BRAIN_TEMPLATE.name} …\n"
        "Do not interrupt — first lines appear after dataset build (~30s).",
        flush=True,
    )

    train(
        [ATLAS_BRAIN_TEMPLATE],
        output_path=out,
        epochs=args.epochs,
        batch_size=16,
        samples_per_volume=192,
        device=device,
    )
    print(f"ML brain volume generator ready: {out}", flush=True)


if __name__ == "__main__":
    main()
