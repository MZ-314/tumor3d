#!/usr/bin/env python3
"""Download MedSAM ViT-B checkpoint for organ segmentation."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BACKEND))

from config_pipeline import MEDSAM_CHECKPOINT  # noqa: E402

MEDSAM_URL = (
    "https://github.com/bowang-lab/MedSAM/raw/main/work_dir/MedSAM/medsam_vit_b.pth"
)


def main() -> int:
    MEDSAM_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    if MEDSAM_CHECKPOINT.is_file():
        print(f"OK: MedSAM checkpoint already at {MEDSAM_CHECKPOINT}")
        return 0

    print(f"Downloading MedSAM to {MEDSAM_CHECKPOINT} …")
    try:
        urllib.request.urlretrieve(MEDSAM_URL, MEDSAM_CHECKPOINT)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        print("Also install: pip install git+https://github.com/facebookresearch/segment-anything.git")
        return 1

    print("OK: MedSAM ready")
    print("Next:")
    print("  pip install git+https://github.com/facebookresearch/segment-anything.git")
    print("  export SEGMENTATION_BACKEND=monai")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
