#!/usr/bin/env python3
"""Download MONAI BraTS bundle for RunPod GPU inference."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BACKEND))

from pipeline.segment.monai_brats import ensure_brats_bundle  # noqa: E402


def main() -> int:
    try:
        path = ensure_brats_bundle()
        print(f"OK: MONAI bundle ready at {path}")
        print("Next:")
        print("  export SEGMENTATION_BACKEND=monai")
        print("  export PYTHONPATH=/workspace/tumor3d:/workspace/tumor3d/backend")
        print("  uvicorn backend.api.main:app --host 0.0.0.0 --port 8000")
        return 0
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
