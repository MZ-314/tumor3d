#!/usr/bin/env python3
"""Download MNI brain template for atlas registration (Phase 4)."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BACKEND))

from config_pipeline import ATLAS_BRAIN_DIR, ATLAS_BRAIN_TEMPLATE  # noqa: E402

# OASIS-1 single T1 1mm (public); small template suitable for registration bootstrap.
TEMPLATE_URL = (
    "https://ndownloader.figshare.com/files/3133838"
)


def main() -> int:
    ATLAS_BRAIN_DIR.mkdir(parents=True, exist_ok=True)
    if ATLAS_BRAIN_TEMPLATE.is_file():
        print(f"OK: brain atlas at {ATLAS_BRAIN_TEMPLATE}")
        return 0

    archive = ATLAS_BRAIN_DIR / "template_archive.zip"
    print(f"Downloading brain atlas to {ATLAS_BRAIN_DIR} …")
    try:
        urllib.request.urlretrieve(TEMPLATE_URL, archive)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        print(
            "Manual: place a T1 brain template NIfTI at",
            ATLAS_BRAIN_TEMPLATE,
        )
        return 1

    import zipfile

    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(ATLAS_BRAIN_DIR)
    archive.unlink(missing_ok=True)

    # Find first .nii or .nii.gz
    candidates = sorted(ATLAS_BRAIN_DIR.rglob("*.nii*"))
    if not candidates:
        print("FAIL: no NIfTI found in downloaded archive", file=sys.stderr)
        return 1

    import shutil

    shutil.copy2(candidates[0], ATLAS_BRAIN_TEMPLATE)

    print(f"OK: brain atlas template at {ATLAS_BRAIN_TEMPLATE}")
    print("Also: pip install SimpleITK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
