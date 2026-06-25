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

# Primary: OASIS-1 T1 (figshare). Fallback: direct MNI152 1mm brain (public mirror).
TEMPLATE_URLS = (
    "https://ndownloader.figshare.com/files/3133838",
    "https://raw.githubusercontent.com/InstitutdeNeurosciencesdesSystems/ADF-pipeline-datarefs/main/MNI152_T1_1mm.nii.gz",
)


def main() -> int:
    ATLAS_BRAIN_DIR.mkdir(parents=True, exist_ok=True)
    if ATLAS_BRAIN_TEMPLATE.is_file():
        print(f"OK: brain atlas at {ATLAS_BRAIN_TEMPLATE}")
        return 0

    archive = ATLAS_BRAIN_DIR / "template_download"
    print(f"Downloading brain atlas to {ATLAS_BRAIN_DIR} …")
    last_err: Exception | None = None
    for url in TEMPLATE_URLS:
        try:
            dest = archive.with_suffix(".zip" if url.endswith(".zip") else ".nii.gz")
            if url.endswith(".nii.gz"):
                dest = ATLAS_BRAIN_DIR / "mni152_download.nii.gz"
                urllib.request.urlretrieve(url, dest)
                import shutil

                shutil.copy2(dest, ATLAS_BRAIN_TEMPLATE)
                dest.unlink(missing_ok=True)
                print(f"OK: brain atlas template at {ATLAS_BRAIN_TEMPLATE}")
                print("Also: pip install SimpleITK")
                return 0
            urllib.request.urlretrieve(url, dest)
            import zipfile

            with zipfile.ZipFile(dest, "r") as zf:
                zf.extractall(ATLAS_BRAIN_DIR)
            dest.unlink(missing_ok=True)
            candidates = sorted(ATLAS_BRAIN_DIR.rglob("*.nii*"))
            if not candidates:
                raise OSError("no NIfTI in archive")
            import shutil

            shutil.copy2(candidates[0], ATLAS_BRAIN_TEMPLATE)
            print(f"OK: brain atlas template at {ATLAS_BRAIN_TEMPLATE}")
            print("Also: pip install SimpleITK")
            return 0
        except Exception as exc:
            last_err = exc
            continue

    print(
        "Manual: place a T1 brain template NIfTI at",
        ATLAS_BRAIN_TEMPLATE,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
