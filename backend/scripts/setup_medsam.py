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

# Official MedSAM weights (Zenodo) — GitHub raw URL is no longer valid.
MEDSAM_ZENODO_URL = "https://zenodo.org/records/10689643/files/medsam_vit_b.pth"
MEDSAM_HF_URL = (
    "https://huggingface.co/GleghornLab/medsam-vit-b/resolve/main/medsam_vit_b.pth"
)


def _download(url: str, dest: Path) -> None:
    print(f"Trying {url} …")
    urllib.request.urlretrieve(url, dest)


def main() -> int:
    MEDSAM_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    if MEDSAM_CHECKPOINT.is_file():
        print(f"OK: MedSAM checkpoint already at {MEDSAM_CHECKPOINT}")
        return 0

    print(f"Downloading MedSAM to {MEDSAM_CHECKPOINT} …")
    errors: list[str] = []

    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id="GleghornLab/medsam-vit-b",
            filename="medsam_vit_b.pth",
            local_dir=str(MEDSAM_CHECKPOINT.parent),
        )
        downloaded = Path(path)
        if downloaded.resolve() != MEDSAM_CHECKPOINT.resolve():
            downloaded.replace(MEDSAM_CHECKPOINT)
        print("OK: MedSAM ready (Hugging Face)")
        _print_next_steps()
        return 0
    except Exception as exc:
        errors.append(f"Hugging Face: {exc}")

    for url in (MEDSAM_ZENODO_URL, MEDSAM_HF_URL):
        try:
            _download(url, MEDSAM_CHECKPOINT)
            if MEDSAM_CHECKPOINT.stat().st_size < 1_000_000:
                raise OSError("download too small — likely an error page")
            print(f"OK: MedSAM ready ({url})")
            _print_next_steps()
            return 0
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            MEDSAM_CHECKPOINT.unlink(missing_ok=True)

    print("FAIL: could not download MedSAM checkpoint.", file=sys.stderr)
    for line in errors:
        print(f"  - {line}", file=sys.stderr)
    print(
        "\nManual: wget -O "
        f"{MEDSAM_CHECKPOINT} {MEDSAM_ZENODO_URL}",
        file=sys.stderr,
    )
    print(
        "Also: pip install git+https://github.com/facebookresearch/segment-anything.git",
        file=sys.stderr,
    )
    return 1


def _print_next_steps() -> None:
    print("Next:")
    print("  pip install git+https://github.com/facebookresearch/segment-anything.git")
    print("  export SEGMENTATION_BACKEND=monai")


if __name__ == "__main__":
    raise SystemExit(main())
