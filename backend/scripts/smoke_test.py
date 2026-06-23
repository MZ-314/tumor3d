#!/usr/bin/env python3
"""Smoke test for POST /reconstruct (requires GPU backend running)."""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "fixtures"


def make_sample_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (512, 512), color=(245, 245, 245))
    draw = ImageDraw.Draw(img)
    draw.ellipse([160, 140, 352, 372], fill=(180, 90, 70))
    img.save(path)


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    sample = FIXTURES / "sample-object.png"
    if not sample.exists():
        make_sample_png(sample)

    health = httpx.get(f"{base}/health", timeout=10.0)
    if health.status_code != 200:
        print(f"FAIL: backend not healthy ({health.status_code})")
        return 1

    with sample.open("rb") as f:
        r = httpx.post(
            f"{base}/reconstruct",
            files={"image": ("sample-object.png", f, "image/png")},
            timeout=900.0,
        )

    if r.status_code != 200:
        print(f"FAIL: {r.status_code} {r.text}")
        return 1

    data = r.json()
    required = [
        "reconstruction_id",
        "mesh_url",
        "source_image_url",
        "mesh_format",
        "file_size_bytes",
        "pipeline",
        "assistant_summary",
        "disclaimer",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        print(f"FAIL: missing fields: {missing}")
        return 1

    print("OK: smoke test passed")
    print(f"  reconstruction_id: {data['reconstruction_id']}")
    print(f"  mesh_url: {data['mesh_url']}")
    print(f"  pipeline: {data['pipeline']}")
    print(f"  size: {data['file_size_bytes']} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
