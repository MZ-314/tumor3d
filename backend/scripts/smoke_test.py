#!/usr/bin/env python3
"""POST a test slice to /reconstruct and print lesion summary."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import httpx
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def make_test_slice() -> bytes:
    img = Image.new("L", (128, 128), color=30)
    for y in range(48, 80):
        for x in range(48, 80):
            img.putpixel((x, y), 210)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    data = make_test_slice()
    with httpx.Client(timeout=120.0) as client:
        health = client.get(f"{base}/health")
        print("health:", health.json())
        r = client.post(
            f"{base}/reconstruct",
            files=[("images", ("slice.png", data, "image/png"))],
            data={"modality": "brain_mri"},
        )
        r.raise_for_status()
        body = r.json()
        print("reconstruction_id:", body["reconstruction_id"])
        print("lesions:", len(body["lesions"]))
        for i, lesion in enumerate(body["lesions"], 1):
            c = lesion["centroid_mm"]
            print(f"  lesion {i}: ({c['x']:.1f}, {c['y']:.1f}, {c['z']:.1f}) mm")


if __name__ == "__main__":
    main()
