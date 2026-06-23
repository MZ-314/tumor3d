import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BACKEND))

from config_reconstruction import (  # noqa: E402
    ReconstructionConfig,
    Stage1IsolationError,
)
from reconstruction_3d import process_image_to_3d  # noqa: E402


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from api.main import app

    return TestClient(app)


def test_health(client) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["pipeline"] == "sam2_trellis2_blender"


def test_reconstruct_rejects_empty(client) -> None:
    r = client.post(
        "/reconstruct",
        files={"image": ("empty.png", b"", "image/png")},
    )
    assert r.status_code == 422


def test_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRELLIS2_SEED", "99")
    monkeypatch.setenv("BLENDER_MAX_TRIANGLES", "50000")
    cfg = ReconstructionConfig.from_env()
    assert cfg.trellis2_seed == 99
    assert cfg.blender_max_triangles == 50000


@pytest.mark.asyncio
async def test_missing_input_raises(tmp_path: Path) -> None:
    with pytest.raises(Stage1IsolationError, match="not found"):
        await process_image_to_3d(
            str(tmp_path / "nonexistent.png"),
            str(tmp_path / "out"),
        )


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_full_pipeline_e2e(tmp_path: Path) -> None:
    """Run on GPU pod with SAM2 + TRELLIS.2 installed. See docs/runpod-setup.md."""
    from PIL import Image, ImageDraw

    img_path = tmp_path / "test_object.png"
    img = Image.new("RGB", (512, 512), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    draw.ellipse([156, 156, 356, 356], fill=(200, 80, 60))
    img.save(img_path)

    out_dir = tmp_path / "output"
    result = await process_image_to_3d(str(img_path), str(out_dir))
    assert Path(result).is_file()
    assert Path(result).suffix == ".glb"
    assert Path(result).stat().st_size > 0
