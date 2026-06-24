"""Medical pipeline and API tests (CPU / stub backend)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BACKEND))

from shared.schemas.pydantic.common import AccuracyTier  # noqa: E402
from medical_pipeline import process_medical_slices  # noqa: E402


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CHAT_DB_PATH", str(tmp_path / "data" / "chat.db"))
    monkeypatch.setenv("SEGMENTATION_BACKEND", "stub")

    from fastapi.testclient import TestClient
    from api.main import app

    return TestClient(app)


def test_health(client) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["pipeline"] == "meddollina_3d"
    assert body["segmentation_backend"] == "stub"


def test_reconstruct_rejects_empty(client) -> None:
    r = client.post(
        "/reconstruct",
        files=[("images", ("empty.png", b"", "image/png"))],
    )
    assert r.status_code == 422


def test_reconstruct_single_slice(client) -> None:
    from PIL import Image
    import io

    img = Image.new("L", (128, 128), color=40)
    # bright blob for stub segmentation
    for y in range(50, 78):
        for x in range(50, 78):
            img.putpixel((x, y), 220)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    r = client.post(
        "/reconstruct",
        files=[("images", ("slice.png", buf.read(), "image/png"))],
        data={"modality": "brain_mri"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["slice_count"] == 1
    assert body["accuracy_tier"] == "single_slice"
    assert len(body["lesions"]) >= 1
    assert body["scene_mesh_url"].endswith("_scene.glb")
    assert body["viewer_mode"] == "volume"
    assert body["volume_nifti_url"] is not None


def test_reconstruct_multi_slice_async(client) -> None:
    from PIL import Image
    import io
    import time

    files = []
    for i in range(3):
        img = Image.new("L", (64, 64), color=30 + i * 5)
        for y in range(28, 36):
            for x in range(28, 36):
                img.putpixel((x, y), 200)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        files.append(("images", (f"slice{i}.png", buf.getvalue(), "image/png")))

    r = client.post("/reconstruct", files=files, data={"modality": "brain_mri"})
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]

    deadline = time.time() + 60
    body = None
    while time.time() < deadline:
        poll = client.get(f"/reconstruct/jobs/{job_id}")
        assert poll.status_code == 200
        data = poll.json()
        if data.get("status") == "processing":
            time.sleep(0.5)
            continue
        if data.get("status") == "error":
            pytest.fail(data.get("detail", "job failed"))
        body = data
        break

    assert body is not None
    assert body["slice_count"] == 3
    assert body["viewer_mode"] == "volume"


def test_reconstruct_knee_volume_only(client) -> None:
    from PIL import Image
    import io

    img = Image.new("L", (96, 96), color=50)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    r = client.post(
        "/reconstruct",
        files=[("images", ("knee.png", buf.read(), "image/png"))],
        data={"modality": "volume_mri"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["segmentation_backend"] == "volume_only"
    assert body["lesions"] == []
    assert "Volume-only" in body["disclaimer"]


@pytest.mark.asyncio
async def test_ai_3d_pipeline_direct(tmp_path: Path) -> None:
    from PIL import Image

    work = tmp_path / "ai1"
    work.mkdir()
    img_path = work / "photo.png"
    Image.new("RGB", (64, 64), color=(100, 150, 200)).save(img_path)

    from image_to_3d_pipeline import process_image_to_3d

    result = await process_image_to_3d(img_path, work)
    assert result.pipeline_type == "ai_3d"
    assert result.geometry_source == "ai_generated"
    assert result.viewer_mode == "mesh"
    assert (work / f"{work.name}_scene.glb").exists()


@pytest.mark.asyncio
async def test_medical_pipeline_direct(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEGMENTATION_BACKEND", "stub")
    from PIL import Image

    work = tmp_path / "job1"
    work.mkdir()
    img_path = work / "slice.png"
    img = Image.new("L", (96, 96), color=30)
    for y in range(40, 56):
        for x in range(40, 56):
            img.putpixel((x, y), 200)
    img.save(img_path)

    result = await process_medical_slices([img_path], work, modality="brain_mri")
    assert result.accuracy_tier == AccuracyTier.SINGLE_SLICE
    assert len(result.lesions) >= 1
    assert result.viewer_mode == "volume"
    assert result.volume_nifti_url is not None
    assert result.tumor_mask_nifti_url is not None
    assert (work / f"{work.name}_volume.nii.gz").exists()
    assert (work / f"{work.name}_tumor.nii.gz").exists()


def test_chat_crud(client) -> None:
    created = client.post("/chats", params={"title": "Test chat"})
    assert created.status_code == 200
    chat_id = created.json()["id"]

    listed = client.get("/chats")
    assert listed.status_code == 200
    assert any(c["id"] == chat_id for c in listed.json())

    detail = client.get(f"/chats/{chat_id}")
    assert detail.status_code == 200
    assert detail.json()["messages"] == []


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_monai_backend_gpu(tmp_path: Path) -> None:
    """Run on GPU pod with MONAI bundle. See docs/runpod-setup.md."""
    pytest.importorskip("monai")
    pytest.importorskip("torch")
