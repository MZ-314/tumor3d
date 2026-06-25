import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BACKEND))


def gpu_required() -> None:
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required — deploy on RunPod for real inference")


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CHAT_DB_PATH", str(tmp_path / "data" / "chat.db"))
    monkeypatch.setenv("SEGMENTATION_BACKEND", "monai")

    from fastapi.testclient import TestClient
    from api.main import app

    return TestClient(app)
