"""Multi-stage reconstruction pipeline tests (GPU for brain MRI)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BACKEND))

from pipeline.reconstruct.runner import STAGES  # noqa: E402
from shared.schemas.pydantic.pipeline import PipelineArtifacts  # noqa: E402
from tests.conftest import gpu_required  # noqa: E402


def test_pipeline_stage_list_matches_architecture() -> None:
    names = [name for name, _ in STAGES]
    assert names == [
        "input_intelligence",
        "medical_analysis",
        "consensus",
        "atlas_matching",
        "blueprint",
        "synthesis",
        "mesh_generation",
        "validation",
    ]


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_reconstruction_pipeline_e2e(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gpu_required()
    monkeypatch.setenv("SEGMENTATION_BACKEND", "monai")
    monkeypatch.setenv("SYNTHESIS_BACKEND", "atlas")

    from PIL import Image
    from pipeline.reconstruct import run_reconstruction_pipeline

    work = tmp_path / "job_pipeline"
    work.mkdir()
    img_path = work / "slice.png"
    img = Image.new("L", (96, 96), color=30)
    for y in range(40, 56):
        for x in range(40, 56):
            img.putpixel((x, y), 200)
    img.save(img_path)

    stages_seen: list[str] = []
    result = await run_reconstruction_pipeline(
        [img_path],
        work,
        modality="brain_mri",
        on_stage=stages_seen.append,
    )

    assert result.pipeline_artifacts is not None
    assert len(result.pipeline_artifacts.stage_timings) == len(STAGES)
    assert stages_seen == [name for name, _ in STAGES]
    assert (work / "scan_context.json").exists()
    assert (work / "validation_report.json").exists()
    PipelineArtifacts.model_validate_json((work / "pipeline_artifacts.json").read_text(encoding="utf-8"))


def test_health_reports_pipeline_version(client) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["reconstruct_pipeline_version"] == "0.1.0"
