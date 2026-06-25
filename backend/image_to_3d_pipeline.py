"""Single-image AI → 3D mesh via TripoSR on GPU."""

from __future__ import annotations

import asyncio
from pathlib import Path

from config_reconstruction import IMAGE3D_BACKEND, Image3DError
from pipeline.image_to_3d.image_preflight import validate_ai_3d_input
from pipeline.image_to_3d.triposr_infer import run_triposr, triposr_available
from services.groq_assistant import build_assistant_summary
from shared.schemas.pydantic.common import AccuracyTier
from shared.schemas.pydantic.reconstruct import ReconstructResponse


async def process_image_to_3d(
    image_path: Path,
    work_dir: Path,
    *,
    chat_id: str | None = None,
    user_text: str | None = None,
    backend: str | None = None,
) -> ReconstructResponse:
    reconstruction_id = work_dir.name
    backend = (backend or IMAGE3D_BACKEND).lower()

    if backend != "triposr":
        raise Image3DError(
            f"IMAGE3D_BACKEND={backend} is not supported. Use IMAGE3D_BACKEND=triposr on RunPod GPU."
        )
    if not triposr_available():
        raise Image3DError(
            "TripoSR is not installed. On RunPod run:\n"
            "  python backend/scripts/setup_triposr.py\n"
            "  export IMAGE3D_BACKEND=triposr"
        )

    source_path = work_dir / "source.png"
    if not source_path.exists():
        from PIL import Image

        Image.open(image_path).convert("RGB").save(source_path)

    validate_ai_3d_input(image_path)

    scene_path = work_dir / f"{reconstruction_id}_scene.glb"

    def _build_mesh() -> None:
        run_triposr(image_path, work_dir, scene_path)

    await asyncio.to_thread(_build_mesh)

    base = "/static"
    disclaimer = (
        "AI-GENERATED 3D: this mesh was inferred from a single 2D image, not measured from a scan. "
        "Shape and hidden surfaces are model guesses — not anatomy. "
        "For real medical volumes, upload a DICOM series instead. Not for clinical use."
    )

    response = ReconstructResponse(
        reconstruction_id=reconstruction_id,
        chat_id=chat_id,
        source_image_url=f"{base}/{reconstruction_id}/source.png",
        overlay_image_url=None,
        scene_mesh_url=f"{base}/{reconstruction_id}/{scene_path.name}",
        volume_nifti_url=None,
        tumor_mask_nifti_url=None,
        viewer_mode="mesh",
        mesh_format="glb",
        slice_count=1,
        accuracy_tier=AccuracyTier.SINGLE_SLICE,
        modality="ai_3d",
        pipeline_type="ai_3d",
        geometry_source="ai_generated",
        segmentation_backend="triposr",
        lesions=[],
        assistant_summary="",
        disclaimer=disclaimer,
    )
    response.assistant_summary = await build_assistant_summary(response, user_text=user_text)
    return response
