from __future__ import annotations

import logging
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.pipeline_routing import (
    PIPELINE_MEDICAL_TUMOR,
    PIPELINE_MEDICAL_VOLUME,
    is_dicom_path,
    resolve_pipeline,
)
from api.reconstruct_jobs import get_job, should_run_async, start_job
from config_medical import DATA_DIR, MedicalPipelineError, SEGMENTATION_BACKEND, ensure_data_dirs
from config_pipeline import PIPELINE_VERSION
from db.database import add_message, create_chat, get_chat, init_db, list_chats, touch_chat
from db.jobs import init_jobs_db
from image_to_3d_pipeline import process_image_to_3d
from config_reconstruction import IMAGE3D_BACKEND, TRIPOSR_DIR
from pipeline.reconstruct import run_reconstruction_pipeline
from pipeline.image_to_3d.triposr_infer import triposr_available
from shared.schemas.pydantic.reconstruct import ChatDetail, ChatSummary, ReconstructResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

ensure_data_dirs()
init_db()
init_jobs_db()


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_data_dirs()
    init_db()
    init_jobs_db()
    yield


app = FastAPI(title="Meddollina 3D Reconstruction API", version="0.4.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(DATA_DIR)), name="static")


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "pipeline": "meddollina_3d",
        "reconstruct_pipeline_version": PIPELINE_VERSION,
        "image3d_backend": IMAGE3D_BACKEND,
        "triposr_ready": triposr_available(),
        "segmentation_backend": SEGMENTATION_BACKEND,
    }


@app.get("/chats", response_model=list[ChatSummary])
def chats_list() -> list[ChatSummary]:
    return list_chats()


@app.post("/chats", response_model=ChatSummary)
def chats_create(title: str = "New scan") -> ChatSummary:
    return create_chat(title=title)


@app.get("/chats/{chat_id}", response_model=ChatDetail)
def chats_get(chat_id: str) -> ChatDetail:
    detail = get_chat(chat_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Chat not found")
    return detail



async def _read_uploads(images: list[UploadFile], work_dir: Path) -> list[Path]:
    slice_paths: list[Path] = []
    for i, upload in enumerate(images):
        data = await upload.read()
        if not data:
            raise HTTPException(status_code=422, detail=f"Uploaded file {i} is empty")
        name = upload.filename or f"slice_{i}.png"
        path = work_dir / f"slice_{i:03d}_{name}"
        path.write_bytes(data)
        slice_paths.append(path)
    return slice_paths


def _attach_chat_messages(
    *,
    chat_id: str,
    images: list[UploadFile],
    modality: str,
    text: str | None,
    result: ReconstructResponse,
) -> None:
    detail = get_chat(chat_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Chat not found")

    first_name = images[0].filename or "scan"
    title = f"{modality}: {first_name}"[:80]
    if detail.title == "New scan":
        touch_chat(chat_id, title=title)

    add_message(
        chat_id,
        "user",
        text=text or f"Uploaded {len(images)} slice(s)",
        attachment_url=result.source_image_url,
    )
    add_message(
        chat_id,
        "assistant",
        text=result.assistant_summary,
        reconstruction=result,
    )
    result.chat_id = chat_id


@app.post("/reconstruct")
async def reconstruct(
    images: list[UploadFile] = File(...),
    modality: str = Form("ai_3d"),
    chat_id: str | None = Form(None),
    text: str | None = Form(None),
):
    if not images:
        raise HTTPException(status_code=422, detail="At least one image is required")

    if chat_id and not get_chat(chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")

    reconstruction_id = uuid.uuid4().hex[:12]
    work_dir = DATA_DIR / reconstruction_id
    work_dir.mkdir(parents=True, exist_ok=True)
    slice_paths = await _read_uploads(images, work_dir)
    pipeline = resolve_pipeline(modality, slice_paths)

    # DICOM in AI 3D mode is a common mistake — route to the medical pipeline.
    if pipeline == "ai_3d" and slice_paths and is_dicom_path(slice_paths[0]):
        if len(slice_paths) == 1:
            pipeline = PIPELINE_MEDICAL_TUMOR
            modality = "brain_mri"
        else:
            pipeline = PIPELINE_MEDICAL_VOLUME
            modality = "volume_mri"

    if pipeline == "ai_3d" and len(slice_paths) != 1:
        raise HTTPException(
            status_code=422,
            detail="AI 3D mode accepts exactly one image. Use DICOM volume mode for multiple slices.",
        )

    upload_label = (
        f"Uploaded image for AI 3D"
        if pipeline == "ai_3d"
        else f"Uploaded {len(images)} slice(s)"
    )

    if should_run_async(pipeline, len(images)):
        start_job(
            reconstruction_id,
            pipeline=pipeline,
            slice_paths=slice_paths,
            work_dir=work_dir,
            modality=modality,
            chat_id=chat_id,
            user_text=text,
            upload_label=upload_label,
            first_filename=images[0].filename or "scan",
        )
        return JSONResponse(
            status_code=202,
            content={
                "status": "processing",
                "job_id": reconstruction_id,
                "slice_count": len(images),
                "pipeline": pipeline,
            },
        )

    try:
        if pipeline == "ai_3d":
            result = await process_image_to_3d(
                slice_paths[0],
                work_dir,
                chat_id=chat_id,
                user_text=text,
            )
        else:
            result = await run_reconstruction_pipeline(
                slice_paths,
                work_dir,
                modality=modality,
                chat_id=chat_id,
                user_text=text,
            )
    except MedicalPipelineError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        from config_reconstruction import Image3DError

        if isinstance(exc, Image3DError):
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        raise

    if chat_id:
        _attach_chat_messages(
            chat_id=chat_id,
            images=images,
            modality=modality,
            text=text,
            result=result,
        )

    return result


@app.get("/reconstruct/jobs/{job_id}")
async def reconstruct_job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status == "processing":
        payload: dict[str, object] = {
            "status": "processing",
            "job_id": job_id,
            "slice_count": job.slice_count,
        }
        if job.stage:
            payload["stage"] = job.stage
        return payload
    if job.status == "error":
        return {"status": "error", "job_id": job_id, "detail": job.error or "Processing failed"}
    return job.result


@app.post("/chats/{chat_id}/messages")
async def chat_send_message(
    chat_id: str,
    images: list[UploadFile] = File(...),
    text: str | None = Form(None),
    modality: str = Form("ai_3d"),
):
    if not get_chat(chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")
    return await reconstruct(
        images=images,
        modality=modality,
        chat_id=chat_id,
        text=text,
    )


@app.get("/meshes/{reconstruction_id}/{filename}")
def get_mesh_file(reconstruction_id: str, filename: str) -> FileResponse:
    path = DATA_DIR / reconstruction_id / filename
    if not path.exists():
        path = DATA_DIR / "outputs" / reconstruction_id / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Mesh not found")
    return FileResponse(path, media_type="model/gltf-binary")


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
