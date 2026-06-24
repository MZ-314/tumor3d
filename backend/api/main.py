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
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config_medical import DATA_DIR, MedicalPipelineError, SEGMENTATION_BACKEND, ensure_data_dirs
from db.database import add_message, create_chat, get_chat, init_db, list_chats, touch_chat
from medical_pipeline import process_medical_slices
from shared.schemas.pydantic.reconstruct import ChatDetail, ChatSummary, ReconstructResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

ensure_data_dirs()
init_db()


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_data_dirs()
    init_db()
    yield


app = FastAPI(title="Meddollina Medical Chat API", version="0.3.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(DATA_DIR)), name="static")


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "pipeline": "medical_segmentation",
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


@app.post("/reconstruct", response_model=ReconstructResponse)
async def reconstruct(
    images: list[UploadFile] = File(...),
    modality: str = Form("brain_mri"),
    chat_id: str | None = Form(None),
    text: str | None = Form(None),
) -> ReconstructResponse:
    if not images:
        raise HTTPException(status_code=422, detail="At least one image is required")

    reconstruction_id = uuid.uuid4().hex[:12]
    work_dir = DATA_DIR / reconstruction_id
    work_dir.mkdir(parents=True, exist_ok=True)

    slice_paths: list[Path] = []
    for i, upload in enumerate(images):
        data = await upload.read()
        if not data:
            raise HTTPException(status_code=422, detail=f"Uploaded file {i} is empty")
        name = upload.filename or f"slice_{i}.png"
        path = work_dir / f"slice_{i:03d}_{name}"
        path.write_bytes(data)
        slice_paths.append(path)

    try:
        result = await process_medical_slices(
            slice_paths,
            work_dir,
            modality=modality,
            chat_id=chat_id,
            user_text=text,
        )
    except MedicalPipelineError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if chat_id:
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

    return result


@app.post("/chats/{chat_id}/messages", response_model=ReconstructResponse)
async def chat_send_message(
    chat_id: str,
    images: list[UploadFile] = File(...),
    text: str | None = Form(None),
    modality: str = Form("brain_mri"),
) -> ReconstructResponse:
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
