from __future__ import annotations

import logging
import shutil
import sys
import uuid
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config_reconstruction import (
    Reconstruction3DError,
    Stage1IsolationError,
    Stage2ReconstructionError,
    Stage3ExportError,
)
from reconstruction_3d import process_image_to_3d
from shared.schemas.pydantic.reconstruct import ReconstructResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

DATA_DIR = _BACKEND / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Tumor3D Reconstruction API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(DATA_DIR)), name="static")


def _build_assistant_summary(reconstruction_id: str, file_size: int) -> str:
    size_mb = file_size / (1024 * 1024)
    return (
        f"I've generated a 3D model from your image (job {reconstruction_id}). "
        f"The mesh is {size_mb:.1f} MB — drag to rotate and scroll to zoom. "
        f"Note: sides not visible in the photo are AI-inferred."
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "pipeline": "sam2_trellis2_blender"}


@app.post("/reconstruct", response_model=ReconstructResponse)
async def reconstruct(image: UploadFile = File(...)) -> ReconstructResponse:
    filename = image.filename or "upload.png"
    data = await image.read()
    if not data:
        raise HTTPException(status_code=422, detail="Uploaded file is empty")

    reconstruction_id = uuid.uuid4().hex[:12]
    work_dir = DATA_DIR / reconstruction_id
    work_dir.mkdir(parents=True, exist_ok=True)

    input_path = work_dir / f"input_{filename}"
    input_path.write_bytes(data)

    source_path = work_dir / "source.png"
    shutil.copy(input_path, source_path)

    try:
        glb_path = await process_image_to_3d(str(input_path), str(work_dir))
    except Stage1IsolationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Stage2ReconstructionError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Stage3ExportError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Reconstruction3DError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    glb_file = Path(glb_path)
    if not glb_file.is_file():
        raise HTTPException(status_code=500, detail="Reconstruction produced no output file")

    static_glb = work_dir / "model.glb"
    if glb_file.resolve() != static_glb.resolve():
        shutil.copy(glb_file, static_glb)

    file_size = static_glb.stat().st_size
    base = "/static"

    return ReconstructResponse(
        reconstruction_id=reconstruction_id,
        mesh_url=f"{base}/{reconstruction_id}/model.glb",
        source_image_url=f"{base}/{reconstruction_id}/source.png",
        isolated_image_url=None,
        mesh_format="glb",
        file_size_bytes=file_size,
        pipeline="sam2_trellis2_blender",
        assistant_summary=_build_assistant_summary(reconstruction_id, file_size),
    )


@app.get("/meshes/{reconstruction_id}.glb")
def get_mesh(reconstruction_id: str) -> FileResponse:
    path = DATA_DIR / reconstruction_id / "model.glb"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Mesh not found")
    return FileResponse(path, media_type="model/gltf-binary")


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
