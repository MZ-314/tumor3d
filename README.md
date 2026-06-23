# Tumor3D — Image to 3D Reconstruction

High-fidelity **single image → 3D model** pipeline for Meddollina integration:

| Stage | Technology |
|---|---|
| 1. Foreground isolation | **Meta SAM 2** |
| 2. 3D generation | **Microsoft TRELLIS.2** (4B) |
| 3. Mesh cleanup | **Blender Python API** (headless) |

Delivered as a chat demo + embeddable viewer widget + standalone `reconstruction_3d.py` module.

## Requirements

- **Inference:** Linux GPU server with **24GB+ VRAM** (RunPod recommended)
- **Dev (laptop):** Python 3.11+, Node 18+ for frontend only

## Quick start — GPU server (RunPod)

See [`docs/runpod-setup.md`](docs/runpod-setup.md) for full install.

```bash
export PYTHONPATH=/workspace/tumor3d:/workspace/tumor3d/backend
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
```

## Quick start — frontend (laptop)

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 (proxy to backend on port 8000).

## Core module

```python
from reconstruction_3d import process_image_to_3d

glb_path = await process_image_to_3d("/path/to/image.png", "/path/to/output_dir")
```

No web framework coupling — disk path in, absolute GLB path out.

## Tests

```bash
cd backend && pip install -e ".[dev]"
cd .. && PYTHONPATH=. pytest backend/tests -m "not gpu" -v
```

GPU end-to-end (on RunPod only):

```bash
pytest backend/tests -m gpu -v
```

## Project structure

```
backend/
  reconstruction_3d.py    # Main pipeline orchestrator
  config_reconstruction.py
  blender_export.py       # Headless Blender cleanup script
  api/main.py             # FastAPI wrapper
frontend/
  chat/                   # Chat demo UI
  viewer/                 # Reusable 3D viewer
  plugin-shell/           # mountTumorViewer() embed API
docs/
  runpod-setup.md
```

## API

`POST /reconstruct` — multipart image upload → `ReconstructResponse` with `mesh_url` (.glb)

`GET /health` — liveness check

## Honesty note

3D models are **AI-generated** from a single photo. Sides not visible in the image are inferred by TRELLIS.2, not measured.
