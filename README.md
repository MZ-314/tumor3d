# Tumor3D — Medical Imaging Chat (Meddollina prototype)

Brain MRI/CT slice upload → **tumor segmentation** → **3D lesion meshes** with coordinates, saved chat history, and optional **Groq** narration.

| Layer | Technology |
|---|---|
| Segmentation (GPU) | **MONAI** BraTS-style bundle on RunPod |
| Segmentation (dev) | **stub** heuristic (CPU — laptop / Mac M1) |
| 3D mesh | Mask extrusion / marching cubes per lesion |
| Chat | FastAPI + SQLite |
| Viewer | React + Three.js |

**Not a diagnostic device.** Single-slice depth (Z) and volume are estimated; accuracy improves with more slices.

## Quick start — backend (any machine)

```bash
cd backend
pip install -e ".[dev]"
cd ..
set PYTHONPATH=.   # Windows
# export PYTHONPATH=.  # Mac/Linux
uvicorn backend.api.main:app --reload --port 8000
```

Set `SEGMENTATION_BACKEND=stub` (default) for CPU demo.

## Quick start — frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 (Vite proxies API to port 8000).

## GPU inference (RunPod)

See [`docs/runpod-setup.md`](docs/runpod-setup.md) for MONAI bundle setup.

```bash
export SEGMENTATION_BACKEND=monai
export MONAI_BUNDLE_DIR=/workspace/monai_bundle
export PYTHONPATH=/workspace/tumor3d:/workspace/tumor3d/backend
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
```

## Environment

Copy [`.env.example`](.env.example). Optional `GROQ_API_KEY` enables LLM summaries; without it, a structured template is used.

## API

| Endpoint | Description |
|---|---|
| `GET /health` | Liveness + backend mode |
| `POST /reconstruct` | Multipart `images[]`, optional `chat_id`, `modality`, `text` |
| `GET/POST /chats` | List / create chats |
| `GET /chats/{id}` | Chat with messages + reconstructions |
| `POST /chats/{id}/messages` | Send slices in a chat |

Response includes `lesions[]` with `centroid_mm`, `bounding_box_3d_mm`, `volume_mm3`, and `scene_mesh_url`.

## Tests

```bash
PYTHONPATH=. pytest backend/tests -m "not gpu" -v
```

GPU (RunPod with MONAI):

```bash
pytest backend/tests -m gpu -v
```

## Project structure

```
backend/
  medical_pipeline.py       # Orchestrator
  config_medical.py
  pipeline/ingest/          # PNG / DICOM loading
  pipeline/segment/         # stub + MONAI backends
  pipeline/mesh/            # Per-lesion GLB export
  db/database.py            # SQLite chat history
  services/groq_assistant.py
  api/main.py
frontend/
  chat/                     # Medical chat UI + sidebar
  viewer/                   # 3D lesion viewer
  plugin-shell/             # mountTumorViewer() embed
docs/
  runpod-setup.md
```

## Embed

```ts
import { mountTumorViewer } from "./plugin-shell";
await mountTumorViewer(container, { imageUrl: "/path/to/slice.png", apiBase: "http://localhost:8000" });
```
