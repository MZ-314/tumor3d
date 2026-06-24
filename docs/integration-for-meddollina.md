# Meddollina integration guide

## Stack

- **MONAI** (GPU) or **stub** (CPU dev) — tumor segmentation on brain MRI/CT slices
- **Marching cubes / extrusion** — per-lesion 3D meshes (GLB)
- **SQLite** — saved chat history
- **Groq** (optional) — assistant narration

## Embed API

```ts
import { mountTumorViewer, unmountTumorViewer } from "./dist/plugin-shell/plugin-shell.es.js";

await mountTumorViewer(messageBubble, {
  imageUrl: scanUrl,
  apiBase: "https://your-gpu-api.meddollina.ai",
  onComplete: (result) => console.log(result.assistant_summary),
});
```

## Backend

Deploy `backend/api/main.py`. GPU pod for MONAI; stub for UI-only dev. See [`runpod-setup.md`](runpod-setup.md).

`POST /reconstruct` — multipart `images[]` → JSON with `lesions[]`, `scene_mesh_url`, coordinates

`GET/POST /chats` — persisted chat sessions

## Standalone module

```python
from medical_pipeline import process_medical_slices
from pathlib import Path

result = await process_medical_slices([Path("slice.png")], Path("out/job1"))
```

## Limitations

- Research prototype — not for clinical diagnosis
- Single slice: depth (Z) and volume are estimated; upload more axial slices for better accuracy
- CT-specific models: MRI path first; CT routing planned
