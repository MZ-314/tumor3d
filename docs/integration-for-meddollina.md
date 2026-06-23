# Meddollina integration guide

## Stack

- **SAM 2** — foreground cutout
- **TRELLIS.2-4B** — image-to-3D with PBR textures
- **Blender** — mesh cleanup and GLB export

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

Deploy `backend/api/main.py` on a GPU server. See [`runpod-setup.md`](runpod-setup.md).

`POST /reconstruct` — multipart image → JSON with `mesh_url` pointing to `.glb`

## Standalone module

For non-HTTP integration:

```python
from reconstruction_3d import process_image_to_3d
path = await process_image_to_3d(input_path, output_dir)
```

## Limitations

- Requires 24GB+ NVIDIA GPU for reliable inference
- Single-image input; unseen angles are AI-inferred
- Best results: one clear object, plain background
