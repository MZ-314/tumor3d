# Integration notes

## Pipeline (v0.2)

`backend/reconstruction_3d.py` — SAM 2 → TRELLIS.2 → Blender → GLB

Legacy ellipsoid/tumor stub pipeline removed.

## API contract

`ReconstructResponse` fields: `mesh_url`, `source_image_url`, `file_size_bytes`, `pipeline`, `assistant_summary`, `disclaimer`.

## Deployment

- Inference: RunPod / Linux GPU — see `docs/runpod-setup.md`
- Laptop: frontend dev + CPU tests only

## Meddollina embed

Use `frontend/plugin-shell` `mountTumorViewer()` — unchanged integration pattern, new backend response shape.
