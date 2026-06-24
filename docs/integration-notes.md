# Integration notes

## Pipeline

`backend/medical_pipeline.py` — ingest slices → segment (stub/MONAI) → per-lesion mesh → Groq/template summary

## Response shape

`ReconstructResponse` includes `lesions[]` with `centroid_mm`, `bounding_box_3d_mm`, `volume_mm3` (with confidence + source), and `scene_mesh_url`.

## Dev without GPU

```bash
SEGMENTATION_BACKEND=stub uvicorn backend.api.main:app --reload
```
