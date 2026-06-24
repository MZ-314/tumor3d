# Tumor3D — Meddollina 3D reconstruction plugin

**Core:** single 2D image → **AI 3D mesh** (TripoSR on RunPod) for SETV Global's Meddollina.ai chat.

**Also:** DICOM slice series → **real volume viewer** (NiiVue). **Optional:** brain MRI tumor segmentation (MONAI).

| Mode | Input | Output |
|------|--------|--------|
| **AI 3D** (default) | 1 photo | AI-inferred GLB mesh |
| **DICOM volume** | Many `.dcm` | Measured 3D volume (knee, brain, etc.) |
| **Brain + tumor** | Brain DICOM | Volume + MONAI tumor mask |

**Not a diagnostic device.** AI meshes are inferred geometry, not measured anatomy.

## Quick start — backend

```bash
cd backend
pip install -e ".[dev]"
cd ..
set PYTHONPATH=.   # Windows
uvicorn backend.api.main:app --reload --port 8000
```

`IMAGE3D_BACKEND=relief_stub` on CPU laptop (simple relief mesh). TripoSR on GPU — see RunPod below.

## Quick start — frontend

```bash
cd frontend
npm install
npm run dev
```

## RunPod (GPU)

```bash
cd /workspace/tumor3d/backend
pip install -e ".[dev,gpu,dicom]"
pip install huggingface_hub pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg

# AI 2D→3D
python scripts/setup_triposr.py
export IMAGE3D_BACKEND=triposr
export TRIPOSR_DIR=/workspace/tumor3d/vendor/TripoSR

# Optional brain tumor AI
python scripts/setup_monai_bundle.py
export SEGMENTATION_BACKEND=monai

export PYTHONPATH=/workspace/tumor3d:/workspace/tumor3d/backend
export DATA_DIR=/workspace/tumor3d/data
cd /workspace/tumor3d
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
```

`GET /health` should show `"triposr_ready": true` for real AI 3D.

## Tests

```bash
PYTHONPATH=. pytest backend/tests -m "not gpu" -v
```
