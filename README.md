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

`IMAGE3D_BACKEND=triposr` on RunPod GPU. No CPU fallback — TripoSR required for AI 3D mode.

Medical brain MRI: **MedSAM + MONAI + atlas** on RunPod (see setup below). No stub backends.

## Quick start — frontend

```bash
cd frontend
npm install
npm run dev
```

## RunPod (GPU)

```bash
cd /workspace/tumor3d
git pull origin main

# Option A — full bootstrap script (deps + models + ML train + API)
bash backend/scripts/runpod_start.sh

# Option B — manual steps
cd /workspace/tumor3d/backend
pip install -e ".[dev,gpu,dicom]"
pip install huggingface_hub pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install onnxruntime xatlas==0.0.9 moderngl SimpleITK

cd /workspace/tumor3d
python backend/scripts/setup_triposr.py
python backend/scripts/setup_monai_bundle.py
python backend/scripts/setup_medsam.py
python backend/scripts/setup_brain_atlas.py
python backend/scripts/setup_ml_brain_recon.py   # ML single-slice 3D (~5–15 min GPU)

export PYTHONPATH=/workspace/tumor3d:/workspace/tumor3d/backend
export DATA_DIR=/workspace/tumor3d/data
export SEGMENTATION_BACKEND=monai
export IMAGE3D_BACKEND=triposr
export TRIPOSR_DIR=/workspace/tumor3d/vendor/TripoSR
export ATLAS_BRAIN_DIR=/workspace/tumor3d/data/atlases/brain
export SYNTHESIS_BACKEND=ml
export ML_VOLUME_MODEL_DIR=/workspace/tumor3d/models/brain_recon

uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
```

`pip install` must run from `backend/` (installs `uvicorn`, `pydantic`, `torch`, etc.). Without it, `python` and `uvicorn` will not find project dependencies.

`GET /health` should show `"triposr_ready": true` for real AI 3D.

## Tests

```bash
PYTHONPATH=. pytest backend/tests -m "not gpu" -v
```
