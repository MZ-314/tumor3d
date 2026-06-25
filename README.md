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

### Every restart (pod already set up)

```bash
bash /workspace/tumor3d/backend/scripts/runpod_restart.sh
```

Or copy-paste:

```bash
cd /workspace/tumor3d
git pull origin main

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

Verify (second terminal):

```bash
curl -s http://localhost:8000/health | python -m json.tool
```

Expect: `triposr_ready: true`, `synthesis_backend: "ml"`, `ml_volume_generator_ready: true`, `ml_volume_refiner_ready: true`.

### First-time / fresh pod (deps + models + ML train)

```bash
cd /workspace/tumor3d
git pull origin main
bash backend/scripts/runpod_start.sh
```

Or manual:

```bash
cd /workspace/tumor3d
git pull origin main

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
python backend/scripts/setup_ml_brain_recon.py

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

`pip install` must run from `backend/` on a fresh pod. `runpod_restart.sh` skips install when deps are already present.

`GET /health` should show `"triposr_ready": true` for real AI 3D.

## Tests

```bash
PYTHONPATH=. pytest backend/tests -m "not gpu" -v
```
