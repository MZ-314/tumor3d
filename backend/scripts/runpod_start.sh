#!/usr/bin/env bash
# One-shot RunPod bootstrap: deps, models, modular atlas, ML completion, uvicorn.
set -euo pipefail

cd /workspace/tumor3d

export PYTHONPATH=/workspace/tumor3d:/workspace/tumor3d/backend
export DATA_DIR=/workspace/tumor3d/data
export SEGMENTATION_BACKEND=monai
export IMAGE3D_BACKEND=triposr
export TRIPOSR_DIR=/workspace/tumor3d/vendor/TripoSR
export ATLAS_BRAIN_DIR=/workspace/tumor3d/data/atlases/brain
export MODULAR_BRAIN_DIR=/workspace/tumor3d/data/atlases/brain/modules
export MODULAR_RECON=1
export SYNTHESIS_BACKEND=ml
export ML_VOLUME_MODEL_DIR=/workspace/tumor3d/models/brain_recon

echo "==> Installing Python dependencies..."
cd /workspace/tumor3d/backend
pip install -e ".[dev,gpu,dicom]"
pip install huggingface_hub pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install onnxruntime xatlas==0.0.9 moderngl SimpleITK trimesh scikit-image nibabel

cd /workspace/tumor3d
echo "==> Model assets..."
python backend/scripts/setup_triposr.py
python backend/scripts/setup_monai_bundle.py
python backend/scripts/setup_medsam.py
python backend/scripts/setup_brain_atlas.py
python backend/scripts/setup_brain_modules.py

echo "==> ML brain volume generator (bootstrap train)..."
python backend/scripts/setup_ml_brain_recon.py

echo "==> Modular volume completion (3D refiner)..."
python backend/scripts/train_volume_completion.py

echo "==> Starting API on :8000"
exec uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
