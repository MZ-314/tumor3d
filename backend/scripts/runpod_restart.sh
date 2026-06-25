#!/usr/bin/env bash
# Daily RunPod restart: pull latest code + start API (deps/models already installed).
set -euo pipefail

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

echo "==> Health prerequisites"
if [[ ! -f "${ML_VOLUME_MODEL_DIR}/volume_generator.pt" ]]; then
  echo "ML checkpoint missing — run: python backend/scripts/setup_ml_brain_recon.py"
  exit 1
fi

echo "==> Starting API on :8000"
echo "    Health: curl -s http://localhost:8000/health | python -m json.tool"
exec uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
