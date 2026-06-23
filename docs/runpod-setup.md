# RunPod setup — SAM 2 + TRELLIS.2 + Blender

Use this guide to run the full `reconstruction_3d.py` pipeline on a RunPod GPU instance.

## Pod requirements

| Resource | Minimum |
|---|---|
| GPU | NVIDIA 24GB VRAM (A5000, A6000, RTX 4090, A100) — TRELLIS.2-4B official minimum |
| OS | Linux (RunPod PyTorch 2.x template) |
| Disk | 50GB+ persistent volume |
| RAM | 32GB+ recommended |

> 16GB pods may OOM at 512³. Start with 24GB if possible.

## One-time setup (SSH into pod)

```bash
# 1. Clone this repo
git clone <your-repo-url> /workspace/tumor3d
cd /workspace/tumor3d

# 2. Install SAM 2
git clone https://github.com/facebookresearch/sam2.git /workspace/sam2
cd /workspace/sam2
pip install -e .
mkdir -p checkpoints
# Download sam2.1_hiera_large.pt from Meta SAM 2 releases into checkpoints/

# 3. Install TRELLIS.2
git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive /workspace/TRELLIS.2
cd /workspace/TRELLIS.2
. ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
conda activate trellis2

# 4. Install Blender
apt-get update && apt-get install -y blender

# 5. Install API dependencies
cd /workspace/tumor3d/backend
pip install -e ".[dev]"
pip install -r requirements-gpu.txt
```

## Environment variables

```bash
export SAM2_CHECKPOINT=/workspace/sam2/checkpoints/sam2.1_hiera_large.pt
export SAM2_CONFIG=configs/sam2.1/sam2.1_hiera_l.yaml
export SAM2_DEVICE=cuda
export TRELLIS2_MODEL_ID=microsoft/TRELLIS.2-4B
export TRELLIS2_DEVICE=cuda
export BLENDER_BIN=blender
export PYTHONPATH=/workspace/tumor3d:/workspace/tumor3d/backend
```

## Start API

```bash
cd /workspace/tumor3d
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
```

## Smoke test

```bash
# From another terminal
python backend/scripts/smoke_test.py http://127.0.0.1:8000
```

## GPU end-to-end test

```bash
pytest backend/tests/test_reconstruction_3d.py -m gpu -v
```

## RunPod free trial tips

- Use a **24GB** template if available (PyTorch 2.x + CUDA 12.x).
- First TRELLIS.2 inference downloads ~10GB weights from Hugging Face — allow time.
- Set `RECON_STAGE2_TIMEOUT_S=900` for first run (model download + compile).
- Expose port 8000 in RunPod TCP settings to hit the API from your laptop.

## Laptop (Windows)

Your local machine can edit code and run **CPU-only tests** (`pytest -m "not gpu"`).
Full pipeline inference **must** run on the GPU pod.
