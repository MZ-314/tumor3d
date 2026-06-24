# RunPod setup — MONAI medical segmentation

**New pod?** Start with [`runpod-first-steps.md`](runpod-first-steps.md) (web terminal, port 8000, laptop `.env`).

Use this guide to run **real** brain tumor segmentation on a RunPod GPU. Laptops and Macs should use `SEGMENTATION_BACKEND=stub` for UI development only.

## Pod requirements

| Resource | Minimum |
|---|---|
| GPU | NVIDIA 16GB+ VRAM (24GB A6000 recommended) |
| OS | Linux (RunPod PyTorch 2.x template) |
| Disk | 30GB+ persistent volume |
| RAM | 16GB+ |

## One-time setup (SSH into pod)

```bash
git clone https://github.com/MZ-314/tumor3d.git /workspace/tumor3d
cd /workspace/tumor3d/backend

pip install -e ".[dev,gpu,dicom]"
```

### MONAI bundle

Download or configure a BraTS-style segmentation bundle (example paths vary by bundle):

```bash
# Example — follow your bundle's README for exact download commands
mkdir -p /workspace/monai_bundle
# monai bundle download ... into /workspace/monai_bundle
```

Set `MONAI_BUNDLE_DIR` to the directory containing the bundle `configs/` and weights.

## Environment variables

```bash
export SEGMENTATION_BACKEND=monai
export MONAI_BUNDLE_DIR=/workspace/monai_bundle
export DATA_DIR=/workspace/tumor3d/data
export PYTHONPATH=/workspace/tumor3d:/workspace/tumor3d/backend

# Optional narration
export GROQ_API_KEY=gsk_...
```

## Start API

```bash
cd /workspace/tumor3d
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
```

Expose port **8000** in RunPod TCP settings and point your frontend `vite.config` proxy or `apiBase` at the pod URL.

## Smoke test

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok","pipeline":"medical_segmentation","segmentation_backend":"monai"}
```

Upload a slice:

```bash
curl -F "images=@/path/to/slice.png" -F "modality=brain_mri" \
  http://127.0.0.1:8000/reconstruct
```

## GPU test

```bash
pytest backend/tests/test_medical_pipeline.py -m gpu -v
```

## Laptop / Mac (no GPU)

```bash
export SEGMENTATION_BACKEND=stub
pip install -e backend/.[dev]
PYTHONPATH=. pytest backend/tests -m "not gpu" -v
cd frontend && npm run dev
```

Stub segmentation uses bright-region heuristics — good for **UI and API** testing, not clinical accuracy.

## Accuracy tiers

| Slices uploaded | `accuracy_tier` | Z / volume |
|---|---|---|
| 1 | `single_slice` | Depth extruded / estimated |
| 2–9 | `partial_volume` | Improving |
| 10+ | `multi_slice` | Best available in this prototype |
