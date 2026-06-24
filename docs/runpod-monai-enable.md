# RunPod — copy/paste after git clone

## 1. Install + download MONAI BraTS model (~1–3 GB first time)

```bash
cd /workspace/tumor3d
git pull origin main
cd backend
pip install -e ".[dev,gpu,dicom]"
pip install huggingface_hub
python scripts/setup_monai_bundle.py
```

## 2. Start API with MONAI (not stub)

```bash
export SEGMENTATION_BACKEND=monai
export PYTHONPATH=/workspace/tumor3d:/workspace/tumor3d/backend
export DATA_DIR=/workspace/tumor3d/data
cd /workspace/tumor3d
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
```

## 3. Verify

```bash
curl http://127.0.0.1:8000/health
```

Must show: `"segmentation_backend": "monai"`  
If it says `"stub"` you forgot `export SEGMENTATION_BACKEND=monai`.

## Notes

- **Stub is not tumor AI** — only for UI tests without GPU.
- BraTS model expects brain MRI; best with **DICOM** or several axial slices.
- Single JPEG works as a fallback (one channel repeated 4×) but quality is limited.
