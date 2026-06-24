# RunPod first steps (start here)

You do **not** need SSH on day one. Use the **web terminal**.

## What you see in the RunPod UI

| Item | What to do |
|------|------------|
| **SSH key** | Skip for now (optional later) |
| **Web terminal** | Turn **Enable web terminal** ON → open it |
| **Port 8888 / Jupyter** | Ignore for this project (unless you prefer notebooks) |
| **Port 8000** | **Required** for our API — see below |

### Expose port 8000

If **Connect → HTTP services** only shows port 8888:

1. **Stop** the pod (or use Edit if available while running).
2. **Edit pod** → find **Expose HTTP ports** (or **Custom ports**).
3. Add **`8000`** (comma-separated with 8888 is fine: `8888,8000`).
4. **Start** the pod again.

After the API is running, you should see something like:

`https://occupational_olive_wasp-8000.proxy.runpod.net`

Open that URL + `/health` in your browser.

---

## On the pod (web terminal) — real tumor AI (MONAI)

**Do not use `stub` on RunPod for brain scans** — stub is fake. See [`runpod-monai-enable.md`](runpod-monai-enable.md).

```bash
git clone https://github.com/MZ-314/tumor3d.git /workspace/tumor3d
cd /workspace/tumor3d/backend
pip install -e ".[dev,gpu,dicom]"
python scripts/setup_monai_bundle.py

export SEGMENTATION_BACKEND=monai
export PYTHONPATH=/workspace/tumor3d:/workspace/tumor3d/backend
export DATA_DIR=/workspace/tumor3d/data

cd /workspace/tumor3d
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
```

Leave that terminal running. In a **second** web terminal tab:

```bash
curl http://127.0.0.1:8000/health
```

Must show `"segmentation_backend": "monai"`.

---

## On your Windows laptop

```powershell
cd c:\Projects\tumor3d\frontend
copy .env.example .env
```

Edit `frontend/.env` — set your real pod URL:

```env
VITE_API_BASE=https://occupational_olive_wasp-8000.proxy.runpod.net
```

(Replace with the exact URL from RunPod **Connect → HTTP services → Port 8000**.)

```powershell
npm install
npm run dev
```

Open http://localhost:5173 and upload a test slice.

---

## When you're done

**Stop the pod** in RunPod so you don't burn credits at $0.57/hr.

---

## Stub (laptop only — not for real MRI)

```bash
export SEGMENTATION_BACKEND=stub
```

Only for testing the chat UI without a GPU.

---
