"""Clone TripoSR and download weights for image-to-3D on RunPod."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND.parent
DEFAULT_DIR = REPO_ROOT / "vendor" / "TripoSR"
REPO_URL = "https://github.com/VAST-AI-Research/TripoSR.git"
INFER_REQ = Path(__file__).with_name("triposr_infer_requirements.txt")

# RunPod often sets HF_HUB_ENABLE_HF_TRANSFER=1 without installing hf_transfer.
_HF_ENV = {**os.environ, "HF_HUB_ENABLE_HF_TRANSFER": "0"}


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, check=True, env=env or os.environ)


def _restore_backend_deps() -> None:
    """TripoSR pins can downgrade numpy/Pillow/trimesh; restore our backend stack."""
    print("Restoring tumor3d-backend dependencies …")
    _run([sys.executable, "-m", "pip", "install", "-e", f"{BACKEND}[gpu,dicom]"])


def main() -> None:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    target.parent.mkdir(parents=True, exist_ok=True)

    if not (target / "run.py").is_file():
        print(f"Cloning TripoSR into {target} …")
        _run(["git", "clone", "--depth", "1", REPO_URL, str(target)])
    else:
        print(f"TripoSR already present at {target}")

    if INFER_REQ.is_file():
        print(f"Installing TripoSR inference deps from {INFER_REQ.name} …")
        _run([sys.executable, "-m", "pip", "install", "-r", str(INFER_REQ)])
    else:
        print("WARNING: triposr_infer_requirements.txt missing; skipping pip install")

    _restore_backend_deps()

    ckpt_dest = target / "model.ckpt"
    if not ckpt_dest.is_file():
        print("Downloading TripoSR weights from Hugging Face …")
        download_py = (
            "from huggingface_hub import hf_hub_download\n"
            "import shutil\n"
            "from pathlib import Path\n"
            "p = hf_hub_download('stabilityai/TripoSR', 'model.ckpt')\n"
            f"shutil.copy2(p, Path(r'{ckpt_dest}'))\n"
            "print('Saved', p)\n"
        )
        _run([sys.executable, "-c", download_py], env=_HF_ENV)
    else:
        print(f"Weights already at {ckpt_dest}")

    print(f"Done. Set: export TRIPOSR_DIR={target}")
    print("export IMAGE3D_BACKEND=triposr")


if __name__ == "__main__":
    main()
