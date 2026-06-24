"""Clone TripoSR and download weights for image-to-3D on RunPod."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND.parent
DEFAULT_DIR = REPO_ROOT / "vendor" / "TripoSR"
REPO_URL = "https://github.com/VAST-AI-Research/TripoSR.git"


def main() -> None:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    target.parent.mkdir(parents=True, exist_ok=True)

    if not (target / "run.py").is_file():
        print(f"Cloning TripoSR into {target} …")
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(target)],
            check=True,
        )
    else:
        print(f"TripoSR already present at {target}")

    req = target / "requirements.txt"
    if req.is_file():
        print("Installing TripoSR requirements …")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req)], check=True)

    ckpt_dest = target / "model.ckpt"
    if not ckpt_dest.is_file():
        download_py = (
            "from huggingface_hub import hf_hub_download\n"
            "import shutil\n"
            "from pathlib import Path\n"
            f"p = hf_hub_download('stabilityai/TripoSR', 'model.ckpt')\n"
            f"shutil.copy2(p, Path(r'{ckpt_dest}'))\n"
        )
        subprocess.run([sys.executable, "-c", download_py], check=True)

    print(f"Done. Set: export TRIPOSR_DIR={target}")
    print("export IMAGE3D_BACKEND=triposr")


if __name__ == "__main__":
    main()
