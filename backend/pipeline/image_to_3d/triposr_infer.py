"""Run TripoSR single-image-to-3D on GPU."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from config_reconstruction import TRIPOSR_DIR, TRIPOSR_MC_RESOLUTION, Image3DError

logger = logging.getLogger(__name__)


def triposr_available() -> bool:
    run_py = TRIPOSR_DIR / "run.py"
    ckpt = TRIPOSR_DIR / "model.ckpt"
    return run_py.is_file() and (ckpt.is_file() or (TRIPOSR_DIR / "models").is_dir())


def run_triposr(image_path: Path, work_dir: Path, out_glb: Path) -> Path:
    """Invoke TripoSR run.py and copy the resulting GLB."""
    run_py = TRIPOSR_DIR / "run.py"
    if not run_py.is_file():
        raise Image3DError(
            f"TripoSR not installed at {TRIPOSR_DIR}.\n"
            "On RunPod: python backend/scripts/setup_triposr.py"
        )

    output_dir = work_dir / "triposr_out"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    cmd = [
        sys.executable,
        str(run_py),
        str(image_path),
        "--output-dir",
        str(output_dir),
        "--model-save-format",
        "glb",
        "--mc-resolution",
        str(TRIPOSR_MC_RESOLUTION),
    ]
    logger.info("TripoSR: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(TRIPOSR_DIR),
            capture_output=True,
            text=True,
            timeout=int(os.environ.get("TRIPOSR_TIMEOUT_SEC", "600")),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise Image3DError("TripoSR timed out (>10 min). Try a smaller image.") from exc

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        raise Image3DError(f"TripoSR failed:\n{tail}")

    candidates = list(output_dir.rglob("mesh.glb"))
    if not candidates:
        candidates = list(output_dir.rglob("*.glb"))
    if not candidates:
        raise Image3DError(f"TripoSR produced no GLB under {output_dir}")

    shutil.copy2(candidates[0], out_glb)
    return out_glb
