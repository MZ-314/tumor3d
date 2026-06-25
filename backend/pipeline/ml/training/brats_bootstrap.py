"""BraTS bootstrap volumes for 3D volume-completion training."""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import numpy as np

from config_medical import REPO_ROOT
from config_pipeline import ATLAS_BRAIN_TEMPLATE

logger = logging.getLogger(__name__)

BRATS_BOOTSTRAP_DIR = Path(
    __import__("os").environ.get(
        "BRATS_BOOTSTRAP_DIR",
        str(REPO_ROOT / "data" / "training" / "brats_bootstrap"),
    )
)

# Public T1 brain volumes (bootstrap when full BraTS challenge set is unavailable).
BOOTSTRAP_URLS: tuple[str, ...] = (
    "https://raw.githubusercontent.com/InstitutdeNeurosciencesdesSystems/"
    "ADF-pipeline-datarefs/main/MNI152_T1_1mm.nii.gz",
)


def _augment_volume(vol: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = vol.astype(np.float32).copy()
    out = out - out.min()
    out = out / (out.max() or 1.0)
    bias = 1.0 + 0.08 * np.sin(np.linspace(0, 4 * np.pi, out.shape[0], dtype=np.float32))
    out = out * bias[:, None, None]
    gamma = float(rng.uniform(0.85, 1.15))
    out = np.power(np.clip(out, 0, 1), gamma)
    noise = rng.normal(0, 0.02, size=out.shape).astype(np.float32)
    return np.clip(out + noise, 0, 1).astype(np.float32)


def _save_nifti(vol: np.ndarray, path: Path) -> None:
    import nibabel as nib

    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(vol.astype(np.float32), np.eye(4))
    nib.save(img, str(path))


def _load_nifti(path: Path) -> np.ndarray:
    from pipeline.ml.training.dataset import load_brain_volume_nifti

    return load_brain_volume_nifti(path, max_dim=None)


def ensure_brats_bootstrap_volumes(*, min_volumes: int = 4) -> list[Path]:
    """
    Ensure a small set of 3D brain volumes for anchor-locked completion training.

    Priority: existing files in BRATS_BOOTSTRAP_DIR → download public T1 →
    MNI152 template with BraTS-style intensity augmentations.
    """
    BRATS_BOOTSTRAP_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(BRATS_BOOTSTRAP_DIR.glob("*.nii.gz"))
    if len(existing) >= min_volumes:
        return existing[: min(len(existing), 12)]

    paths: list[Path] = list(existing)
    for i, url in enumerate(BOOTSTRAP_URLS):
        dest = BRATS_BOOTSTRAP_DIR / f"bootstrap_download_{i}.nii.gz"
        if not dest.is_file():
            try:
                logger.info("Downloading bootstrap volume %s", url)
                urllib.request.urlretrieve(url, dest)
            except Exception as exc:
                logger.warning("Bootstrap download failed: %s", exc)
                continue
        if dest.is_file():
            paths.append(dest)

    if ATLAS_BRAIN_TEMPLATE.is_file():
        base = _load_nifti(ATLAS_BRAIN_TEMPLATE)
        for seed in range(max(0, min_volumes - len(paths))):
            aug_path = BRATS_BOOTSTRAP_DIR / f"mni_augment_{seed}.nii.gz"
            if not aug_path.is_file():
                _save_nifti(_augment_volume(base, seed + 17), aug_path)
            paths.append(aug_path)

    if not paths:
        raise FileNotFoundError(
            f"No bootstrap volumes in {BRATS_BOOTSTRAP_DIR}. "
            "Run setup_brain_atlas.py first."
        )
    return paths
