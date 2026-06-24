"""MONAI BraTS brain tumor segmentation (GPU)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from config_medical import DATA_DIR, SegmentationError
from pipeline.segment.backends import LesionMask, SegmentationResult

logger = logging.getLogger(__name__)

BUNDLE_NAME = "brats_mri_segmentation"
ROI_SIZE = (128, 128, 128)
SW_BATCH_SIZE = 1


def _bundle_root() -> Path:
    explicit = os.environ.get("MONAI_BUNDLE_DIR", "").strip()
    if explicit:
        return Path(explicit)
    return DATA_DIR / "monai_bundle" / BUNDLE_NAME


def ensure_brats_bundle() -> Path:
    """Download MONAI Model Zoo BraTS bundle if missing."""
    root = _bundle_root()
    inference_cfg = root / "configs" / "inference.json"
    weights = root / "models" / "model.pt"

    if inference_cfg.is_file() and weights.is_file():
        return root

    if os.environ.get("MONAI_AUTO_DOWNLOAD", "1").lower() in {"0", "false", "no"}:
        raise SegmentationError(
            f"MONAI bundle not found at {root}. Run on RunPod:\n"
            f"  python backend/scripts/setup_monai_bundle.py"
        )

    try:
        import huggingface_hub  # noqa: F401
        from monai.bundle import download as bundle_download
    except ImportError as exc:
        raise SegmentationError(
            "MONAI GPU deps missing. On RunPod run:\n"
            "  pip install huggingface_hub\n"
            "  pip install -e backend/.[gpu]"
        ) from exc

    parent = root.parent
    parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s bundle to %s (first run may take several minutes)…", BUNDLE_NAME, parent)

    try:
        bundle_download(
            name=BUNDLE_NAME,
            bundle_dir=str(parent),
            source="huggingface_hub",
            repo="MONAI/brats_mri_segmentation",
            progress=True,
        )
    except Exception:
        bundle_download(
            name=BUNDLE_NAME,
            bundle_dir=str(parent),
            source="monaihosting",
            progress=True,
        )

    if not inference_cfg.is_file():
        candidates = list(parent.rglob("configs/inference.json"))
        if candidates:
            found = candidates[0].parents[1]
            if found.is_dir():
                return found
    if not inference_cfg.is_file():
        raise SegmentationError(f"Bundle download finished but inference.json missing under {root}")

    return root


def _prepare_brats_tensor(volume: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    BraTS model expects (B, 4, Z, H, W). We repeat a single grayscale channel 4×
  (T1c/T1/T2/FLAIR proxy) — best with real multi-sequence DICOM later.
    """
    if volume.ndim != 3:
        raise SegmentationError(f"Expected volume (Z,H,W), got shape {volume.shape}")

    z, h, w = volume.shape
    # 3D sliding-window needs enough depth; pad thin stacks.
    min_z = 32
    if z < min_z:
        pad = min_z - z
        volume = np.pad(volume, ((0, pad), (0, 0), (0, 0)), mode="edge")

    # In-plane resize toward BraTS-friendly resolution (keeps aspect via min side).
    target = 192
    scale = target / max(h, w)
    new_h, new_w = max(32, int(h * scale)), max(32, int(w * scale))
    if (new_h, new_w) != (h, w):
        from scipy.ndimage import zoom

        volume = zoom(volume, (1, new_h / h, new_w / w), order=1)

    four = np.stack([volume, volume, volume, volume], axis=0)  # (4, Z, H, W)
    tensor = torch.from_numpy(four).float().unsqueeze(0).to(device)  # (1, 4, Z, H, W)
    return tensor


def _load_network(bundle_root: Path, device: torch.device):
    from monai.bundle import ConfigParser

    cfg = bundle_root / "configs" / "inference.json"
    parser = ConfigParser()
    parser.read_config(str(cfg))
    parser["bundle_root"] = str(bundle_root)
    parser["device"] = str(device)

    network = parser.get_parsed_content("network_def")
    weights = bundle_root / "models" / "model.pt"
    if weights.is_file():
        state = torch.load(weights, map_location=device, weights_only=True)
        network.load_state_dict(state)
    network.to(device)
    network.eval()
    return network


def segment_brats(volume: np.ndarray) -> SegmentationResult:
    if not torch.cuda.is_available():
        raise SegmentationError(
            "MONAI BraTS segmentation requires a CUDA GPU. On laptop use SEGMENTATION_BACKEND=stub."
        )

    bundle_root = ensure_brats_bundle()
    device = torch.device("cuda")

    try:
        from monai.inferers import sliding_window_inference
    except ImportError as exc:
        raise SegmentationError("monai.inferers not available") from exc

    network = _load_network(bundle_root, device)
    inputs = _prepare_brats_tensor(volume, device)
    orig_z = volume.shape[0]

    def _predictor(x: torch.Tensor) -> torch.Tensor:
        return network(x)

    with torch.no_grad():
        logits = sliding_window_inference(
            inputs,
            roi_size=ROI_SIZE,
            sw_batch_size=SW_BATCH_SIZE,
            predictor=_predictor,
            overlap=0.25,
        )

    # BraTS: channels = enhancing tumor, tumor core, whole tumor (typical ordering).
    probs = torch.sigmoid(logits)[0].cpu().numpy()
    if probs.shape[0] >= 3:
        wt = probs[2]
    else:
        wt = probs.max(axis=0)

    # Trim depth padding
    wt = wt[:orig_z] if wt.shape[0] > orig_z else wt
    mask3d = wt > 0.5

    if volume.shape[0] == 1 and mask3d.shape[0] > 1:
        mask3d = mask3d[:1]

    labeled, n = ndimage.label(mask3d)
    lesions: list[LesionMask] = []
    for label_id in range(1, n + 1):
        mask = labeled == label_id
        if mask.sum() < 25:
            continue
        lesions.append(
            LesionMask(lesion_id=len(lesions) + 1, mask=mask, in_plane_confidence=0.72)
        )

    if not lesions:
        raise SegmentationError(
            "MONAI did not detect a whole-tumor region. "
            "Try more axial slices, DICOM (T1c), or a scan with visible enhancing tumor."
        )

    lesions.sort(key=lambda l: l.mask.sum(), reverse=True)
    lesions = lesions[:5]
    for i, lesion in enumerate(lesions, start=1):
        lesion.lesion_id = i

    return SegmentationResult(lesions=lesions, global_confidence=0.72)
