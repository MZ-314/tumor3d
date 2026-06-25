"""ML volume synthesis — conditional slice generator with anchor lock."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter, zoom

from config_pipeline import ML_POSE_CHECKPOINT, ML_VOLUME_MODEL_DIR
from pipeline.ml.brain_envelope import apply_volume_mask, build_extruded_mask_3d
from pipeline.ml.pose import estimate_pose
from pipeline.ml.types import PoseEstimate
from pipeline.ml.preprocess import fit_organ_mask
from pipeline.ml.volume_refiner import refine_volume_3d

logger = logging.getLogger(__name__)

MODEL_SIZE = 128


def _normalize(vol: np.ndarray) -> np.ndarray:
    v = vol.astype(np.float32)
    v = v - v.min()
    return v / (v.max() or 1.0)


def _prepare_plane(
    slice_img: np.ndarray,
    organ_mask: np.ndarray | None,
    *,
    size: int = MODEL_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (2, size, size) tensor planes: intensity + mask."""
    patient = _normalize(slice_img)
    h, w = patient.shape
    if organ_mask is not None and organ_mask.any():
        mask = fit_organ_mask(organ_mask, (h, w)).astype(np.float32)
    else:
        mask = (patient > np.percentile(patient, 35)).astype(np.float32)

    patient_r = zoom(patient, (size / h, size / w), order=1)
    mask_r = zoom(mask, (size / h, size / w), order=0)
    stacked = np.stack([patient_r, mask_r], axis=0).astype(np.float32)
    return stacked, mask


def _extruded_mask(
    organ_mask_2d: np.ndarray,
    target_z: int,
    anchor_z: int,
    h: int,
    w: int,
) -> np.ndarray:
    return build_extruded_mask_3d((target_z, h, w), organ_mask_2d, anchor_z)


def _resolve_checkpoint(explicit: Path | None) -> Path | None:
    if explicit is not None and explicit.is_file():
        return explicit
    default = ML_VOLUME_MODEL_DIR / "volume_generator.pt"
    if default.is_file():
        return default
    return None


def _generate_with_model(
    anchor_slice: np.ndarray,
    organ_mask: np.ndarray | None,
    *,
    target_z: int,
    checkpoint: Path,
    pose: PoseEstimate,
) -> tuple[np.ndarray, str]:
    import torch

    from pipeline.ml.models.conditional_unet import load_volume_generator_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, version = load_volume_generator_checkpoint(checkpoint, device=device)

    patient = _normalize(anchor_slice)
    h, w = patient.shape
    anchor_z = target_z // 2

    if organ_mask is not None and organ_mask.any():
        mask2d = fit_organ_mask(organ_mask, (h, w))
    else:
        mask2d = patient > np.percentile(patient, 35)

    anchor_stack, _ = _prepare_plane(patient, mask2d, size=MODEL_SIZE)

    bg = float(np.median(patient[~mask2d])) if mask2d.any() else 0.0
    extruded = _extruded_mask(mask2d.astype(np.float32), target_z, anchor_z, h, w)

    vol = np.zeros((target_z, h, w), dtype=np.float32)
    vol[anchor_z] = patient

    half = max(anchor_z, 1)
    with torch.no_grad():
        for z in range(target_z):
            if z == anchor_z:
                continue
            offset = float((z - anchor_z) / half)
            offset_plane = np.full((1, MODEL_SIZE, MODEL_SIZE), offset, dtype=np.float32)
            inp_np = np.concatenate(
                [anchor_stack[:1], anchor_stack[1:2], offset_plane],
                axis=0,
            )
            inp = torch.from_numpy(inp_np).unsqueeze(0).to(device)
            pred = model(inp)
            plane = pred.squeeze().cpu().numpy()
            plane = zoom(plane, (h / MODEL_SIZE, w / MODEL_SIZE), order=1)
            vol[z] = plane

    vol = vol * extruded + bg * (1.0 - extruded)
    vol[anchor_z] = patient

    # Depth-aware intensity prior from pose (soft, not atlas texture)
    depth = pose.slice_index_normalized
    for z in range(target_z):
        if z == anchor_z:
            continue
        t = 1.0 - abs(z - anchor_z) / half
        vol[z] = vol[z] * (0.85 + 0.15 * t * depth)

    vol = gaussian_filter(vol, sigma=(0.6, 0.3, 0.3))
    vol[anchor_z] = patient
    vol = apply_volume_mask(vol, extruded, anchor_z=anchor_z, anchor_plane=patient, background=bg)
    vol[anchor_z] = patient

    vol, refiner_used = refine_volume_3d(
        vol,
        mask2d,
        anchor_z=anchor_z,
        anchor_plane=patient,
        background=bg,
    )

    strategy = f"{version}_{pose.mri_view.value}_pose_{pose.source}"
    if refiner_used:
        strategy = f"{strategy}_3d_refined"
    logger.info("ML volume generated: %s planes=%d checkpoint=%s", strategy, target_z, checkpoint.name)
    return np.clip(vol, 0.0, 1.0).astype(np.float32), strategy


def _generate_propagation_fallback(
    anchor_slice: np.ndarray,
    organ_mask: np.ndarray | None,
    *,
    target_z: int,
    pose: PoseEstimate,
) -> tuple[np.ndarray, str]:
    """Gaussian propagation inside brain mask when no checkpoint (dev only)."""
    patient = _normalize(anchor_slice)
    h, w = patient.shape
    anchor_z = target_z // 2

    if organ_mask is not None and organ_mask.any():
        mask2d = fit_organ_mask(organ_mask, (h, w))
    else:
        mask2d = patient > np.percentile(patient, 35)

    extruded = _extruded_mask(mask2d.astype(np.float32), target_z, anchor_z, h, w)
    bg = float(np.median(patient[~mask2d])) if mask2d.any() else 0.0

    vol = np.zeros((target_z, h, w), dtype=np.float32)
    vol[anchor_z] = patient
    for _ in range(6):
        blurred = gaussian_filter(vol, sigma=(2.5, 0.8, 0.8))
        vol = blurred * extruded + bg * (1.0 - extruded)
        vol[anchor_z] = patient

    return vol, f"ml_propagation_fallback_{pose.mri_view.value}"


def generate_ml_volume(
    anchor_slice: np.ndarray,
    *,
    target_z: int,
    organ_mask_2d: np.ndarray | None = None,
    dicom_path: Path | None = None,
    modality: str = "MR",
    checkpoint: Path | None = None,
    pose: PoseEstimate | None = None,
) -> tuple[np.ndarray, str, PoseEstimate]:
    """
    Build a 3D brain volume from one anchor slice using learned slice synthesis.

    Anchor plane is locked to measured intensities. Off-slice planes are ML-generated
    parallel to the anchor (same imaging plane family), not orthogonal atlas textures.
    """
    if pose is None:
        pose = estimate_pose(
            anchor_slice,
            organ_mask=organ_mask_2d,
            dicom_path=dicom_path,
            modality=modality,
            pose_checkpoint=ML_POSE_CHECKPOINT if ML_POSE_CHECKPOINT.is_file() else None,
        )

    ckpt = _resolve_checkpoint(checkpoint)
    if ckpt is not None:
        try:
            vol, strategy = _generate_with_model(
                anchor_slice,
                organ_mask_2d,
                target_z=target_z,
                checkpoint=ckpt,
                pose=pose,
            )
            return vol, strategy, pose
        except Exception as exc:
            logger.error("ML volume inference failed, falling back: %s", exc)
            raise

    vol, strategy = _generate_propagation_fallback(
        anchor_slice,
        organ_mask_2d,
        target_z=target_z,
        pose=pose,
    )
    return vol, strategy, pose
