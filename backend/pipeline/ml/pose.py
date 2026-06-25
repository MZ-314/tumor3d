"""Automatic organ + slice pose estimation from image and DICOM metadata."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from pipeline.ingest.dicom_meta import detect_mri_view_from_dicom, organ_type_from_dicom
from pipeline.ml.types import PoseEstimate
from shared.schemas.pydantic.pipeline import MriView, OrganType

logger = logging.getLogger(__name__)

_VIEW_TO_AXIS: dict[MriView, int] = {
    MriView.AXIAL: 0,
    MriView.CORONAL: 1,
    MriView.SAGITTAL: 2,
    MriView.UNKNOWN: 0,
}


def _mri_view_to_axis(view: MriView) -> int:
    return _VIEW_TO_AXIS.get(view, 0)


def _estimate_depth_from_image(slice_img: np.ndarray, organ_mask: np.ndarray | None) -> float:
    """Coarse slice depth prior from in-plane brain size (larger ≈ mid-brain)."""
    img = slice_img.astype(np.float32)
    mask = organ_mask.astype(bool) if organ_mask is not None else img > np.percentile(img, 55)
    if not mask.any():
        return 0.5
    area = float(mask.sum()) / float(mask.size)
    # Typical axial mid-slice has larger brain cross-section than vertex/cerebellum cuts.
    return float(np.clip(0.25 + area * 0.55, 0.05, 0.95))


def _ml_pose_from_image(
    slice_img: np.ndarray,
    organ_mask: np.ndarray | None,
    checkpoint_path: Path | None,
) -> PoseEstimate | None:
    """Optional CNN pose head when checkpoint is available."""
    if checkpoint_path is None or not checkpoint_path.is_file():
        return None
    try:
        import torch

        from pipeline.ml.models.pose_net import PoseNet, load_pose_checkpoint
    except ImportError:
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_pose_checkpoint(checkpoint_path, device=device)
    if model is None:
        return None

    from pipeline.ml.volume_generator import _prepare_plane

    plane, _ = _prepare_plane(slice_img, organ_mask, size=128)
    tensor = torch.from_numpy(plane).unsqueeze(0).to(device)
    with torch.no_grad():
        axis_logits, depth = model(tensor)
    axis = int(axis_logits.argmax(dim=1).item())
    depth_val = float(depth.squeeze().item())
    view = (MriView.AXIAL, MriView.CORONAL, MriView.SAGITTAL)[axis]
    return PoseEstimate(
        organ_type=OrganType.BRAIN,
        through_plane_axis=axis,
        slice_index_normalized=depth_val,
        mri_view=view,
        confidence=0.72,
        source="ml",
    )


def route_organ(
    slice_img: np.ndarray,
    *,
    dicom_path: Path | None = None,
    modality: str = "MR",
) -> OrganType:
    """Infer organ from DICOM tags when available, else intensity heuristics."""
    if dicom_path is not None and dicom_path.is_file():
        return organ_type_from_dicom(dicom_path, modality)
    # Bright oval on dark background → brain MRI heuristic for untagged PNG uploads.
    img = slice_img.astype(np.float32)
    bright = img > np.percentile(img, 70)
    if bright.sum() / bright.size > 0.08:
        return OrganType.BRAIN
    return OrganType.UNKNOWN


def estimate_pose(
    slice_img: np.ndarray,
    *,
    organ_mask: np.ndarray | None = None,
    dicom_path: Path | None = None,
    modality: str = "MR",
    pose_checkpoint: Path | None = None,
) -> PoseEstimate:
    """
    Infer organ + continuous slice pose without user-supplied view labels.

    Priority: DICOM orientation tags → ML pose head → pixel heuristics.
    """
    organ = route_organ(slice_img, dicom_path=dicom_path, modality=modality)

    if dicom_path is not None and dicom_path.is_file():
        view = detect_mri_view_from_dicom(dicom_path)
        if view != MriView.UNKNOWN:
            depth = _estimate_depth_from_image(slice_img, organ_mask)
            return PoseEstimate(
                organ_type=organ,
                through_plane_axis=_mri_view_to_axis(view),
                slice_index_normalized=depth,
                mri_view=view,
                confidence=0.92,
                source="dicom",
            )

    ml_pose = _ml_pose_from_image(slice_img, organ_mask, pose_checkpoint)
    if ml_pose is not None:
        return PoseEstimate(
            organ_type=organ,
            through_plane_axis=ml_pose.through_plane_axis,
            slice_index_normalized=ml_pose.slice_index_normalized,
            mri_view=ml_pose.mri_view,
            confidence=ml_pose.confidence,
            source=ml_pose.source,
        )

    from pipeline.reconstruct.view_orient import detect_mri_view_from_pixels

    view = detect_mri_view_from_pixels(slice_img, organ_mask)
    depth = _estimate_depth_from_image(slice_img, organ_mask)
    logger.info("Pose from pixels: view=%s depth=%.2f", view.value, depth)
    return PoseEstimate(
        organ_type=organ,
        through_plane_axis=_mri_view_to_axis(view),
        slice_index_normalized=depth,
        mri_view=view,
        confidence=0.55,
        source="heuristic",
    )
