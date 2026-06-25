"""Atlas registration for patient-specific reconstruction (Phase 4)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from config_medical import MedicalPipelineError
from config_pipeline import ATLAS_BRAIN_TEMPLATE, ATLAS_BRAIN_DIR
from pipeline.ingest.images import SliceVolume
from pipeline.reconstruct.atlas_volume import find_best_atlas_slice_index, load_oriented_atlas
from pipeline.reconstruct.view_orient import fit_atlas_plane_to_patient
from shared.schemas.pydantic.pipeline import AtlasWarpResult, MriView, OrganType

logger = logging.getLogger(__name__)


def _require_atlas_template() -> Path:
    if not ATLAS_BRAIN_TEMPLATE.is_file():
        raise MedicalPipelineError(
            f"Brain atlas template missing at {ATLAS_BRAIN_TEMPLATE}.\n"
            "On RunPod run: python backend/scripts/setup_brain_atlas.py"
        )
    return ATLAS_BRAIN_TEMPLATE


def _centroid_xy(mask: np.ndarray) -> tuple[float, float]:
    rows, cols = np.where(mask)
    if rows.size == 0:
        h, w = mask.shape
        return w / 2.0, h / 2.0
    return float(cols.mean()), float(rows.mean())


def register_brain_atlas(
    volume: SliceVolume,
    organ_mask_2d: np.ndarray,
    *,
    work_dir: Path,
    anchor_z: int,
    mri_view: MriView = MriView.UNKNOWN,
) -> AtlasWarpResult:
    template_path = _require_atlas_template()

    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise MedicalPipelineError(
            "Atlas registration requires SimpleITK. Install with:\n"
            "  pip install SimpleITK"
        ) from exc

    patient_slice = volume.data[anchor_z].astype(np.float32)
    row_sp, col_sp = volume.pixel_spacing_mm

    patient_img = sitk.GetImageFromArray(patient_slice)
    patient_img.SetSpacing((col_sp, row_sp))

    atlas_oriented = load_oriented_atlas(mri_view)
    if atlas_oriented is None:
        raise MedicalPipelineError("Brain atlas volume could not be loaded for registration.")

    best_i = find_best_atlas_slice_index(patient_slice, atlas_oriented, organ_mask_2d)
    atlas_2d = atlas_oriented[best_i].astype(np.float32)
    atlas_2d = fit_atlas_plane_to_patient(atlas_2d, patient_slice, organ_mask_2d)

    fixed = sitk.GetImageFromArray(atlas_2d)
    fixed.SetSpacing((col_sp, row_sp))

    # Initialize translation from organ-mask centroids (atlas vs patient)
    transform = sitk.Euler2DTransform()
    if organ_mask_2d.any():
        atlas_mask = atlas_2d > np.percentile(atlas_2d, 55)
        ax, ay = _centroid_xy(atlas_mask)
        px, py = _centroid_xy(organ_mask_2d)
        transform.SetTranslation((float(px - ax), float(py - ay)))

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=64)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=300,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetInitialTransform(transform, inPlace=False)

    try:
        final_transform = registration.Execute(fixed, patient_img)
        confidence = 0.72
    except Exception as exc:
        logger.warning("Atlas registration failed, using centroid init: %s", exc)
        final_transform = transform
        confidence = 0.35

    transform_path = work_dir / "atlas_transform.tfm"
    sitk.WriteTransform(final_transform, str(transform_path))

    return AtlasWarpResult(
        atlas_id="brain_mni_template",
        atlas_version=ATLAS_BRAIN_DIR.name,
        registration_confidence=confidence,
        estimated_slice_index=best_i,
        transform_path=str(transform_path.relative_to(work_dir)),
        constraint_weights={"symmetry": 0.5, "continuity": 0.5},
    )


def run_atlas_for_organ(
    organ_type: OrganType,
    volume: SliceVolume,
    organ_mask_2d: np.ndarray,
    *,
    work_dir: Path,
    anchor_z: int,
    mri_view: MriView = MriView.UNKNOWN,
) -> AtlasWarpResult:
    if organ_type != OrganType.BRAIN:
        raise MedicalPipelineError(
            f"Atlas matching for organ '{organ_type.value}' is not configured yet. "
            "Brain MRI is supported in v1."
        )
    return register_brain_atlas(
        volume,
        organ_mask_2d,
        work_dir=work_dir,
        anchor_z=anchor_z,
        mri_view=mri_view,
    )
