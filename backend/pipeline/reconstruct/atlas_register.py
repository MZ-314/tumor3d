"""Atlas registration for patient-specific reconstruction (Phase 4)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from config_medical import MedicalPipelineError
from config_pipeline import ATLAS_BRAIN_TEMPLATE, ATLAS_BRAIN_DIR
from pipeline.ingest.images import SliceVolume
from pipeline.reconstruct.view_orient import atlas_reference_slice, fit_atlas_plane_to_patient
from shared.schemas.pydantic.pipeline import AtlasWarpResult, MriView, OrganType

logger = logging.getLogger(__name__)


def _require_atlas_template() -> Path:
    if not ATLAS_BRAIN_TEMPLATE.is_file():
        raise MedicalPipelineError(
            f"Brain atlas template missing at {ATLAS_BRAIN_TEMPLATE}.\n"
            "On RunPod run: python backend/scripts/setup_brain_atlas.py"
        )
    return ATLAS_BRAIN_TEMPLATE


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
    thickness = volume.slice_thickness_mm

    patient_img = sitk.GetImageFromArray(patient_slice)
    patient_img.SetSpacing((col_sp, row_sp))

    atlas_img = sitk.ReadImage(str(template_path))
    atlas_vol = sitk.GetArrayFromImage(atlas_img).astype(np.float32)
    if atlas_vol.ndim == 3:
        atlas_2d = atlas_reference_slice(atlas_vol, mri_view)
        atlas_2d = fit_atlas_plane_to_patient(atlas_2d, patient_slice, organ_mask_2d)
        atlas_spacing = atlas_img.GetSpacing()
    else:
        atlas_2d = atlas_vol.astype(np.float32)
        atlas_spacing = atlas_img.GetSpacing()

    fixed = sitk.GetImageFromArray(atlas_2d)
    fixed.SetSpacing(atlas_spacing[:2] if len(atlas_spacing) >= 2 else (1.0, 1.0))

    transform = sitk.Euler2DTransform()
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation()
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(1.0, 1e-4, 200)
    registration.SetInitialTransform(transform, inPlace=False)

    try:
        # SimpleITK 2.x: fixed/moving images are passed to Execute(), not SetFixedImage().
        final_transform = registration.Execute(fixed, patient_img)
        confidence = 0.65
    except Exception as exc:
        logger.warning("Atlas registration failed, using identity: %s", exc)
        final_transform = transform
        confidence = 0.25

    transform_path = work_dir / "atlas_transform.tfm"
    sitk.WriteTransform(final_transform, str(transform_path))

    return AtlasWarpResult(
        atlas_id="brain_mni_template",
        atlas_version=ATLAS_BRAIN_DIR.name,
        registration_confidence=confidence,
        estimated_slice_index=anchor_z,
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
