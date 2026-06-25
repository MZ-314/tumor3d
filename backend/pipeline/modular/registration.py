"""Patient atlas registration — SimpleITK rigid + optional VoxelMorph."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from config_medical import MedicalPipelineError, REPO_ROOT
from config_pipeline import ATLAS_BRAIN_TEMPLATE
from pipeline.reconstruct.atlas_register import register_brain_atlas
from pipeline.modular.types import ModularContext
from shared.schemas.pydantic.pipeline import MriView


def _identity_4x4() -> list[list[float]]:
    return [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def _scale_from_mask(mask: np.ndarray, organ_mask: np.ndarray) -> float:
    def extent(m: np.ndarray) -> float:
        rows, cols = np.where(m)
        if rows.size == 0:
            return 1.0
        return float(max(rows.max() - rows.min(), cols.max() - cols.min(), 1))

    patient_ext = extent(organ_mask)
    atlas_ext = extent(mask)
    if atlas_ext <= 0:
        return 1.0
    return float(np.clip(patient_ext / atlas_ext, 0.75, 1.35))


def _sitk_transform_to_4x4(
    transform_path: Path,
    *,
    row_sp: float,
    col_sp: float,
    scale: float,
) -> list[list[float]]:
    """Convert SimpleITK 2D Euler registration to a 3D module transform."""
    matrix = _identity_4x4()
    matrix[0][0] = scale
    matrix[1][1] = scale
    matrix[2][2] = scale

    if not transform_path.is_file():
        return matrix

    try:
        import SimpleITK as sitk

        tfm = sitk.ReadTransform(str(transform_path))
        params = list(tfm.GetParameters())
        # Euler2DTransform: [angle, tx, ty] in physical units (mm)
        if len(params) >= 3:
            angle, tx, ty = float(params[0]), float(params[1]), float(params[2])
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            # Map image row/col translation into module X/Y (col→x, row→y)
            matrix[0][0] = scale * cos_a
            matrix[0][1] = -scale * sin_a
            matrix[1][0] = scale * sin_a
            matrix[1][1] = scale * cos_a
            matrix[0][3] = tx * col_sp * 0.01
            matrix[1][3] = ty * row_sp * 0.01
    except Exception:
        pass
    return matrix


def _flow_translation(flow: np.ndarray, row_sp: float, col_sp: float) -> tuple[float, float]:
    """Median in-plane displacement from VoxelMorph flow field."""
    if flow.ndim != 2:
        return 0.0, 0.0
    rows, cols = np.mgrid[0 : flow.shape[0], 0 : flow.shape[1]]
    weights = np.abs(flow)
    total = float(weights.sum()) or 1.0
    mean_row = float((rows * weights).sum() / total)
    mean_col = float((cols * weights).sum() / total)
    return mean_col * col_sp * 0.002, mean_row * row_sp * 0.002


def _voxelmorph_checkpoint() -> Path:
    return REPO_ROOT / "models" / "voxelmorph" / "voxelmorph_brain.pt"


def _try_voxelmorph_displacement(
    patient_slice: np.ndarray,
    atlas_slice: np.ndarray,
) -> np.ndarray | None:
    """Optional MONAI VoxelMorph inference when weights are present."""
    ckpt = _voxelmorph_checkpoint()
    if not ckpt.is_file():
        return None
    try:
        import torch
        from monai.networks.nets import VoxelMorph

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = VoxelMorph(spatial_dims=2, in_channels=2, out_channels=2).to(device)
        state = torch.load(ckpt, map_location=device, weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        model.eval()

        p = patient_slice.astype(np.float32)
        a = atlas_slice.astype(np.float32)
        p = (p - p.min()) / (p.max() - p.min() or 1.0)
        a = (a - a.min()) / (a.max() - a.min() or 1.0)

        fixed = torch.from_numpy(a[None, None]).float().to(device)
        moving = torch.from_numpy(p[None, None]).float().to(device)
        with torch.no_grad():
            _warped, flow = model(moving, fixed)
        return flow.detach().cpu().numpy()[0, 0]
    except Exception:
        return None


def run_registration(ctx: ModularContext) -> None:
    if not ATLAS_BRAIN_TEMPLATE.is_file():
        raise MedicalPipelineError(
            f"Brain template missing at {ATLAS_BRAIN_TEMPLATE}. "
            "Run: python backend/scripts/setup_brain_atlas.py"
        )

    mri_view = ctx.pose.mri_view if ctx.pose else ctx.scan_context.mri_view
    if mri_view == MriView.UNKNOWN:
        mri_view = MriView.AXIAL

    ctx.atlas_warp = register_brain_atlas(
        ctx.slice_volume,
        ctx.organ_mask_2d,
        work_dir=ctx.work_dir,
        anchor_z=ctx.anchor_z,
        mri_view=mri_view,
    )

    anchor = ctx.anchor_z
    patient_slice = ctx.slice_volume.data[anchor].astype(np.float32)
    row_sp, col_sp = ctx.slice_volume.pixel_spacing_mm
    from pipeline.reconstruct.atlas_volume import find_best_atlas_slice_index, load_oriented_atlas
    from pipeline.reconstruct.view_orient import fit_atlas_plane_to_patient

    atlas_vol = load_oriented_atlas(mri_view)
    if atlas_vol is None:
        raise MedicalPipelineError("Could not load oriented atlas volume")

    best_i = find_best_atlas_slice_index(patient_slice, atlas_vol, ctx.organ_mask_2d)
    atlas_slice = fit_atlas_plane_to_patient(
        atlas_vol[best_i].astype(np.float32),
        patient_slice,
        ctx.organ_mask_2d,
    )
    atlas_mask = atlas_slice > np.percentile(atlas_slice, 55)
    scale = _scale_from_mask(atlas_mask, ctx.organ_mask_2d)

    flow = _try_voxelmorph_displacement(patient_slice, atlas_slice)

    tfm_path = ctx.work_dir / "atlas_transform.tfm"
    if ctx.atlas_warp.transform_path:
        rel = Path(ctx.atlas_warp.transform_path)
        candidate = ctx.work_dir / rel
        if candidate.is_file():
            tfm_path = candidate

    transform = _sitk_transform_to_4x4(tfm_path, row_sp=row_sp, col_sp=col_sp, scale=scale)
    if flow is not None:
        dx, dy = _flow_translation(flow, row_sp, col_sp)
        transform[0][3] += dx
        transform[1][3] += dy

    reg_meta = {
        "scale_xy": scale,
        "atlas_slice_index": int(best_i),
        "voxelmorph_applied": flow is not None,
        "transform_4x4": transform,
    }
    (ctx.work_dir / "module_registration.json").write_text(
        json.dumps(reg_meta, indent=2),
        encoding="utf-8",
    )

    updated: list = []
    for mod in ctx.modules:
        updated.append(mod.model_copy(update={"transform_4x4": transform}))
    ctx.modules = updated
