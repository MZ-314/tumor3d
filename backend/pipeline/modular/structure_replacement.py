"""MRI structure replacement — warp module geometry only; anchor intensities locked."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from pipeline.modular.types import ModularContext
from shared.schemas.pydantic.pipeline import StructureReplacementResult


def _apply_transform(mesh: trimesh.Trimesh, transform_4x4: list[list[float]]) -> trimesh.Trimesh:
    m = np.asarray(transform_4x4, dtype=np.float64)
    out = mesh.copy()
    ones = np.ones((out.vertices.shape[0], 1))
    hom = np.hstack([out.vertices, ones])
    out.vertices = (hom @ m.T)[:, :3]
    return out


def run_structure_replacement(ctx: ModularContext) -> None:
    warped_ids: list[str] = []
    warped_meshes: dict[str, trimesh.Trimesh] = {}

    for mod in ctx.modules:
        if not Path(mod.mesh_path).is_file():
            continue
        mesh = trimesh.load(mod.mesh_path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        warped = _apply_transform(mesh, mod.transform_4x4)
        warped_meshes[mod.module_id] = warped
        warped_ids.append(mod.module_id)
        out_path = ctx.work_dir / "modules" / f"{mod.module_id}_warped.glb"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        warped.export(out_path)

    ctx.structure_replacement = StructureReplacementResult(
        modules_warped=warped_ids,
        anchor_locked=True,
        notes=["Geometry-only warp; patient anchor slice intensities preserved."],
    )
    ctx._warped_meshes = warped_meshes  # type: ignore[attr-defined]
