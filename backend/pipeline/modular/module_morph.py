"""Local morphing — vertex displacement for affected modules."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from pipeline.modular.types import ModularContext
from shared.schemas.pydantic.common import SourceType


def _morph_mesh(mesh: trimesh.Trimesh, strength: float = 0.08) -> trimesh.Trimesh:
    out = mesh.copy()
    center = out.vertices.mean(axis=0)
    dirs = out.vertices - center
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-6)
    out.vertices = out.vertices + (dirs / norms) * strength * norms.mean()
    return out


def run_local_morph(ctx: ModularContext) -> None:
    modules_dir = ctx.work_dir / "modules"
    updated = []
    for mod in ctx.modules:
        if mod.module_id not in ctx.affected_module_ids or not mod.connects_to:
            updated.append(mod)
            continue
        src = modules_dir / f"{mod.module_id}_warped.glb"
        if not src.is_file():
            src = Path(mod.mesh_path)
        mesh = trimesh.load(src, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        morphed = _morph_mesh(mesh)
        out_path = modules_dir / f"{mod.module_id}_morphed.glb"
        morphed.export(out_path)
        updated.append(
            mod.model_copy(
                update={
                    "morph_applied": True,
                    "mesh_path": str(out_path),
                    "geometry_source": SourceType.INFERENCE,
                    "confidence": min(0.95, mod.confidence + 0.05),
                }
            )
        )
    ctx.modules = updated
