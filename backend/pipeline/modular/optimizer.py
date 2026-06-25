"""Math optimization — Laplacian smooth on module interfaces."""

from __future__ import annotations

from pathlib import Path

import trimesh

from pipeline.modular.types import ModularContext


def run_optimizer(ctx: ModularContext) -> None:
    modules_dir = ctx.work_dir / "modules"
    updated = []
    for mod in ctx.modules:
        path = Path(mod.mesh_path)
        if not path.is_file():
            updated.append(mod)
            continue
        mesh = trimesh.load(path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        try:
            trimesh.smoothing.filter_laplacian(mesh, iterations=2)
        except Exception:
            pass
        opt_path = modules_dir / f"{mod.module_id}_optimized.glb"
        mesh.export(opt_path)
        updated.append(mod.model_copy(update={"mesh_path": str(opt_path)}))
    ctx.modules = updated
