"""Neighbor propagation — deform connected modules with falloff."""

from __future__ import annotations

from pathlib import Path

import trimesh

from pipeline.modular.types import ModularContext


def run_neighbor_propagation(ctx: ModularContext) -> None:
    if not ctx.graph or not ctx.affected_module_ids:
        return

    primary = ctx.affected_module_ids[0]
    neighbor_ids: set[str] = set()
    for edge in ctx.graph.edges:
        if edge.source_id == primary:
            neighbor_ids.add(edge.target_id)
        if edge.target_id == primary:
            neighbor_ids.add(edge.source_id)

    modules_dir = ctx.work_dir / "modules"
    updated = []
    for mod in ctx.modules:
        if mod.module_id not in neighbor_ids or mod.morph_applied:
            updated.append(mod)
            continue
        src = modules_dir / f"{mod.module_id}_warped.glb"
        if not src.is_file():
            updated.append(mod)
            continue
        mesh = trimesh.load(src, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        center = mesh.vertices.mean(axis=0)
        mesh.vertices = center + (mesh.vertices - center) * 1.02
        out_path = modules_dir / f"{mod.module_id}_propagated.glb"
        mesh.export(out_path)
        updated.append(
            mod.model_copy(
                update={
                    "mesh_path": str(out_path),
                    "morph_applied": True,
                }
            )
        )
    ctx.modules = updated
