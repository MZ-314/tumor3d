"""Mesh optimization — decimate and weld module meshes."""

from __future__ import annotations

from pathlib import Path

import trimesh

from pipeline.modular.types import ModularContext


def run_mesh_optimize(ctx: ModularContext) -> None:
    if ctx.assembly is None:
        return
    root = Path(ctx.assembly.root_glb_path)
    if not root.is_file():
        return
    scene = trimesh.load(root, force="scene")
    if not isinstance(scene, trimesh.Scene):
        return
    for name, geom in scene.geometry.items():
        if hasattr(geom, "merge_vertices"):
            geom.merge_vertices()
        if len(geom.faces) > 8000:
            try:
                geom = geom.simplify_quadric_decimation(8000)
                scene.geometry[name] = geom
            except Exception:
                pass
    scene.export(root)
