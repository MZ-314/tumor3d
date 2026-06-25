"""Module assembly — Brain.glb + per-module exports + tumor module."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import trimesh

from config_pipeline import MODULAR_BRAIN_DIR
from pipeline.modular.types import ModularContext
from shared.schemas.pydantic.common import SourceType
from shared.schemas.pydantic.pipeline import AnatomicalModule, ModuleAssemblyResult


def _build_tumor_mesh(ctx: ModularContext) -> trimesh.Trimesh | None:
    if ctx.lesion_mask_2d is None or not ctx.lesion_mask_2d.any():
        return None
    from skimage import measure

    mask = ctx.lesion_mask_2d.astype(np.float32)
    row_sp, col_sp = ctx.slice_volume.pixel_spacing_mm
    try:
        verts, faces, _n, _v = measure.marching_cubes(mask, level=0.5, spacing=(row_sp, col_sp, 1.0))
    except (ValueError, RuntimeError):
        rows, cols = np.where(ctx.lesion_mask_2d)
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        w = (c1 - c0 + 1) * col_sp
        h = (r1 - r0 + 1) * row_sp
        box = trimesh.creation.box(extents=[w, h, row_sp * 2])
        box.apply_translation([(c0 + c1) / 2 * col_sp, (r0 + r1) / 2 * row_sp, 0])
        return box
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    mesh.vertices[:, 2] *= 0.5
    return mesh


def run_module_assembly(ctx: ModularContext) -> ModuleAssemblyResult:
    out_dir = ctx.work_dir / "assembly"
    out_dir.mkdir(parents=True, exist_ok=True)
    modules_json_dir = ctx.work_dir / "modules_meta"
    modules_json_dir.mkdir(parents=True, exist_ok=True)

    scene = trimesh.Scene()
    final_modules: list[AnatomicalModule] = []

    for mod in ctx.modules:
        path = Path(mod.mesh_path)
        if not path.is_file():
            fallback = MODULAR_BRAIN_DIR / mod.module_id / "mesh.glb"
            path = fallback
        if path.is_file():
            mesh = trimesh.load(path, force="mesh")
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
            scene.add_geometry(mesh, node_name=mod.module_id)
        meta_out = modules_json_dir / f"{mod.module_id}.json"
        meta_out.write_text(mod.model_dump_json(indent=2), encoding="utf-8")
        final_modules.append(
            mod.model_copy(
                update={
                    "metadata_path": str(meta_out),
                    "mesh_path": str(path) if path.is_file() else mod.mesh_path,
                }
            )
        )

    root_glb = out_dir / f"{ctx.reconstruction_id}_Brain.glb"
    scene.export(root_glb)

    tumor_glb_path: str | None = None
    tumor_mesh = _build_tumor_mesh(ctx)
    if tumor_mesh is not None:
        tumor_path = out_dir / f"{ctx.reconstruction_id}_tumor.glb"
        tumor_mesh.export(tumor_path)
        tumor_glb_path = str(tumor_path)
        tumor_mod = AnatomicalModule(
            module_id="tumor",
            display_name="Tumor",
            mesh_path=str(tumor_path),
            geometry_source=SourceType.MEASURED,
            confidence=0.9,
            anchor_locked=True,
            morph_applied=False,
        )
        final_modules.append(tumor_mod)
        (modules_json_dir / "tumor.json").write_text(
            tumor_mod.model_dump_json(indent=2),
            encoding="utf-8",
        )

    manifest_copy = out_dir / "module_manifest.json"
    src_manifest = MODULAR_BRAIN_DIR / "manifest.json"
    if src_manifest.is_file():
        shutil.copy2(src_manifest, manifest_copy)

    assembly = ModuleAssemblyResult(
        root_glb_path=str(root_glb),
        tumor_glb_path=tumor_glb_path,
        module_manifest_path=str(manifest_copy) if manifest_copy.is_file() else None,
        modules=final_modules,
        graph=ctx.graph,
        structure_replacement=ctx.structure_replacement,
    )
    ctx.assembly = assembly
    (ctx.work_dir / "module_assembly.json").write_text(
        assembly.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return assembly
