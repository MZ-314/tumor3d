"""Modular atlas library — load manifest.json + module meshes."""

from __future__ import annotations

import json
from pathlib import Path

from config_medical import MedicalPipelineError
from config_pipeline import MODULAR_BRAIN_DIR
from pipeline.modular.types import ModularContext
from shared.schemas.pydantic.common import SourceType
from shared.schemas.pydantic.pipeline import AnatomicalModule, ModuleGraph, ModuleGraphEdge


def load_manifest(atlas_dir: Path | None = None) -> dict:
    root = atlas_dir or MODULAR_BRAIN_DIR
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise MedicalPipelineError(
            f"Modular brain manifest missing at {manifest_path}.\n"
            "Run: python backend/scripts/setup_brain_modules.py"
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def run_atlas_library(ctx: ModularContext) -> list[AnatomicalModule]:
    root = MODULAR_BRAIN_DIR
    manifest = load_manifest(root)
    modules: list[AnatomicalModule] = []
    nodes: list[str] = []
    edges: list[ModuleGraphEdge] = []

    for entry in manifest.get("modules", []):
        module_id = str(entry["id"])
        mesh_rel = str(entry["mesh"])
        mesh_path = str((root / mesh_rel).resolve())
        meta_rel = entry.get("metadata")
        meta_path = str((root / meta_rel).resolve()) if meta_rel else None
        display_name = module_id.replace("_", " ")
        if meta_path and Path(meta_path).is_file():
            import json

            meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
            display_name = str(meta.get("display_name", display_name))
        connects = list(entry.get("connects_to", []))
        modules.append(
            AnatomicalModule(
                module_id=module_id,
                display_name=display_name,
                mesh_path=mesh_path,
                geometry_source=SourceType.INFERENCE,
                confidence=float(entry.get("default_confidence", 0.65)),
                connects_to=connects,
                metadata_path=meta_path,
            )
        )
        nodes.append(module_id)
        for target in connects:
            edges.append(ModuleGraphEdge(source_id=module_id, target_id=target))

    ctx.modules = modules
    ctx.graph = ModuleGraph(nodes=nodes, edges=edges)
    return modules
