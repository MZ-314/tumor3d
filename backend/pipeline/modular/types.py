"""Internal types for modular reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pipeline.ingest.images import SliceVolume
from shared.schemas.pydantic.pipeline import (
    AnatomicalMap,
    AnatomicalModule,
    AtlasWarpResult,
    ModuleAssemblyResult,
    ModuleGraph,
    PoseEstimate,
    ReconstructionBlueprint,
    ScanContext,
    StructureReplacementResult,
    SynthesisResult,
)


@dataclass
class ModularContext:
    reconstruction_id: str
    work_dir: Path
    slice_volume: SliceVolume
    organ_mask_2d: np.ndarray
    anatomical_map: AnatomicalMap | None
    scan_context: ScanContext
    anchor_z: int

    pose: PoseEstimate | None = None
    atlas_warp: AtlasWarpResult | None = None
    blueprint: ReconstructionBlueprint | None = None
    modules: list[AnatomicalModule] = field(default_factory=list)
    graph: ModuleGraph | None = None
    structure_replacement: StructureReplacementResult | None = None
    affected_module_ids: list[str] = field(default_factory=list)
    synthesis: SynthesisResult | None = None
    assembly: ModuleAssemblyResult | None = None
    lesion_mask_2d: np.ndarray | None = None
    output_volume: SliceVolume | None = None
