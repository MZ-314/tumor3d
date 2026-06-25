"""Mutable state passed through reconstruction stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pipeline.ingest.images import SliceVolume
from pipeline.segment.backends import SegmentationResult
from shared.schemas.pydantic.pipeline import (
    AnatomicalMap,
    AtlasWarpResult,
    ModelOutputs,
    PipelineArtifacts,
    ReconstructionBlueprint,
    ScanContext,
    StageTiming,
    SynthesisResult,
    ValidationReport,
)
from shared.schemas.pydantic.reconstruct import ReconstructResponse


@dataclass
class PipelineState:
    reconstruction_id: str
    work_dir: Path
    slice_paths: list[Path]
    modality: str
    chat_id: str | None = None
    user_text: str | None = None

    slice_volume: SliceVolume | None = None
    output_volume: SliceVolume | None = None
    segmentation: SegmentationResult | None = None
    organ_mask_2d: object | None = None  # np.ndarray bool HxW

    scan_context: ScanContext | None = None
    model_outputs: ModelOutputs | None = None
    anatomical_map: AnatomicalMap | None = None
    atlas_warp: AtlasWarpResult | None = None
    blueprint: ReconstructionBlueprint | None = None
    synthesis: SynthesisResult | None = None
    validation: ValidationReport | None = None
    response: ReconstructResponse | None = None
    timings: list[StageTiming] = field(default_factory=list)

    def artifacts(self, pipeline_version: str) -> PipelineArtifacts:
        if self.scan_context is None:
            raise RuntimeError("scan_context is required to build PipelineArtifacts")
        return PipelineArtifacts(
            reconstruction_id=self.reconstruction_id,
            pipeline_version=pipeline_version,
            scan_context=self.scan_context,
            model_outputs=self.model_outputs,
            anatomical_map=self.anatomical_map,
            atlas_warp=self.atlas_warp,
            blueprint=self.blueprint,
            synthesis=self.synthesis,
            validation=self.validation,
            stage_timings=self.timings,
        )
