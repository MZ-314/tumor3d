"""Pipeline contracts for patient-specific 3D reconstruction (Phases 0–10)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from shared.schemas.pydantic.common import AccuracyTier, ProvenanceField, SourceType, Vec3


class MriView(str, Enum):
    AXIAL = "axial"
    CORONAL = "coronal"
    SAGITTAL = "sagittal"
    UNKNOWN = "unknown"


class OrganType(str, Enum):
    BRAIN = "brain"
    KNEE = "knee"
    OTHER = "other"
    UNKNOWN = "unknown"


class InputSource(str, Enum):
    DICOM = "dicom"
    IMAGE = "image"
    MONTAGE = "montage"


class SliceSpacing(BaseModel):
    row_mm: float = Field(..., gt=0.0)
    col_mm: float = Field(..., gt=0.0)
    slice_mm: float = Field(..., gt=0.0)
    source: SourceType = SourceType.INFERENCE


class ScanContext(BaseModel):
    """Normalized scan metadata after input intelligence (Phase 1)."""

    reconstruction_id: str
    input_source: InputSource
    organ_type: OrganType
    modality: str
    mri_view: MriView = MriView.UNKNOWN
    accuracy_tier: AccuracyTier
    slice_count: int = Field(..., ge=1)
    anchor_slice_indices: list[int] = Field(default_factory=list)
    slice_spacing_mm: SliceSpacing | None = None
    volume_shape_zyx: list[int] | None = Field(
        default=None,
        description="[z, y, x] voxel dimensions when volume is assembled",
    )
    body_part_examined: str | None = None
    series_description: str | None = None
    montage_panels: int | None = None
    quality_score: float = Field(1.0, ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)


class Landmark2D(BaseModel):
    name: str
    row: float
    col: float
    confidence: float = Field(..., ge=0.0, le=1.0)


class LesionCandidate2D(BaseModel):
    lesion_id: str
    mask_rle_path: str | None = None
    bounding_box_path: str | None = None
    in_plane_confidence: float = Field(..., ge=0.0, le=1.0)
    model_source: str


class ModelOutputs(BaseModel):
    """Raw per-model 2D analysis (Phase 2)."""

    organ_mask_path: str | None = None
    organ_confidence: float = Field(0.0, ge=0.0, le=1.0)
    lesion_candidates: list[LesionCandidate2D] = Field(default_factory=list)
    landmarks: list[Landmark2D] = Field(default_factory=list)
    organ_classifier_confidence: float = Field(0.0, ge=0.0, le=1.0)
    model_versions: dict[str, str] = Field(default_factory=dict)


class AnatomicalMap(BaseModel):
    """Fused 2D organ + lesion map with measurements (Phase 3)."""

    organ_mask_path: str | None = None
    lesion_mask_paths: list[str] = Field(default_factory=list)
    organ_area_mm2: ProvenanceField | None = None
    lesion_centroids_mm: list[Vec3] = Field(default_factory=list)
    landmarks: list[Landmark2D] = Field(default_factory=list)
    distances_mm: dict[str, ProvenanceField] = Field(default_factory=dict)
    fusion_confidence: float = Field(..., ge=0.0, le=1.0)
    consensus_notes: list[str] = Field(default_factory=list)


class AtlasWarpResult(BaseModel):
    """Atlas registration outcome (Phase 4)."""

    atlas_id: str
    atlas_version: str
    registration_confidence: float = Field(..., ge=0.0, le=1.0)
    estimated_slice_index: int | None = None
    transform_path: str | None = None
    constraint_weights: dict[str, float] = Field(default_factory=dict)


class ReconstructionBlueprint(BaseModel):
    """Patient-specific volume blueprint (Phase 5)."""

    volume_shape_zyx: list[int] = Field(..., min_length=3, max_length=3)
    anchor_slice_indices: list[int] = Field(default_factory=list)
    label_map_seed_path: str | None = None
    organ_extent_zyx: list[int] | None = Field(default=None, min_length=3, max_length=3)
    synthesis_strategy: str = "ml_volume_generator"
    locked_lesion_planes: list[int] = Field(default_factory=list)


class PoseEstimate(BaseModel):
    """Learned or DICOM-derived slice pose (Phase 6b)."""

    organ_type: OrganType
    through_plane_axis: int = Field(..., ge=0, le=2)
    slice_index_normalized: float = Field(..., ge=0.0, le=1.0)
    mri_view: MriView
    confidence: float = Field(..., ge=0.0, le=1.0)
    source: str = Field(..., description="dicom | ml | heuristic")


class SynthesisResult(BaseModel):
    """Synthetic slice generation outcome (Phase 6)."""

    intensity_volume_path: str | None = None
    label_volume_path: str | None = None
    confidence_volume_path: str | None = None
    real_slices_preserved: int = 0
    synthetic_slices_generated: int = 0
    strategy: str = "measured"
    model_version: str | None = None
    pose_estimate_path: str | None = None


class PlaneMetrics(BaseModel):
    plane_index: int
    view: MriView
    dice: float | None = Field(None, ge=0.0, le=1.0)
    iou: float | None = Field(None, ge=0.0, le=1.0)
    ssim: float | None = Field(None, ge=0.0, le=1.0)
    validated: bool = False


class ConfidenceRegion(BaseModel):
    region_id: str
    label: str
    source: SourceType
    confidence: float = Field(..., ge=0.0, le=1.0)
    mesh_url: str | None = None
    volume_mask_path: str | None = None


class ValidationReport(BaseModel):
    """Re-slice validation and QA (Phase 8)."""

    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    anchor_plane_metrics: list[PlaneMetrics] = Field(default_factory=list)
    qa_passed: bool = True
    qa_messages: list[str] = Field(default_factory=list)
    confidence_regions: list[ConfidenceRegion] = Field(default_factory=list)


class StageTiming(BaseModel):
    stage: str
    duration_ms: float
    status: str = "ok"
    message: str | None = None


class PipelineArtifacts(BaseModel):
    """Full pipeline state persisted per reconstruction job."""

    reconstruction_id: str
    pipeline_version: str = "0.1.0"
    scan_context: ScanContext
    model_outputs: ModelOutputs | None = None
    anatomical_map: AnatomicalMap | None = None
    atlas_warp: AtlasWarpResult | None = None
    blueprint: ReconstructionBlueprint | None = None
    synthesis: SynthesisResult | None = None
    validation: ValidationReport | None = None
    stage_timings: list[StageTiming] = Field(default_factory=list)
