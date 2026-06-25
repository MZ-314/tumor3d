from pydantic import BaseModel, Field

from shared.schemas.pydantic.common import (
    AccuracyTier,
    BoundingBox2D,
    BoundingBox3D,
    ProvenanceField,
    SourceType,
    Vec3,
)
from shared.schemas.pydantic.pipeline import PipelineArtifacts, AnatomicalModule


class LesionResult(BaseModel):
    lesion_id: str
    mesh_url: str
    centroid_mm: Vec3
    bounding_box_2d: BoundingBox2D
    bounding_box_3d_mm: BoundingBox3D
    volume_mm3: ProvenanceField
    in_plane_confidence: float = Field(..., ge=0.0, le=1.0)
    depth_confidence: float = Field(..., ge=0.0, le=1.0)
    vertices: list[list[float]] = Field(
        default_factory=list,
        description="Mesh vertices [[x,y,z], ...] in mm (may be decimated)",
    )


class ReconstructResponse(BaseModel):
    reconstruction_id: str
    chat_id: str | None = None
    source_image_url: str
    overlay_image_url: str | None = None
    scene_mesh_url: str
    volume_nifti_url: str | None = None
    tumor_mask_nifti_url: str | None = None
    module_manifest_url: str | None = None
    modules: list["AnatomicalModule"] = Field(default_factory=list)
    explorer_mode: str = "legacy"
    viewer_mode: str = "mesh"
    mesh_format: str = "glb"
    pipeline_type: str = "medical"
    geometry_source: str = "measured"
    slice_count: int = Field(..., ge=1)
    accuracy_tier: AccuracyTier
    modality: str
    segmentation_backend: str
    lesions: list[LesionResult]
    assistant_summary: str
    disclaimer: str = (
        "Tumor location on the slice is model-inferred. Depth and volume improve with "
        "more slices. Not for diagnosis."
    )
    pipeline_artifacts: PipelineArtifacts | None = None


class ChatSummary(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str


class ChatMessageRecord(BaseModel):
    id: str
    role: str
    text: str | None = None
    attachment_url: str | None = None
    reconstruction: ReconstructResponse | None = None
    created_at: str


class ChatDetail(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[ChatMessageRecord]
