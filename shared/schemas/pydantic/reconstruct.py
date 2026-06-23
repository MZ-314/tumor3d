from pydantic import BaseModel, Field


class ReconstructResponse(BaseModel):
    reconstruction_id: str
    mesh_url: str
    source_image_url: str
    isolated_image_url: str | None = None
    mesh_format: str = "glb"
    file_size_bytes: int = Field(..., ge=0)
    pipeline: str = "sam2_trellis2_blender"
    assistant_summary: str
    disclaimer: str = (
        "This 3D model is AI-generated from a single 2D image. "
        "Unseen sides are inferred, not photographed."
    )
