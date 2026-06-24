from enum import Enum

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    MEASURED = "measured"
    INFERENCE = "inference"


class AccuracyTier(str, Enum):
    SINGLE_SLICE = "single_slice"
    PARTIAL_VOLUME = "partial_volume"
    MULTI_SLICE = "multi_slice"


class BoundingBox3D(BaseModel):
    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float


class BoundingBox2D(BaseModel):
    min_row: int
    min_col: int
    max_row: int
    max_col: int


class ProvenanceField(BaseModel):
    value: float
    confidence: float = Field(..., ge=0.0, le=1.0)
    source: SourceType


class Vec3(BaseModel):
    x: float
    y: float
    z: float
