export type SourceType = "measured" | "inference";
export type AccuracyTier = "single_slice" | "partial_volume" | "multi_slice";

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface BoundingBox2D {
  min_row: number;
  min_col: number;
  max_row: number;
  max_col: number;
}

export interface BoundingBox3D {
  min_x: number;
  min_y: number;
  min_z: number;
  max_x: number;
  max_y: number;
  max_z: number;
}

export interface ProvenanceField {
  value: number;
  confidence: number;
  source: SourceType;
}

export interface LesionResult {
  lesion_id: string;
  mesh_url: string;
  centroid_mm: Vec3;
  bounding_box_2d: BoundingBox2D;
  bounding_box_3d_mm: BoundingBox3D;
  volume_mm3: ProvenanceField;
  in_plane_confidence: number;
  depth_confidence: number;
  vertices: number[][];
}

export interface ReconstructResponse {
  reconstruction_id: string;
  chat_id: string | null;
  source_image_url: string;
  overlay_image_url: string | null;
    scene_mesh_url: string;
    volume_nifti_url?: string | null;
    tumor_mask_nifti_url?: string | null;
    viewer_mode?: string;
    pipeline_type?: string;
    geometry_source?: string;
    mesh_format: string;
  slice_count: number;
  accuracy_tier: AccuracyTier;
  modality: string;
  segmentation_backend: string;
  lesions: LesionResult[];
  assistant_summary: string;
  disclaimer: string;
}

export interface ChatSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface ChatMessageRecord {
  id: string;
  role: string;
  text?: string | null;
  attachment_url?: string | null;
  reconstruction?: ReconstructResponse | null;
  created_at: string;
}

export interface ChatDetail {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  messages: ChatMessageRecord[];
}

export type ChatRole = "user" | "assistant" | "system";

export interface ChatMessage {
  id: string;
  role: ChatRole;
  text?: string;
  attachmentUrl?: string;
  attachmentName?: string;
  reconstruction?: ReconstructResponse;
  loading?: boolean;
  error?: string;
  /** Set while waiting on /reconstruct */
  analysisSliceCount?: number;
  analysisStartedAt?: number;
  analysisMode?: string;
}
