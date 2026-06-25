export type SourceType = "measured" | "inference";
export type AccuracyTier = "single_slice" | "partial_volume" | "multi_slice";
export type MriView = "axial" | "coronal" | "sagittal" | "unknown";
export type OrganType = "brain" | "knee" | "other" | "unknown";
export type InputSource = "dicom" | "image" | "montage";

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

export interface AnatomicalModule {
  module_id: string;
  display_name: string;
  mesh_path: string;
  transform_4x4: number[][];
  geometry_source: SourceType;
  confidence: number;
  anchor_locked: boolean;
  morph_applied: boolean;
  connects_to: string[];
  metadata_path?: string | null;
}

export interface ModuleGraphEdge {
  source_id: string;
  target_id: string;
  relation: string;
}

export interface ModuleGraph {
  nodes: string[];
  edges: ModuleGraphEdge[];
}

export interface ModuleAssemblyResult {
  root_glb_path: string;
  tumor_glb_path?: string | null;
  module_manifest_path?: string | null;
  modules: AnatomicalModule[];
  graph?: ModuleGraph | null;
}

export interface ReconstructResponse {
  reconstruction_id: string;
  chat_id: string | null;
  source_image_url: string;
  overlay_image_url: string | null;
  scene_mesh_url: string;
  volume_nifti_url?: string | null;
  tumor_mask_nifti_url?: string | null;
  module_manifest_url?: string | null;
  modules?: AnatomicalModule[];
  explorer_mode?: "modular" | "volume" | "legacy";
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
  pipeline_artifacts?: PipelineArtifacts | null;
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

export interface SliceSpacing {
  row_mm: number;
  col_mm: number;
  slice_mm: number;
  source: SourceType;
}

export interface ScanContext {
  reconstruction_id: string;
  input_source: InputSource;
  organ_type: OrganType;
  modality: string;
  mri_view: MriView;
  accuracy_tier: AccuracyTier;
  slice_count: number;
  anchor_slice_indices: number[];
  slice_spacing_mm?: SliceSpacing | null;
  volume_shape_zyx?: number[] | null;
  body_part_examined?: string | null;
  series_description?: string | null;
  montage_panels?: number | null;
  quality_score: number;
  warnings: string[];
}

export interface ConfidenceRegion {
  region_id: string;
  label: string;
  source: SourceType;
  confidence: number;
  mesh_url?: string | null;
  volume_mask_path?: string | null;
}

export interface ValidationReport {
  overall_confidence: number;
  anchor_plane_metrics: unknown[];
  qa_passed: boolean;
  qa_messages: string[];
  confidence_regions: ConfidenceRegion[];
}

export interface StageTiming {
  stage: string;
  duration_ms: number;
  status: string;
  message?: string | null;
}

export interface PipelineArtifacts {
  reconstruction_id: string;
  pipeline_version: string;
  scan_context: ScanContext;
  model_outputs?: unknown | null;
  anatomical_map?: unknown | null;
  atlas_warp?: unknown | null;
  blueprint?: unknown | null;
  synthesis?: unknown | null;
  module_assembly?: ModuleAssemblyResult | null;
  validation?: ValidationReport | null;
  stage_timings: StageTiming[];
}
