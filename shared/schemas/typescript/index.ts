export type ChatRole = "user" | "assistant" | "system";

export interface ReconstructResponse {
  reconstruction_id: string;
  mesh_url: string;
  source_image_url: string;
  isolated_image_url: string | null;
  mesh_format: string;
  file_size_bytes: number;
  pipeline: string;
  assistant_summary: string;
  disclaimer: string;
}

export interface ChatMessage {
  id: string;
  role: ChatRole;
  text?: string;
  attachmentUrl?: string;
  attachmentName?: string;
  reconstruction?: ReconstructResponse;
  loading?: boolean;
  error?: string;
}
