import type {
  ChatDetail,
  ChatMessage,
  ChatSummary,
  ReconstructResponse,
} from "@shared/index";

/** Set in frontend/.env — e.g. https://YOUR_POD_ID-8000.proxy.runpod.net */
export const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, "") ?? "";

export async function reconstructFromFiles(
  files: File[],
  options: {
    apiBase?: string;
    modality?: string;
    chatId?: string;
    text?: string;
  } = {},
): Promise<ReconstructResponse> {
  const apiBase = options.apiBase ?? API_BASE;
  const form = new FormData();
  for (const file of files) {
    form.append("images", file);
  }
  form.append("modality", options.modality ?? "brain_mri");
  if (options.chatId) form.append("chat_id", options.chatId);
  if (options.text) form.append("text", options.text);

  const response = await fetch(`${apiBase}/reconstruct`, {
    method: "POST",
    body: form,
  });

  if (!response.ok) {
    const detail = await response.json().catch(() => ({ detail: response.statusText }));
    const message =
      typeof detail.detail === "string"
        ? detail.detail
        : `Reconstruction failed (${response.status})`;
    throw new Error(message);
  }

  return response.json() as Promise<ReconstructResponse>;
}

/** @deprecated Use reconstructFromFiles */
export async function reconstructFromFile(
  file: File,
  apiBase: string = API_BASE,
): Promise<ReconstructResponse> {
  return reconstructFromFiles([file], { apiBase });
}

export async function listChats(apiBase: string = API_BASE): Promise<ChatSummary[]> {
  const r = await fetch(`${apiBase}/chats`);
  if (!r.ok) throw new Error("Failed to load chats");
  return r.json();
}

export async function createChat(
  title = "New scan",
  apiBase: string = API_BASE,
): Promise<ChatSummary> {
  const r = await fetch(`${apiBase}/chats?title=${encodeURIComponent(title)}`, {
    method: "POST",
  });
  if (!r.ok) throw new Error("Failed to create chat");
  return r.json();
}

export async function getChat(
  chatId: string,
  apiBase: string = API_BASE,
): Promise<ChatDetail> {
  const r = await fetch(`${apiBase}/chats/${chatId}`);
  if (!r.ok) throw new Error("Chat not found");
  return r.json();
}

export function chatRecordToMessage(record: ChatDetail["messages"][number]): ChatMessage {
  return {
    id: record.id,
    role: record.role as ChatMessage["role"],
    text: record.text ?? undefined,
    attachmentUrl: record.attachment_url ?? undefined,
    reconstruction: record.reconstruction ?? undefined,
  };
}

export function resolveAssetUrl(url: string, apiBase: string = API_BASE): string {
  if (url.startsWith("http")) return url;
  return `${apiBase}${url}`;
}

export const ACCURACY_TIER_LABELS: Record<string, string> = {
  single_slice: "Single slice — depth estimated",
  partial_volume: "Partial volume — improving Z",
  multi_slice: "Multi-slice — better depth",
};
