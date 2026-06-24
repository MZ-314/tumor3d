import type {
  ChatDetail,
  ChatMessage,
  ChatSummary,
  ReconstructResponse,
} from "@shared/index";

/** Set in frontend/.env — e.g. https://YOUR_POD_ID-8000.proxy.runpod.net */
export const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, "") ?? "";

const POLL_INTERVAL_MS = 2000;
const POLL_TIMEOUT_MS = 15 * 60 * 1000;

interface ReconstructJobPoll {
  status: string;
  job_id?: string;
  detail?: string;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function pollReconstructJob(
  apiBase: string,
  jobId: string,
): Promise<ReconstructResponse> {
  const deadline = Date.now() + POLL_TIMEOUT_MS;

  while (Date.now() < deadline) {
    await sleep(POLL_INTERVAL_MS);

    let poll: Response;
    try {
      poll = await fetch(`${apiBase}/reconstruct/jobs/${jobId}`);
    } catch {
      continue;
    }

    if (!poll.ok) {
      throw new Error(`Job status check failed (${poll.status})`);
    }

    const body = (await poll.json()) as ReconstructJobPoll & Partial<ReconstructResponse>;
    if (body.status === "processing") {
      continue;
    }
    if (body.status === "error") {
      throw new Error(body.detail ?? "GPU processing failed");
    }
    return body as ReconstructResponse;
  }

  throw new Error(
    "GPU processing timed out after 15 minutes. Check the RunPod uvicorn terminal — the job may still be running.",
  );
}

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

  let response: Response;
  try {
    response = await fetch(`${apiBase}/reconstruct`, {
      method: "POST",
      body: form,
    });
  } catch (err) {
    const lost =
      err instanceof TypeError &&
      (err.message === "Failed to fetch" || err.message.includes("NetworkError"));
    if (lost) {
      throw new Error(
        "Connection lost while uploading or waiting for GPU segmentation. " +
          "22+ DICOM slices can take 3–10 minutes. Check the RunPod uvicorn terminal for progress. " +
          "If DICOM errors mention JPEG decompression, run on the pod: " +
          "pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg",
      );
    }
    throw err;
  }

  if (response.status === 202) {
    const started = (await response.json()) as { job_id: string };
    return pollReconstructJob(apiBase, started.job_id);
  }

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

export interface HealthResponse {
  status: string;
  pipeline: string;
  segmentation_backend: string;
}

export async function fetchHealth(apiBase: string = API_BASE): Promise<HealthResponse> {
  const r = await fetch(`${apiBase}/health`);
  if (!r.ok) throw new Error("Health check failed");
  return r.json() as Promise<HealthResponse>;
}
