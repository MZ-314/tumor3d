import type { ReconstructResponse } from "@shared/index";

const DEFAULT_API_BASE = "";

export async function reconstructFromFile(
  file: File,
  apiBase: string = DEFAULT_API_BASE,
  pixelSpacingMm?: number,
): Promise<ReconstructResponse> {
  const form = new FormData();
  form.append("image", file);
  if (pixelSpacingMm !== undefined) {
    form.append("pixel_spacing_mm", String(pixelSpacingMm));
  }

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

export function resolveAssetUrl(url: string, apiBase: string = DEFAULT_API_BASE): string {
  if (url.startsWith("http")) return url;
  return `${apiBase}${url}`;
}
