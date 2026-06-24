/**
 * Meddollina embed API — web component approach (not iframe).
 *
 * Tradeoff: custom element shares the host page's JS context (better perf,
 * simpler props) but requires CSS scoping (shadow DOM) to avoid style clashes.
 * Iframe would give stronger isolation at the cost of messaging overhead.
 */
import { createRoot, type Root } from "react-dom/client";
import { StrictMode } from "react";
import type { ReconstructResponse } from "@shared/index";
import { reconstructFromFile } from "@viewer/api";
import { MeshViewer } from "@viewer/MeshViewer";
import "@viewer/viewer.css";
import "./plugin.css";

export interface MountOptions {
  imageUrl: string;
  apiBase?: string;
  pixelSpacingMm?: number;
  onComplete?: (result: ReconstructResponse) => void;
  onError?: (error: Error) => void;
}

interface PluginViewProps {
  loading: boolean;
  error?: string;
  reconstruction?: ReconstructResponse;
  apiBase: string;
}

function PluginView({ loading, error, reconstruction, apiBase }: PluginViewProps) {
  if (loading) {
    return <p className="plugin-loading">Analyzing scan…</p>;
  }
  if (error) {
    return <p className="plugin-error">{error}</p>;
  }
  if (reconstruction) {
    return <MeshViewer reconstruction={reconstruction} apiBase={apiBase} />;
  }
  return null;
}

const roots = new WeakMap<HTMLElement, Root>();

export async function mountTumorViewer(
  container: HTMLElement,
  opts: MountOptions,
): Promise<void> {
  const apiBase = opts.apiBase ?? "";
  const root = createRoot(container);
  roots.set(container, root);

  root.render(
    <StrictMode>
      <PluginView loading apiBase={apiBase} />
    </StrictMode>,
  );

  try {
    const response = await fetch(opts.imageUrl);
    if (!response.ok) {
      throw new Error(`Failed to fetch image (${response.status})`);
    }
    const blob = await response.blob();
    const name = opts.imageUrl.split("/").pop() ?? "scan.png";
    const file = new File([blob], name, { type: blob.type || "image/png" });

    const reconstruction = await reconstructFromFile(file, apiBase);
    opts.onComplete?.(reconstruction);

    root.render(
      <StrictMode>
        <PluginView reconstruction={reconstruction} apiBase={apiBase} loading={false} />
      </StrictMode>,
    );
  } catch (err) {
    const error = err instanceof Error ? err : new Error("Reconstruction failed");
    opts.onError?.(error);
    root.render(
      <StrictMode>
        <PluginView loading={false} error={error.message} apiBase={apiBase} />
      </StrictMode>,
    );
  }
}

export function unmountTumorViewer(container: HTMLElement): void {
  const root = roots.get(container);
  if (root) {
    root.unmount();
    roots.delete(container);
  }
}

export { MeshViewer, reconstructFromFile };
export type { ReconstructResponse };
