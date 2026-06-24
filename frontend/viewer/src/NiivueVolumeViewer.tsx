import { useEffect, useRef, useState } from "react";
import { Niivue } from "@niivue/niivue";
import { resolveAssetUrl } from "./api";
import "./niivue-viewer.css";

interface NiivueVolumeViewerProps {
  volumeUrl: string;
  maskUrl?: string | null;
  apiBase?: string;
  sliceCount?: number;
}

type ViewerStatus = "loading" | "ready" | "error";

export function NiivueVolumeViewer({
  volumeUrl,
  maskUrl,
  apiBase = "",
  sliceCount = 1,
}: NiivueVolumeViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const nvRef = useRef<Niivue | null>(null);
  const [status, setStatus] = useState<ViewerStatus>("loading");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    let cancelled = false;
    setStatus("loading");
    setErrorMsg(null);

    const nv = new Niivue({
      backColor: [0.04, 0.06, 0.1, 1],
      show3Dcrosshair: true,
      crosshairColor: [0.3, 0.7, 1, 0.8],
      isRadiologicalConvention: false,
    });

    const resize = () => {
      nv.resizeListener();
    };

    const observer = new ResizeObserver(resize);
    observer.observe(container);

    async function init() {
      const canvasEl = canvas;
      if (!canvasEl) return;

      const volUrl = resolveAssetUrl(volumeUrl, apiBase);

      try {
        const probe = await fetch(volUrl, { method: "HEAD" });
        if (!probe.ok) {
          throw new Error(
            `Volume file missing (${probe.status}). On RunPod run: git pull && pip install -e ".[dev,gpu,dicom]" then restart uvicorn.`,
          );
        }
      } catch (err) {
        if (err instanceof Error && err.message.includes("Volume file missing")) {
          if (!cancelled) {
            setStatus("error");
            setErrorMsg(err.message);
          }
          return;
        }
        // HEAD may fail on some proxies; continue to loadVolumes.
      }

      try {
        await nv.attachToCanvas(canvasEl);
        if (cancelled) return;

        const volumes: Parameters<Niivue["loadVolumes"]>[0] = [
          { url: volUrl, colormap: "gray" },
        ];

        if (maskUrl) {
          volumes.push({
            url: resolveAssetUrl(maskUrl, apiBase),
            colormap: "red",
            opacity: 0.55,
            cal_min: 0.5,
            cal_max: 5,
          });
        }

        await nv.loadVolumes(volumes);
        if (cancelled) return;

        nv.setSliceType(nv.sliceTypeMultiplanar);
        nv.setMultiplanarLayout(2);
        nv.updateGLVolume();
        resize();

        nvRef.current = nv;
        setStatus("ready");
      } catch (err) {
        if (cancelled) return;
        const message =
          err instanceof Error ? err.message : "Could not load NIfTI volume in browser";
        setStatus("error");
        setErrorMsg(message);
        nv.cleanup();
      }
    }

    void init();

    return () => {
      cancelled = true;
      observer.disconnect();
      nvRef.current?.cleanup();
      nvRef.current = null;
    };
  }, [volumeUrl, maskUrl, apiBase]);

  return (
    <div ref={containerRef} className="niivue-wrap">
      {status === "loading" && (
        <div className="niivue-status" role="status">
          <span className="niivue-status__spinner" aria-hidden />
          Loading MRI volume…
        </div>
      )}
      {status === "error" && (
        <p className="niivue-error">
          {errorMsg ?? "Volume viewer failed."} Check RunPod is running and the API has the latest
          code (NIfTI export).
        </p>
      )}
      <canvas
        ref={canvasRef}
        className={`niivue-canvas${status === "ready" ? " niivue-canvas--ready" : ""}`}
      />
      {status === "ready" && (
        <p className="niivue-hint">
          {sliceCount < 10
            ? "Thin stack — drag crosshair to move slice planes (upload more DICOM for full brain)"
            : "Drag crosshair to slice · scroll to zoom · right panel = 3D volume"}
        </p>
      )}
    </div>
  );
}
