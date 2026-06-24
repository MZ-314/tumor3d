import { useEffect, useState } from "react";

interface AnalysisProgressProps {
  sliceCount: number;
  startedAt: number;
  mode?: string;
}

export function AnalysisProgress({ sliceCount, startedAt, mode = "ai_3d" }: AnalysisProgressProps) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const tick = () => setElapsed(Math.floor((Date.now() - startedAt) / 1000));
    tick();
    const id = window.setInterval(tick, 1000);
    return () => window.clearInterval(id);
  }, [startedAt]);

  return (
    <div className="analysis-progress" role="status">
      <div className="bubble__spinner" aria-hidden>
        <span />
        <span />
        <span />
      </div>
      <p className="analysis-progress__title">
        {mode === "ai_3d"
          ? "Building AI 3D mesh on RunPod GPU…"
          : `Processing ${sliceCount} slice${sliceCount === 1 ? "" : "s"}…`}
      </p>
      <p className="analysis-progress__hint">
        {mode === "ai_3d"
          ? "TripoSR infers a 3D shape from your photo (30s–3min). This tab updates automatically."
          : sliceCount >= 3
            ? "Large DICOM uploads run in the background (often 2–10 min)."
            : "Usually under a minute for a single slice."}
      </p>
      <p className="analysis-progress__elapsed">Elapsed: {elapsed}s</p>
    </div>
  );
}
