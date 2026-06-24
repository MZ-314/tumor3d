import { useEffect, useRef } from "react";
import { Niivue } from "@niivue/niivue";
import { resolveAssetUrl } from "./api";
import "./niivue-viewer.css";

interface NiivueVolumeViewerProps {
  volumeUrl: string;
  maskUrl?: string | null;
  apiBase?: string;
}

export function NiivueVolumeViewer({ volumeUrl, maskUrl, apiBase = "" }: NiivueVolumeViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const nvRef = useRef<Niivue | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const nv = new Niivue({
      backColor: [0.04, 0.06, 0.1, 1],
      show3Dcrosshair: true,
      crosshairColor: [0.3, 0.7, 1, 0.8],
      isRadiologicalConvention: false,
    });
    nv.attachToCanvas(canvas);
    nvRef.current = nv;

    const volUrl = resolveAssetUrl(volumeUrl, apiBase);
    const volumes: Parameters<Niivue["loadVolumes"]>[0] = [{ url: volUrl, name: "MRI" }];

    if (maskUrl) {
      volumes.push({
        url: resolveAssetUrl(maskUrl, apiBase),
        name: "Tumor",
        colormap: "red",
        opacity: 0.55,
        cal_min: 0.5,
        cal_max: 5,
      });
    }

    void nv.loadVolumes(volumes).then(() => {
      // Multiplanar: axial / coronal / sagittal + drag crosshair to slice through volume.
      nv.setSliceType(nv.sliceTypeMultiplanar);
      nv.setMultiplanarLayout(2);
      nv.updateGLVolume();
    });

    return () => {
      nvRef.current = null;
    };
  }, [volumeUrl, maskUrl, apiBase]);

  return (
    <div className="niivue-wrap">
      <canvas ref={canvasRef} className="niivue-canvas" />
      <p className="niivue-hint">
        Drag crosshair to slice · scroll to zoom · right panel = 3D volume
      </p>
    </div>
  );
}
