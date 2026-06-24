import { Canvas } from "@react-three/fiber";
import { OrbitControls, PerspectiveCamera } from "@react-three/drei";
import { Suspense } from "react";
import type { ReconstructResponse } from "@shared/index";
import { ACCURACY_TIER_LABELS, resolveAssetUrl } from "./api";
import { TumorMesh } from "./TumorMesh";
import "./viewer.css";

interface MeshViewerProps {
  reconstruction: ReconstructResponse;
  apiBase?: string;
  compact?: boolean;
}

function LoadingFallback() {
  return (
    <mesh>
      <sphereGeometry args={[0.5, 16, 16]} />
      <meshStandardMaterial color="#60a5fa" wireframe />
    </mesh>
  );
}

export function MeshViewer({ reconstruction, apiBase = "", compact = false }: MeshViewerProps) {
  const sourceUrl = resolveAssetUrl(reconstruction.source_image_url, apiBase);
  const overlayUrl = reconstruction.overlay_image_url
    ? resolveAssetUrl(reconstruction.overlay_image_url, apiBase)
    : null;
  const tierLabel =
    ACCURACY_TIER_LABELS[reconstruction.accuracy_tier] ?? reconstruction.accuracy_tier;

  return (
    <div className={`mesh-viewer ${compact ? "mesh-viewer--compact" : ""}`}>
      <div className="mesh-viewer__canvas-wrap">
        <Canvas>
          <PerspectiveCamera makeDefault position={[0, 0, 4]} fov={45} />
          <ambientLight intensity={0.6} />
          <directionalLight position={[4, 6, 8]} intensity={1.1} />
          <directionalLight position={[-3, -2, -4]} intensity={0.35} />
          <Suspense fallback={<LoadingFallback />}>
            <TumorMesh meshUrl={reconstruction.scene_mesh_url} apiBase={apiBase} />
          </Suspense>
          <OrbitControls enablePan enableZoom enableRotate />
        </Canvas>
        <p className="mesh-viewer__hint">Drag to rotate · scroll to zoom</p>
      </div>

      <div className="mesh-viewer__meta">
        <img src={sourceUrl} alt="Source slice" className="mesh-viewer__thumbnail" />
        {overlayUrl && (
          <img src={overlayUrl} alt="Segmentation overlay" className="mesh-viewer__thumbnail" />
        )}
        <div className="mesh-viewer__stats">
          <div>
            <span className="label">Slices</span>
            <span>{reconstruction.slice_count}</span>
          </div>
          <div>
            <span className="label">Accuracy</span>
            <span>{tierLabel}</span>
          </div>
          <div>
            <span className="label">Backend</span>
            <span>{reconstruction.segmentation_backend}</span>
          </div>
          <div>
            <span className="label">Lesions</span>
            <span>{reconstruction.lesions.length}</span>
          </div>
        </div>

        <ul className="mesh-viewer__lesions">
          {reconstruction.lesions.map((lesion, i) => {
            const c = lesion.centroid_mm;
            return (
              <li key={lesion.lesion_id}>
                <strong>Lesion {i + 1}</strong>
                <span>
                  ({c.x.toFixed(1)}, {c.y.toFixed(1)}, {c.z.toFixed(1)}) mm
                </span>
                <span className="mesh-viewer__conf">
                  in-plane {Math.round(lesion.in_plane_confidence * 100)}% · depth{" "}
                  {Math.round(lesion.depth_confidence * 100)}%
                </span>
                <span className="mesh-viewer__vol">
                  ~{lesion.volume_mm3.value.toFixed(0)} mm³ ({lesion.volume_mm3.source})
                </span>
              </li>
            );
          })}
        </ul>

        <p className="mesh-viewer__disclaimer">{reconstruction.disclaimer}</p>
      </div>
    </div>
  );
}
