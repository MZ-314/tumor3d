import { Canvas } from "@react-three/fiber";
import { OrbitControls, PerspectiveCamera } from "@react-three/drei";
import { Suspense } from "react";
import type { ReconstructResponse } from "@shared/index";
import { resolveAssetUrl } from "./api";
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
  const sizeMb = (reconstruction.file_size_bytes / (1024 * 1024)).toFixed(1);

  return (
    <div className={`mesh-viewer ${compact ? "mesh-viewer--compact" : ""}`}>
      <div className="mesh-viewer__canvas-wrap">
        <Canvas>
          <PerspectiveCamera makeDefault position={[0, 0, 4]} fov={45} />
          <ambientLight intensity={0.6} />
          <directionalLight position={[4, 6, 8]} intensity={1.1} />
          <directionalLight position={[-3, -2, -4]} intensity={0.35} />
          <Suspense fallback={<LoadingFallback />}>
            <TumorMesh meshUrl={reconstruction.mesh_url} apiBase={apiBase} />
          </Suspense>
          <OrbitControls enablePan enableZoom enableRotate />
        </Canvas>
        <p className="mesh-viewer__hint">Drag to rotate · scroll to zoom</p>
      </div>

      <div className="mesh-viewer__meta">
        <img
          src={sourceUrl}
          alt="Source image"
          className="mesh-viewer__thumbnail"
        />
        <div className="mesh-viewer__stats">
          <div>
            <span className="label">Format</span>
            <span>{reconstruction.mesh_format.toUpperCase()}</span>
          </div>
          <div>
            <span className="label">Size</span>
            <span>{sizeMb} MB</span>
          </div>
          <div>
            <span className="label">Pipeline</span>
            <span>SAM 2 + TRELLIS.2</span>
          </div>
        </div>
        <p className="mesh-viewer__disclaimer">{reconstruction.disclaimer}</p>
      </div>
    </div>
  );
}
