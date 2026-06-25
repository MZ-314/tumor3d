import { Canvas } from "@react-three/fiber";
import { OrbitControls, PerspectiveCamera } from "@react-three/drei";
import { Suspense, type ReactNode } from "react";
import type { ReconstructResponse } from "@shared/index";
import { ACCURACY_TIER_LABELS, API_BASE, resolveAssetUrl } from "./api";
import { NiivueVolumeViewer } from "./NiivueVolumeViewer";
import { SliceTexturedPlane } from "./SliceTexturedPlane";
import { TumorMesh } from "./TumorMesh";
import { ViewerErrorBoundary } from "./ViewerErrorBoundary";
import "./viewer.css";
import "./niivue-viewer.css";

interface MeshViewerProps {
  reconstruction: ReconstructResponse;
  apiBase?: string;
  /** @deprecated use variant */
  compact?: boolean;
  variant?: "default" | "chat";
}

function LoadingFallback() {
  return (
    <mesh>
      <sphereGeometry args={[0.5, 16, 16]} />
      <meshStandardMaterial color="#60a5fa" wireframe />
    </mesh>
  );
}

export function MeshViewer({
  reconstruction,
  apiBase = API_BASE,
  compact = false,
  variant,
}: MeshViewerProps) {
  const layout = variant ?? (compact ? "chat" : "default");
  const sourceUrl = resolveAssetUrl(reconstruction.source_image_url, apiBase);
  const overlayUrl = reconstruction.overlay_image_url
    ? resolveAssetUrl(reconstruction.overlay_image_url, apiBase)
    : null;
  const tierLabel =
    ACCURACY_TIER_LABELS[reconstruction.accuracy_tier] ?? reconstruction.accuracy_tier;
  const useVolume =
    reconstruction.viewer_mode === "volume" && Boolean(reconstruction.volume_nifti_url);
  const sliceCount = reconstruction.slice_count;
  const isAi3d =
    reconstruction.pipeline_type === "ai_3d" || reconstruction.modality === "ai_3d";
  const showSceneMesh =
    Boolean(reconstruction.scene_mesh_url) &&
    (isAi3d || reconstruction.lesions.length > 0 || (reconstruction.modality === "brain_mri" && sliceCount <= 1));
  const slicePreviewOnly = !useVolume && !showSceneMesh;
  const thinVolume = !isAi3d && sliceCount < 10;
  const volumeOnly =
    reconstruction.segmentation_backend === "volume_only" ||
    reconstruction.modality === "volume_mri" ||
    reconstruction.modality === "knee_mri";
  const volumeLabel = volumeOnly ? "3D volume" : "brain volume";

  let depthBanner: ReactNode = null;
  if (isAi3d) {
    depthBanner = (
      <div className="volume-depth-banner volume-depth-banner--ai">
        <strong>AI-generated 3D mesh</strong>
        <p>
          Shape inferred from your 2D image — not a real CT/MRI volume. Rotate the model; interior
          detail is guessed by the AI.
        </p>
      </div>
    );
  } else if (thinVolume && reconstruction.modality === "brain_mri" && sliceCount <= 1) {
    depthBanner = (
      <div className="volume-depth-banner volume-depth-banner--ai">
        <strong>AI-predicted 3D brain (1 slice)</strong>
        <p>
          Your DICOM slice is the anchor — surrounding anatomy is estimated by AI + atlas.
          Treat off-slice detail as a clinical estimate, not measured volume.
        </p>
      </div>
    );
  } else if (thinVolume) {
    depthBanner = (
      <div
        className={`volume-depth-banner${sliceCount <= 1 ? " volume-depth-banner--critical" : ""}`}
      >
        {sliceCount <= 1 ? (
          <>
            <strong>Only 1 slice — not a full {volumeLabel}.</strong>
            <p>
              The 3D panel is that single MRI sheet. Upload all DICOM slices from the same study
              (use <strong>📁 folder</strong>).
            </p>
          </>
        ) : (
          <>
            <strong>Partial stack ({sliceCount} slices).</strong>
            <p>
              Upload more slices from the same series for a fuller {volumeLabel}.
              {volumeOnly && " Volume mode skips tumor AI — scan stack only."}
            </p>
          </>
        )}
      </div>
    );
  } else if (volumeOnly) {
    depthBanner = (
      <div className="volume-depth-banner">
        <strong>DICOM volume viewer</strong>
        <p>Real scan stack — drag the crosshair in the 3D panel to slice through your data.</p>
      </div>
    );
  }

  const meta = (
    <>
      <img src={sourceUrl} alt="Source slice" className="mesh-viewer__thumbnail" />
      {overlayUrl && (
        <>
          <p className="mesh-viewer__overlay-label">
            Segmentation overlay (check this matches the tumor)
          </p>
          <img
            src={overlayUrl}
            alt="Segmentation overlay"
            className="mesh-viewer__thumbnail mesh-viewer__thumbnail--overlay"
          />
        </>
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
            <span className="label">Pipeline</span>
            <span>
              {isAi3d
                ? "AI 3D"
                : useVolume
                  ? "DICOM volume"
                  : reconstruction.modality === "brain_mri"
                    ? "Brain tumor"
                    : "Mesh"}
            </span>
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
        {reconstruction.lesions.length === 0 && (
          <li className="mesh-viewer__no-lesion">
            {useVolume && reconstruction.modality === "brain_mri" && sliceCount <= 1
              ? "No tumor mask on this slice — volume shows AI-predicted brain anatomy from your upload."
              : useVolume
                ? "No tumor mask — volume viewer shows your scan; upload more DICOM slices for better 3D."
                : "No tumor region detected — 3D view shows the AI-predicted brain mesh."}
          </li>
        )}
        {reconstruction.lesions.map((lesion, i) => {
          const c = lesion.centroid_mm;
          const vol = lesion.volume_mm3;
          if (!c || !vol) return null;
          return (
            <li key={lesion.lesion_id}>
              <strong>Lesion {i + 1}</strong>
              <span>
                ({c.x.toFixed(1)}, {c.y.toFixed(1)}, {c.z.toFixed(1)}) mm
              </span>
              <span className="mesh-viewer__conf">
                in-plane {Math.round((lesion.in_plane_confidence ?? 0) * 100)}% · depth{" "}
                {Math.round((lesion.depth_confidence ?? 0) * 100)}%
              </span>
              <span className="mesh-viewer__vol">
                ~{vol.value.toFixed(0)} mm³ ({vol.source})
              </span>
            </li>
          );
        })}
      </ul>

      <p className="mesh-viewer__disclaimer">{reconstruction.disclaimer}</p>
    </>
  );

  return (
    <div className={`mesh-viewer mesh-viewer--${layout}`}>
      {depthBanner}
      <div className="mesh-viewer__canvas-wrap">
        <ViewerErrorBoundary>
          {useVolume ? (
            <NiivueVolumeViewer
              volumeUrl={reconstruction.volume_nifti_url!}
              maskUrl={reconstruction.tumor_mask_nifti_url}
              apiBase={apiBase}
              sliceCount={sliceCount}
            />
          ) : (
            <Canvas>
              <PerspectiveCamera makeDefault position={[0, 0, 4]} fov={45} />
              <ambientLight intensity={0.6} />
              <directionalLight position={[4, 6, 8]} intensity={1.1} />
              <directionalLight position={[-3, -2, -4]} intensity={0.35} />
              <Suspense fallback={<LoadingFallback />}>
                {slicePreviewOnly ? (
                  <SliceTexturedPlane imageUrl={sourceUrl} overlayUrl={overlayUrl} />
                ) : (
                  <TumorMesh
                    meshUrl={reconstruction.scene_mesh_url}
                    apiBase={apiBase}
                    variant={isAi3d ? "ai" : "tumor"}
                  />
                )}
              </Suspense>
              <OrbitControls enablePan enableZoom enableRotate />
            </Canvas>
          )}
        </ViewerErrorBoundary>
        {!useVolume && (
          <p className="mesh-viewer__hint">
            {slicePreviewOnly
              ? "Rotate the MRI slice in 3D · scroll to zoom"
              : isAi3d
                ? "Drag to rotate the AI mesh · scroll to zoom"
                : "Drag to rotate · scroll to zoom"}
          </p>
        )}
      </div>

      {layout === "chat" ? (
        <details className="mesh-viewer__meta-details">
          <summary>Scan details & overlays</summary>
          <div className="mesh-viewer__meta">{meta}</div>
        </details>
      ) : (
        <div className="mesh-viewer__meta">{meta}</div>
      )}
    </div>
  );
}
