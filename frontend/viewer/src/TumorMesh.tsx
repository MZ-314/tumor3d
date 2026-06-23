import { useEffect, useState } from "react";
import { useLoader } from "@react-three/fiber";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import * as THREE from "three";
import { resolveAssetUrl } from "./api";

interface TumorMeshProps {
  meshUrl: string;
  apiBase?: string;
}

function TumorMeshInner({ scene }: { scene: THREE.Group }) {
  return <primitive object={scene} />;
}

export function TumorMesh({ meshUrl, apiBase = "" }: TumorMeshProps) {
  const url = resolveAssetUrl(meshUrl, apiBase);
  const gltf = useLoader(GLTFLoader, url);
  const scene = gltf.scene.clone(true);

  useEffect(() => {
    scene.traverse((child) => {
      if (child instanceof THREE.Mesh && child.material) {
        const mat = child.material as THREE.MeshStandardMaterial;
        mat.metalness = 0.1;
        mat.roughness = 0.65;
      }
    });
  }, [scene]);

  const box = new THREE.Box3().setFromObject(scene);
  const center = box.getCenter(new THREE.Vector3());
  scene.position.sub(center);

  return <TumorMeshInner scene={scene} />;
}

export function useConfidenceSidecar(meshUrl: string, apiBase = ""): number[] | null {
  const [confidence, setConfidence] = useState<number[] | null>(null);

  useEffect(() => {
    const glbUrl = resolveAssetUrl(meshUrl, apiBase);
    const sidecarUrl = glbUrl.replace(/\.glb$/, ".confidence.json");
    fetch(sidecarUrl)
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (data?.per_vertex_confidence) {
          setConfidence(data.per_vertex_confidence as number[]);
        }
      })
      .catch(() => setConfidence(null));
  }, [meshUrl, apiBase]);

  return confidence;
}
