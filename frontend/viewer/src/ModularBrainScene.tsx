import { useEffect, useMemo, useState } from "react";
import { useLoader } from "@react-three/fiber";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import * as THREE from "three";
import type { AnatomicalModule } from "@shared/index";
import { resolveAssetUrl } from "./api";

const SOURCE_COLORS: Record<string, string> = {
  measured: "#4ade80",
  inference: "#60a5fa",
};

interface ModularBrainSceneProps {
  meshUrl: string;
  modules?: AnatomicalModule[];
  apiBase?: string;
  visibleModules: Set<string>;
}

function colorForModule(mod: AnatomicalModule | undefined): string {
  if (!mod) return SOURCE_COLORS.inference;
  if (mod.anchor_locked || mod.geometry_source === "measured") {
    return SOURCE_COLORS.measured;
  }
  if (mod.morph_applied) return "#fbbf24";
  return SOURCE_COLORS[mod.geometry_source] ?? SOURCE_COLORS.inference;
}

function ModularBrainSceneInner({
  scene,
  modules,
  visibleModules,
}: {
  scene: THREE.Group;
  modules: AnatomicalModule[];
  visibleModules: Set<string>;
}) {
  const moduleById = useMemo(() => {
    const map = new Map<string, AnatomicalModule>();
    for (const m of modules) map.set(m.module_id, m);
    return map;
  }, [modules]);

  useEffect(() => {
    scene.traverse((child) => {
      if (!(child instanceof THREE.Mesh) || !child.material) return;
      const nodeName = child.name || child.parent?.name || "";
      const mod = moduleById.get(nodeName);
      const visible =
        visibleModules.size === 0 ||
        visibleModules.has(nodeName) ||
        nodeName === "" ||
        nodeName === "Scene";
      child.visible = visible;
      const mat = child.material as THREE.MeshStandardMaterial;
      const hex = colorForModule(mod);
      mat.color = new THREE.Color(hex);
      mat.metalness = 0.1;
      mat.roughness = 0.65;
      if (mod?.morph_applied) {
        mat.emissive = new THREE.Color("#3d2e00");
        mat.emissiveIntensity = 0.12;
      }
    });
  }, [scene, moduleById, visibleModules]);

  return <primitive object={scene} />;
}

export function ModularBrainScene({
  meshUrl,
  modules = [],
  apiBase = "",
  visibleModules,
}: ModularBrainSceneProps) {
  const url = resolveAssetUrl(meshUrl, apiBase);
  const gltf = useLoader(GLTFLoader, url, (loader) => {
    loader.setCrossOrigin("anonymous");
  });
  const scene = gltf.scene.clone(true);

  const box = new THREE.Box3().setFromObject(scene);
  const center = box.getCenter(new THREE.Vector3());
  scene.position.sub(center);

  return (
    <ModularBrainSceneInner scene={scene} modules={modules} visibleModules={visibleModules} />
  );
}

export function useModularVisibility(modules: AnatomicalModule[] = []) {
  const [visible, setVisible] = useState<Set<string>>(() => new Set(modules.map((m) => m.module_id)));

  useEffect(() => {
    setVisible(new Set(modules.map((m) => m.module_id)));
  }, [modules]);

  const toggle = (id: string) => {
    setVisible((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  return { visible, toggle, setVisible };
}
