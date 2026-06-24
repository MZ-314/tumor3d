import { useTexture } from "@react-three/drei";
import { useLayoutEffect, useMemo } from "react";
import * as THREE from "three";

interface SliceTexturedPlaneProps {
  imageUrl: string;
  overlayUrl?: string | null;
}

function planeSize(img: HTMLImageElement | ImageBitmap, max = 2.4): [number, number] {
  const iw = img.width || 1;
  const ih = img.height || 1;
  const aspect = iw / ih;
  if (aspect >= 1) return [max, max / aspect];
  return [max * aspect, max];
}

export function SliceTexturedPlane({ imageUrl, overlayUrl }: SliceTexturedPlaneProps) {
  const textures = useTexture(overlayUrl ? [imageUrl, overlayUrl] : imageUrl);
  const baseTex = (Array.isArray(textures) ? textures[0] : textures) as THREE.Texture;
  const overlayTex = Array.isArray(textures) ? (textures[1] as THREE.Texture) : null;

  useLayoutEffect(() => {
    baseTex.colorSpace = THREE.SRGBColorSpace;
    baseTex.minFilter = THREE.LinearFilter;
    baseTex.magFilter = THREE.LinearFilter;
    if (overlayTex) {
      overlayTex.colorSpace = THREE.SRGBColorSpace;
      overlayTex.minFilter = THREE.LinearFilter;
      overlayTex.magFilter = THREE.LinearFilter;
    }
  }, [baseTex, overlayTex]);

  const [width, height] = useMemo(
    () => planeSize(baseTex.image as HTMLImageElement),
    [baseTex],
  );

  return (
    <group>
      <mesh>
        <planeGeometry args={[width, height]} />
        <meshStandardMaterial
          map={baseTex}
          side={THREE.DoubleSide}
          roughness={0.95}
          metalness={0}
        />
      </mesh>
      {overlayTex && (
        <mesh position={[0, 0, 0.002]}>
          <planeGeometry args={[width, height]} />
          <meshStandardMaterial
            map={overlayTex}
            side={THREE.DoubleSide}
            transparent
            opacity={0.88}
            depthWrite={false}
            roughness={1}
            metalness={0}
          />
        </mesh>
      )}
    </group>
  );
}
