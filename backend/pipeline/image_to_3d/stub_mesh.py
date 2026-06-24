"""CPU fallback: relief mesh from a single grayscale image (dev / no GPU)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from skimage import measure


def build_relief_mesh_glb(image_path: Path, out_glb: Path, *, depth_voxels: int = 32) -> Path:
    """Extrude image intensity into a depth volume and marching-cubes a mesh."""
    with Image.open(image_path) as img:
        gray = np.asarray(img.convert("L"), dtype=np.float32) / 255.0

    h, w = gray.shape
    max_side = 192
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        gray = np.asarray(Image.fromarray((gray * 255).astype(np.uint8)).resize((new_w, new_h))) / 255.0
        h, w = gray.shape

    # Smooth heightmap
    from scipy.ndimage import gaussian_filter

    height = gaussian_filter(gray, sigma=1.2)
    height = (height - height.min()) / (height.max() - height.min() + 1e-6)

    vol = np.repeat(height[np.newaxis, ...], depth_voxels, axis=0)
    z_ramp = np.linspace(0.85, 1.0, depth_voxels, dtype=np.float32)[:, None, None]
    vol = vol * z_ramp

    if float(vol.max() - vol.min()) < 1e-4:
        mesh = trimesh.creation.box(extents=(1.6, 1.6, 0.35))
        mesh.export(out_glb)
        return out_glb

    level = float(vol.min() + (vol.max() - vol.min()) * 0.45)
    try:
        verts, faces, _, _ = measure.marching_cubes(vol, level=level)
    except (ValueError, RuntimeError) as exc:
        raise RuntimeError("Could not build relief mesh from image") from exc

    # marching_cubes returns (z, y, x); center and scale for viewer
    verts = verts[:, [2, 1, 0]]
    verts[:, 0] = (verts[:, 0] / max(w - 1, 1) - 0.5) * 2.0
    verts[:, 1] = (verts[:, 1] / max(h - 1, 1) - 0.5) * 2.0
    verts[:, 2] = (verts[:, 2] / max(depth_voxels - 1, 1) - 0.5) * 0.6

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    mesh.export(out_glb)
    return out_glb
