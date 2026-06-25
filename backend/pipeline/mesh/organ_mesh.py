"""Export a 3D organ surface mesh from a synthesized brain volume."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter

from pipeline.ingest.images import SliceVolume


def build_organ_mesh_scene(
    volume: SliceVolume,
    output_dir: Path,
    reconstruction_id: str,
    *,
    organ_mask_2d: np.ndarray | None = None,
) -> Path:
    """Marching-cubes organ shell from atlas-guided volume (single-slice USP demo)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data = volume.data.astype(np.float32)
    z_count, h, w = data.shape
    sy, sx = volume.pixel_spacing_mm
    sz = volume.slice_thickness_mm
    spacing = (sz, sy, sx)

    if organ_mask_2d is not None:
        mask2d = np.asarray(organ_mask_2d, dtype=bool)
        if mask2d.shape != (h, w):
            from scipy.ndimage import zoom

            mask2d = (
                zoom(mask2d.astype(np.float32), (h / mask2d.shape[0], w / mask2d.shape[1]), order=0)
                > 0.5
            )
        anchor = z_count // 2
        mask3d = np.zeros(data.shape, dtype=bool)
        for z in range(z_count):
            falloff = max(0.35, 1.0 - abs(z - anchor) / max(anchor, 1))
            mask3d[z] = mask2d if z == anchor else (data[z] > 0.12) & (falloff > 0.4)
        mask3d[anchor] = mask2d
        field = gaussian_filter(mask3d.astype(np.float32), sigma=(1.0, 1.0, 1.0))
    else:
        field = gaussian_filter(data, sigma=(0.8, 1.0, 1.0))

    level = float(np.percentile(field, 55))
    mesh = _isosurface_mesh(field, spacing, level=level)
    if mesh is None:
        mesh = _isosurface_mesh(data, spacing, level=0.2)
    if mesh is None:
        scene_path = output_dir / f"{reconstruction_id}_scene.glb"
        box = trimesh.creation.box(extents=[w * sx, h * sy, z_count * sz])
        box.apply_translation([w * sx / 2, h * sy / 2, z_count * sz / 2])
        box.export(scene_path)
        return scene_path

    try:
        mesh.visual.face_colors = [120, 160, 200, 220]
    except Exception:
        pass

    scene_path = output_dir / f"{reconstruction_id}_scene.glb"
    mesh.export(scene_path)
    return scene_path


def _isosurface_mesh(
    field: np.ndarray,
    spacing: tuple[float, float, float],
    *,
    level: float,
) -> trimesh.Trimesh | None:
    try:
        from skimage import measure

        verts, faces, _normals, _values = measure.marching_cubes(
            field.astype(np.float32),
            level=level,
            spacing=spacing,
        )
    except Exception:
        return None
    if len(verts) == 0:
        return None
    return trimesh.Trimesh(vertices=verts, faces=faces, process=True)
