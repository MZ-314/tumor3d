"""Export a rotatable 3D preview of the uploaded slice stack (no tumor mask)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from config_medical import DEFAULT_PIXEL_SPACING_MM, DEFAULT_SLICE_THICKNESS_MM
from pipeline.ingest.images import SliceVolume


def build_slice_preview_scene(
    volume: SliceVolume,
    output_dir: Path,
    reconstruction_id: str,
) -> Path:
    """Thin 3D slab representing the uploaded imaging volume — for viewing when no lesion is found."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sy, sx = volume.pixel_spacing_mm
    sz = volume.slice_thickness_mm
    z_count, h, w = volume.data.shape

    width_mm = w * sx
    height_mm = h * sy
    depth_mm = max(sz, z_count * sz)

    box = trimesh.creation.box(extents=[width_mm, height_mm, depth_mm])
    box.apply_translation([width_mm / 2, height_mm / 2, depth_mm / 2])

    # Slight transparency cue in vertex color (viewer uses standard material).
    try:
        box.visual.face_colors = [80, 120, 180, 200]
    except Exception:
        pass

    scene_path = output_dir / f"{reconstruction_id}_scene.glb"
    box.export(scene_path)
    return scene_path
