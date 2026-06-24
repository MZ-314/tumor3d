"""Build per-lesion meshes and export combined GLB scene."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage

from config_medical import (
    DEFAULT_PIXEL_SPACING_MM,
    DEFAULT_SLICE_THICKNESS_MM,
    SINGLE_SLICE_DEPTH_VOXELS,
    MeshExportError,
)
from pipeline.ingest.images import SliceVolume
from pipeline.segment.backends import LesionMask
from shared.schemas.pydantic.common import BoundingBox2D, BoundingBox3D, ProvenanceField, SourceType, Vec3


@dataclass
class LesionGeometry:
    lesion_id: str
    mesh_path: Path
    centroid_mm: Vec3
    bounding_box_2d: BoundingBox2D
    bounding_box_3d_mm: BoundingBox3D
    volume_mm3: ProvenanceField
    in_plane_confidence: float
    depth_confidence: float
    vertices: list[list[float]]


def _depth_confidence(slice_count: int) -> float:
    if slice_count <= 1:
        return 0.35
    if slice_count < 5:
        return 0.55
    if slice_count < 15:
        return 0.72
    return 0.88


def _extrude_mask(mask2d: np.ndarray, depth_voxels: int) -> np.ndarray:
    vol = np.zeros((depth_voxels, *mask2d.shape), dtype=bool)
    for z in range(depth_voxels):
        vol[z] = mask2d
    return vol


def _mask_to_mesh(
    mask3d: np.ndarray,
    spacing: tuple[float, float, float],
) -> trimesh.Trimesh | None:
    if not mask3d.any():
        return None

    try:
        from skimage import measure

        verts, faces, _normals, _values = measure.marching_cubes(
            mask3d.astype(np.float32),
            level=0.5,
            spacing=spacing,
        )
    except Exception:
        # Fallback: bounding box mesh
        coords = np.argwhere(mask3d)
        z0, y0, x0 = coords.min(axis=0)
        z1, y1, x1 = coords.max(axis=0)
        sz = spacing
        extents = [
            (x1 - x0 + 1) * sz[2],
            (y1 - y0 + 1) * sz[1],
            (z1 - z0 + 1) * sz[0],
        ]
        box = trimesh.creation.box(extents=extents)
        center = np.array(
            [
                (x0 + x1) / 2 * sz[2],
                (y0 + y1) / 2 * sz[1],
                (z0 + z1) / 2 * sz[0],
            ]
        )
        box.apply_translation(center - box.centroid)
        return box

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    return mesh


def build_lesion_geometries(
    volume: SliceVolume,
    lesions: list[LesionMask],
    output_dir: Path,
    reconstruction_id: str,
) -> list[LesionGeometry]:
    output_dir.mkdir(parents=True, exist_ok=True)
    slice_count = volume.data.shape[0]
    sy, sx = volume.pixel_spacing_mm
    sz = volume.slice_thickness_mm
    depth_conf = _depth_confidence(slice_count)

    results: list[LesionGeometry] = []
    scene_meshes: list[trimesh.Trimesh] = []

    for lesion in lesions:
        mask3d = lesion.mask
        if slice_count == 1:
            mask2d = mask3d[0]
            mask3d = _extrude_mask(mask2d, SINGLE_SLICE_DEPTH_VOXELS)
            sz_eff = DEFAULT_SLICE_THICKNESS_MM
        else:
            mask2d = mask3d.max(axis=0)
            sz_eff = sz

        spacing = (sz_eff, sy, sx)
        mesh = _mask_to_mesh(mask3d, spacing)
        if mesh is None:
            continue

        lesion_key = f"lesion_{lesion.lesion_id}"
        mesh_path = output_dir / f"{reconstruction_id}_{lesion_key}.glb"
        mesh.export(mesh_path)
        scene_meshes.append(mesh)

        rows, cols = np.where(mask2d)
        bbox2d = BoundingBox2D(
            min_row=int(rows.min()),
            min_col=int(cols.min()),
            max_row=int(rows.max()),
            max_col=int(cols.max()),
        )

        coords = np.argwhere(mask3d)
        z0, y0, x0 = coords.min(axis=0)
        z1, y1, x1 = coords.max(axis=0)
        bbox3d = BoundingBox3D(
            min_x=float(x0 * sx),
            min_y=float(y0 * sy),
            min_z=float(z0 * sz_eff),
            max_x=float((x1 + 1) * sx),
            max_y=float((y1 + 1) * sy),
            max_z=float((z1 + 1) * sz_eff),
        )

        cy, cx = (rows.mean(), cols.mean())
        cz = float(coords[:, 0].mean() * sz_eff)
        centroid = Vec3(
            x=float(cx * sx),
            y=float(cy * sy),
            z=cz,
        )

        voxel_vol = sx * sy * sz_eff
        vol_voxels = int(mask3d.sum())
        vol_source = SourceType.INFERENCE if slice_count == 1 else SourceType.MEASURED
        vol_conf = depth_conf if slice_count == 1 else min(0.95, depth_conf + 0.1)

        verts = mesh.vertices.tolist()
        if len(verts) > 500:
            step = max(1, len(verts) // 500)
            verts = verts[::step]

        results.append(
            LesionGeometry(
                lesion_id=lesion_key,
                mesh_path=mesh_path,
                centroid_mm=centroid,
                bounding_box_2d=bbox2d,
                bounding_box_3d_mm=bbox3d,
                volume_mm3=ProvenanceField(
                    value=float(vol_voxels * voxel_vol),
                    confidence=vol_conf,
                    source=vol_source,
                ),
                in_plane_confidence=lesion.in_plane_confidence,
                depth_confidence=depth_conf,
                vertices=verts,
            )
        )

    if not results:
        raise MeshExportError("No lesion meshes were generated")

    if scene_meshes:
        scene = trimesh.util.concatenate(scene_meshes)
        scene_path = output_dir / f"{reconstruction_id}_scene.glb"
        scene.export(scene_path)

    return results


def get_scene_path(output_dir: Path, reconstruction_id: str) -> Path:
    return output_dir / f"{reconstruction_id}_scene.glb"
