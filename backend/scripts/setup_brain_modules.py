#!/usr/bin/env python3
"""Generate modular brain atlas GLBs from MNI152 + spatial parcellation."""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BACKEND))

from config_pipeline import ATLAS_BRAIN_DIR, ATLAS_BRAIN_TEMPLATE, MODULAR_BRAIN_DIR  # noqa: E402

# Harvard-Oxford cortical atlas (FSL) — optional download
HO_CORTICAL_URL = (
    "https://raw.githubusercontent.com/InstitutdeNeurosciencesdesSystems/"
    "ADF-pipeline-datarefs/main/HarvardOxford-cort-maxprob-thr25-1mm.nii.gz"
)

MODULE_SPECS: list[dict[str, object]] = [
    {
        "id": "FrontalLobe",
        "display_name": "Frontal Lobe",
        "connects_to": ["ParietalLobe", "TemporalLobe", "CorpusCallosum", "WhiteMatter"],
        "morphable": True,
        "region": "frontal",
    },
    {
        "id": "TemporalLobe",
        "display_name": "Temporal Lobe",
        "connects_to": ["FrontalLobe", "ParietalLobe", "OccipitalLobe", "Hippocampus", "Amygdala"],
        "morphable": True,
        "region": "temporal",
    },
    {
        "id": "ParietalLobe",
        "display_name": "Parietal Lobe",
        "connects_to": ["FrontalLobe", "TemporalLobe", "OccipitalLobe", "CorpusCallosum"],
        "morphable": True,
        "region": "parietal",
    },
    {
        "id": "OccipitalLobe",
        "display_name": "Occipital Lobe",
        "connects_to": ["ParietalLobe", "TemporalLobe"],
        "morphable": True,
        "region": "occipital",
    },
    {
        "id": "Ventricles",
        "display_name": "Ventricles",
        "connects_to": ["WhiteMatter"],
        "morphable": False,
        "region": "ventricles",
    },
    {
        "id": "WhiteMatter",
        "display_name": "White Matter",
        "connects_to": ["FrontalLobe", "ParietalLobe", "TemporalLobe", "CorpusCallosum", "Ventricles"],
        "morphable": True,
        "region": "white_matter",
    },
    {
        "id": "CorpusCallosum",
        "display_name": "Corpus Callosum",
        "connects_to": ["FrontalLobe", "ParietalLobe", "WhiteMatter"],
        "morphable": True,
        "region": "corpus_callosum",
    },
    {
        "id": "Hippocampus",
        "display_name": "Hippocampus",
        "connects_to": ["TemporalLobe", "Amygdala"],
        "morphable": True,
        "region": "hippocampus",
    },
    {
        "id": "Amygdala",
        "display_name": "Amygdala",
        "connects_to": ["TemporalLobe", "Hippocampus"],
        "morphable": True,
        "region": "amygdala",
    },
    {
        "id": "brain_shell",
        "display_name": "Brain Shell",
        "connects_to": ["FrontalLobe", "ParietalLobe", "TemporalLobe", "OccipitalLobe"],
        "morphable": False,
        "region": "shell",
    },
]


def _load_template() -> tuple[np.ndarray, tuple[float, float, float]]:
    if not ATLAS_BRAIN_TEMPLATE.is_file():
        raise FileNotFoundError(
            f"Missing {ATLAS_BRAIN_TEMPLATE}. Run: python backend/scripts/setup_brain_atlas.py"
        )
    import nibabel as nib

    img = nib.load(str(ATLAS_BRAIN_TEMPLATE))
    data = np.asarray(img.get_fdata(), dtype=np.float32)
    zooms = img.header.get_zooms()[:3]
    return data, (float(zooms[0]), float(zooms[1]), float(zooms[2]))


def _brain_mask(data: np.ndarray) -> np.ndarray:
    thresh = float(np.percentile(data[data > 0], 25)) if np.any(data > 0) else 0.1
    mask = data > thresh
    from scipy import ndimage

    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_closing(mask, iterations=2)
    return mask


HO_SUBCORTICAL_URL = (
    "https://raw.githubusercontent.com/InstitutdeNeurosciencesdesSystems/"
    "ADF-pipeline-datarefs/main/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz"
)

# Harvard-Oxford cortical label index groups (1-based, approximate lobe mapping)
HO_LOBE_LABELS: dict[str, tuple[int, ...]] = {
    "frontal": tuple(range(1, 27)),
    "temporal": tuple(range(27, 41)),
    "parietal": tuple(range(41, 53)),
    "occipital": tuple(range(53, 63)),
}


def _download_atlas_nifti(url: str, dest: Path) -> Path | None:
    if dest.is_file():
        return dest
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
        return dest
    except Exception as exc:
        print(f"WARN: could not download {url}: {exc}")
        return None


def _load_label_atlas() -> np.ndarray | None:
    """Try Harvard-Oxford cortical maxprob atlas aligned to MNI152 1mm."""
    cache = ATLAS_BRAIN_DIR / "harvard_oxford_cortical.nii.gz"
    path = _download_atlas_nifti(HO_CORTICAL_URL, cache)
    if path is None:
        return None
    import nibabel as nib

    data = np.asarray(nib.load(str(path)).get_fdata())
    if data.ndim == 4:
        data = data.argmax(axis=-1)
    return data.astype(np.int32)


def _region_mask_from_ho(region: str, labels: np.ndarray, brain: np.ndarray) -> np.ndarray | None:
    key = region
    if key not in HO_LOBE_LABELS:
        return None
    ids = HO_LOBE_LABELS[key]
    mask = np.isin(labels, ids) & brain
    return mask if mask.any() else None


def _region_mask(region: str, data: np.ndarray, brain: np.ndarray, ho_labels: np.ndarray | None = None) -> np.ndarray:
    """Spatial parcellation on MNI152; uses Harvard-Oxford labels when available."""
    if ho_labels is not None and region in HO_LOBE_LABELS:
        ho_mask = _region_mask_from_ho(region, ho_labels, brain)
        if ho_mask is not None:
            return ho_mask
    z, y, x = np.where(brain)
    if z.size == 0:
        return brain
    cz, cy, cx = float(z.mean()), float(y.mean()), float(x.mean())
    zz, yy, xx = np.mgrid[0 : data.shape[0], 0 : data.shape[1], 0 : data.shape[2]]
    rel_y = (yy - cy) / max(data.shape[1] * 0.35, 1.0)
    rel_x = (xx - cx) / max(data.shape[2] * 0.35, 1.0)
    rel_z = (zz - cz) / max(data.shape[0] * 0.35, 1.0)
    dist = np.sqrt(rel_x**2 + rel_y**2 + rel_z**2)

    if region == "shell":
        from scipy import ndimage

        inner = ndimage.binary_erosion(brain, iterations=3)
        return brain & ~inner

    if region == "ventricles":
        csf = (data < np.percentile(data[brain], 30)) & brain
        center = dist < 0.35
        return csf & center

    if region == "white_matter":
        wm = (data > np.percentile(data[brain], 72)) & brain
        outer = dist > 0.25
        return wm & outer

    if region == "corpus_callosum":
        midline = np.abs(xx - cx) < data.shape[2] * 0.04
        bright = data > np.percentile(data[brain], 80)
        superior = yy > cy - data.shape[1] * 0.05
        return brain & midline & bright & superior

    if region == "frontal":
        return brain & (yy > cy + data.shape[1] * 0.02)

    if region == "occipital":
        return brain & (yy < cy - data.shape[1] * 0.12)

    if region == "temporal":
        lateral = np.abs(xx - cx) > data.shape[2] * 0.08
        inferior = yy < cy + data.shape[1] * 0.08
        return brain & lateral & inferior & (yy > cy - data.shape[1] * 0.25)

    if region == "parietal":
        superior = yy > cy - data.shape[1] * 0.05
        not_frontal = yy <= cy + data.shape[1] * 0.02
        not_occipital = yy >= cy - data.shape[1] * 0.12
        central = np.abs(xx - cx) < data.shape[2] * 0.22
        return brain & superior & not_frontal & not_occipital & central

    if region == "hippocampus":
        deep = (yy < cy) & (yy > cy - data.shape[1] * 0.18)
        lateral = (np.abs(xx - cx) > data.shape[2] * 0.12) & (np.abs(xx - cx) < data.shape[2] * 0.28)
        return brain & deep & lateral

    if region == "amygdala":
        deep = (yy < cy - data.shape[1] * 0.02) & (yy > cy - data.shape[1] * 0.14)
        lateral = (np.abs(xx - cx) > data.shape[2] * 0.10) & (np.abs(xx - cx) < data.shape[2] * 0.20)
        return brain & deep & lateral

    return brain


def _decimate_mesh(mesh):
    """Reduce face count; requires fast-simplification when available."""
    import trimesh

    target = max(500, len(mesh.faces) // 2)
    try:
        return mesh.simplify_quadric_decimation(target)
    except (ImportError, ModuleNotFoundError, ValueError):
        pass
    try:
        return mesh.simplify_quadric_decimation(target, aggression=7)
    except Exception:
        return mesh


def _mask_to_mesh(mask: np.ndarray, spacing: tuple[float, float, float]):
    import trimesh
    from skimage import measure

    if not mask.any():
        return None
    try:
        verts, faces, _normals, _values = measure.marching_cubes(
            mask.astype(np.float32),
            level=0.5,
            spacing=spacing,
        )
    except (ValueError, RuntimeError):
        return None
    if verts.size == 0:
        return None
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    if mesh.vertices.shape[0] < 12:
        return None
    return _decimate_mesh(mesh)


def _center_mesh(mesh) -> None:
    c = mesh.vertices.mean(axis=0)
    mesh.vertices -= c


def main() -> int:
    import trimesh

    out_root = MODULAR_BRAIN_DIR
    out_root.mkdir(parents=True, exist_ok=True)

    data, spacing = _load_template()
    brain = _brain_mask(data)
    ho_labels = _load_label_atlas()
    if ho_labels is not None:
        print("OK: using Harvard-Oxford cortical labels for lobe modules")
    else:
        print("WARN: Harvard-Oxford unavailable — using spatial parcellation fallback")

    manifest_modules: list[dict[str, object]] = []
    scene_meshes: list[trimesh.Trimesh] = []

    for spec in MODULE_SPECS:
        module_id = str(spec["id"])
        region = str(spec["region"])
        mod_dir = out_root / module_id
        mod_dir.mkdir(parents=True, exist_ok=True)

        mask = _region_mask(region, data, brain, ho_labels)
        mesh = _mask_to_mesh(mask, spacing)
        if mesh is None:
            print(f"WARN: empty mesh for {module_id}, using small placeholder")
            mesh = trimesh.creation.icosphere(radius=8.0, subdivisions=2)

        _center_mesh(mesh)
        mesh_path = mod_dir / "mesh.glb"
        mesh.export(mesh_path)

        metadata = {
            "id": module_id,
            "display_name": spec["display_name"],
            "label_ids": [region],
            "connects_to": spec["connects_to"],
            "default_confidence": 0.65,
            "morphable": spec["morphable"],
            "geometry_source": "atlas",
        }
        meta_path = mod_dir / "metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        manifest_modules.append(
            {
                "id": module_id,
                "mesh": str(mesh_path.relative_to(out_root)).replace("\\", "/"),
                "metadata": str(meta_path.relative_to(out_root)).replace("\\", "/"),
                "label_ids": [region],
                "connects_to": spec["connects_to"],
                "default_confidence": 0.65,
                "morphable": spec["morphable"],
            }
        )
        scene_meshes.append(mesh)
        print(f"OK: {module_id} -> {mesh_path}")

    brain_glb = out_root / "Brain.glb"
    if scene_meshes:
        scene = trimesh.Scene()
        for spec, mesh in zip(MODULE_SPECS, scene_meshes, strict=True):
            scene.add_geometry(mesh, node_name=str(spec["id"]))
        scene.export(brain_glb)

    manifest = {
        "atlas_id": "mni152_modular_v1",
        "atlas_version": "1.0.0",
        "template": str(ATLAS_BRAIN_TEMPLATE),
        "root_glb": "Brain.glb",
        "modules": manifest_modules,
    }
    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"OK: manifest at {manifest_path}")
    print(f"OK: Brain.glb at {brain_glb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
