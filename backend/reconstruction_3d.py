"""
High-fidelity image-to-3D reconstruction pipeline.

Stages:
  1. SAM 2 — foreground isolation (RGBA cutout)
  2. TRELLIS.2 — image-to-3D with PBR materials
  3. Blender — mesh cleanup and GLB export
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from config_reconstruction import (
    DEFAULT_CONFIG,
    ReconstructionConfig,
    Stage1IsolationError,
    Stage2ReconstructionError,
    Stage3ExportError,
)

logger = logging.getLogger("reconstruction_3d")

_sam2_generator = None
_trellis2_pipeline = None

_BACKEND_DIR = Path(__file__).resolve().parent
_BLENDER_SCRIPT = _BACKEND_DIR / "blender_export.py"


def _register_temp(temp_files: list[Path], path: Path) -> Path:
    temp_files.append(path)
    return path


def _validate_input_path(input_image_path: str) -> Path:
    path = Path(input_image_path)
    if not path.is_file():
        raise Stage1IsolationError(
            f"Stage 1 Failed: Input image not found at '{input_image_path}'"
        )
    return path


def _get_sam2_generator(config: ReconstructionConfig):
    global _sam2_generator
    if _sam2_generator is not None:
        return _sam2_generator

    import torch
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    if not torch.cuda.is_available() and config.sam2_device.startswith("cuda"):
        raise Stage1IsolationError(
            "Stage 1 Failed: CUDA is not available but SAM2_DEVICE requires GPU"
        )

    logger.info("Loading SAM 2 model from %s", config.sam2_checkpoint)
    sam2_model = build_sam2(
        config.sam2_config,
        config.sam2_checkpoint,
        device=config.sam2_device,
        apply_postprocessing=False,
    )
    _sam2_generator = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=32,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.92,
        min_mask_region_area=500,
    )
    return _sam2_generator


def _select_foreground_mask(
    masks: list[dict],
    image_shape: tuple[int, int],
    config: ReconstructionConfig,
) -> dict:
    if not masks:
        raise Stage1IsolationError(
            "Stage 1 Failed: No distinct foreground object isolated"
        )

    height, width = image_shape
    center_y, center_x = height / 2.0, width / 2.0
    best_mask: dict | None = None
    best_score = -1.0

    for mask_data in masks:
        area = float(mask_data.get("area", 0))
        if area <= 0:
            continue
        iou = float(mask_data.get("predicted_iou", 0.5))
        stability = float(mask_data.get("stability_score", 0.5))
        score = area * iou * stability

        bbox = mask_data.get("bbox", [0, 0, width, height])
        cx = bbox[0] + bbox[2] / 2.0
        cy = bbox[1] + bbox[3] / 2.0
        in_center = (
            abs(cx - center_x) < width * 0.2 and abs(cy - center_y) < height * 0.2
        )
        if in_center:
            score *= 1.2

        if score > best_score:
            best_score = score
            best_mask = mask_data

    if best_mask is None or best_score < config.mask_min_score:
        raise Stage1IsolationError(
            "Stage 1 Failed: No distinct foreground object isolated"
        )

    area_ratio = float(best_mask["area"]) / float(height * width)
    if area_ratio < config.mask_min_area_ratio:
        raise Stage1IsolationError(
            f"Stage 1 Failed: Foreground region too small ({area_ratio:.1%} of image)"
        )
    if area_ratio > config.mask_max_area_ratio:
        raise Stage1IsolationError(
            f"Stage 1 Failed: Foreground region too large ({area_ratio:.1%} of image)"
        )

    return best_mask


def _stage1_isolate_foreground(
    input_image_path: Path,
    temp_dir: Path,
    temp_files: list[Path],
    config: ReconstructionConfig,
) -> Path:
    logger.info("Stage 1 start: SAM 2 background isolation — input=%s", input_image_path)
    t0 = time.perf_counter()

    try:
        image = Image.open(input_image_path).convert("RGB")
    except OSError as exc:
        raise Stage1IsolationError(f"Stage 1 Failed: Cannot load image — {exc}") from exc

    image_rgb = np.array(image)
    generator = _get_sam2_generator(config)
    masks = generator.generate(image_rgb)
    selected = _select_foreground_mask(
        masks, (image_rgb.shape[0], image_rgb.shape[1]), config
    )

    segmentation = selected["segmentation"]
    if isinstance(segmentation, dict):
        raise Stage1IsolationError(
            "Stage 1 Failed: Unexpected RLE mask format from SAM 2"
        )

    rgba = np.zeros((*image_rgb.shape[:2], 4), dtype=np.uint8)
    rgba[..., :3] = image_rgb
    rgba[..., 3] = np.where(segmentation, 255, 0).astype(np.uint8)

    if not np.any(rgba[..., 3]):
        raise Stage1IsolationError(
            "Stage 1 Failed: Segmentation produced an empty mask"
        )

    output_path = _register_temp(
        temp_files, temp_dir / f"stage1_{uuid.uuid4().hex}.png"
    )
    Image.fromarray(rgba, mode="RGBA").save(output_path)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Stage 1 complete: isolated foreground — output=%s masks=%d elapsed=%.2fs",
        output_path,
        len(masks),
        elapsed,
    )
    return output_path


def _composite_rgba_on_white(rgba_path: Path) -> Image.Image:
    rgba = Image.open(rgba_path).convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


def _get_trellis2_pipeline(config: ReconstructionConfig):
    global _trellis2_pipeline
    if _trellis2_pipeline is not None:
        return _trellis2_pipeline

    import torch
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    if not torch.cuda.is_available() and config.trellis2_device.startswith("cuda"):
        raise Stage2ReconstructionError(
            "Stage 2 Failed: CUDA is not available but TRELLIS2_DEVICE requires GPU"
        )

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

    logger.info("Loading TRELLIS.2 pipeline from %s", config.trellis2_model_id)
    _trellis2_pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
        config.trellis2_model_id
    )
    if config.trellis2_device.startswith("cuda"):
        _trellis2_pipeline.cuda()
    else:
        _trellis2_pipeline.to(config.trellis2_device)
    return _trellis2_pipeline


def _stage2_trellis_reconstruct(
    rgba_path: Path,
    temp_dir: Path,
    temp_files: list[Path],
    config: ReconstructionConfig,
) -> Path:
    logger.info("Stage 2 start: TRELLIS.2 reconstruction — input=%s", rgba_path)
    t0 = time.perf_counter()

    try:
        import o_voxel
        import torch
    except ImportError as exc:
        raise Stage2ReconstructionError(
            "Stage 2 Failed: TRELLIS.2 dependencies not installed — "
            "see docs/runpod-setup.md"
        ) from exc

    image = _composite_rgba_on_white(rgba_path)
    pipeline = _get_trellis2_pipeline(config)

    torch.manual_seed(config.trellis2_seed)
    try:
        meshes = pipeline.run(image)
    except Exception as exc:
        raise Stage2ReconstructionError(
            f"Stage 2 Failed: TRELLIS.2 inference error — {exc}"
        ) from exc

    if not meshes:
        raise Stage2ReconstructionError(
            "Stage 2 Failed: TRELLIS.2 returned no mesh output"
        )

    mesh = meshes[0]
    try:
        mesh.simplify(16_777_216)
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=config.trellis2_decimation_target,
            texture_size=config.trellis2_texture_size,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            verbose=False,
        )
    except Exception as exc:
        raise Stage2ReconstructionError(
            f"Stage 2 Failed: TRELLIS.2 GLB export error — {exc}"
        ) from exc

    raw_glb_path = _register_temp(
        temp_files, temp_dir / f"stage2_raw_{uuid.uuid4().hex}.glb"
    )
    try:
        glb.export(str(raw_glb_path), extension_webp=True)
    except Exception as exc:
        raise Stage2ReconstructionError(
            f"Stage 2 Failed: Could not write intermediate GLB — {exc}"
        ) from exc

    if not raw_glb_path.is_file() or raw_glb_path.stat().st_size == 0:
        raise Stage2ReconstructionError(
            "Stage 2 Failed: TRELLIS.2 produced an empty mesh file"
        )

    elapsed = time.perf_counter() - t0
    logger.info(
        "Stage 2 complete: raw GLB — output=%s size=%d bytes elapsed=%.2fs",
        raw_glb_path,
        raw_glb_path.stat().st_size,
        elapsed,
    )
    return raw_glb_path


def _stage3_blender_export(
    raw_glb_path: Path,
    output_directory: Path,
    temp_files: list[Path],
    config: ReconstructionConfig,
) -> Path:
    logger.info("Stage 3 start: Blender mesh cleanup — input=%s", raw_glb_path)
    t0 = time.perf_counter()

    output_directory.mkdir(parents=True, exist_ok=True)
    final_path = output_directory / f"{uuid.uuid4().hex}.glb"

    if not _BLENDER_SCRIPT.is_file():
        raise Stage3ExportError(
            f"Stage 3 Failed: Blender script not found at {_BLENDER_SCRIPT}"
        )

    cmd = [
        config.blender_bin,
        "--background",
        "--python",
        str(_BLENDER_SCRIPT),
        "--",
        str(raw_glb_path.resolve()),
        str(final_path.resolve()),
        str(config.blender_decimate_ratio),
        str(config.blender_max_triangles),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=config.stage3_timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise Stage3ExportError(
            "Stage 3 Failed: Blender mesh processing timeout"
        ) from exc
    except FileNotFoundError as exc:
        raise Stage3ExportError(
            f"Stage 3 Failed: Blender binary not found at '{config.blender_bin}'"
        ) from exc

    if result.returncode != 0:
        stderr = (result.stderr or "").strip() or "unknown error"
        raise Stage3ExportError(f"Stage 3 Failed: Blender export error — {stderr}")

    if not final_path.is_file() or final_path.stat().st_size == 0:
        raise Stage3ExportError(
            "Stage 3 Failed: Blender produced an empty output file"
        )

    try:
        import trimesh

        loaded = trimesh.load(final_path, force="mesh")
        if isinstance(loaded, trimesh.Scene):
            tri_count = sum(
                len(g.faces) for g in loaded.geometry.values() if hasattr(g, "faces")
            )
        else:
            tri_count = len(loaded.faces)
        logger.info("Stage 3 mesh validation: triangle_count=%d", tri_count)
    except Exception as exc:
        logger.warning("Stage 3 trimesh validation skipped: %s", exc)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Stage 3 complete: final GLB — output=%s elapsed=%.2fs",
        final_path,
        elapsed,
    )
    return final_path.resolve()


async def process_image_to_3d(
    input_image_path: str,
    output_directory: str,
    config: ReconstructionConfig | None = None,
) -> str:
    """
    Run the full SAM 2 → TRELLIS.2 → Blender pipeline.

    Returns the absolute path to the finalized .glb file.
    """
    cfg = config or DEFAULT_CONFIG
    temp_files: list[Path] = []
    temp_dir = Path(output_directory) / f"_tmp_{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_files.append(temp_dir)

    logger.info(
        "Pipeline start: input=%s output_dir=%s",
        input_image_path,
        output_directory,
    )
    pipeline_t0 = time.perf_counter()

    try:
        input_path = _validate_input_path(input_image_path)

        try:
            rgba_path = await asyncio.wait_for(
                asyncio.to_thread(
                    _stage1_isolate_foreground, input_path, temp_dir, temp_files, cfg
                ),
                timeout=cfg.stage1_timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise Stage1IsolationError(
                "Stage 1 Failed: Background removal timeout"
            ) from exc

        try:
            raw_glb_path = await asyncio.wait_for(
                asyncio.to_thread(
                    _stage2_trellis_reconstruct, rgba_path, temp_dir, temp_files, cfg
                ),
                timeout=cfg.stage2_timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise Stage2ReconstructionError(
                "Stage 2 Failed: TRELLIS.2 reconstruction timeout"
            ) from exc

        try:
            final_path = await asyncio.wait_for(
                asyncio.to_thread(
                    _stage3_blender_export,
                    raw_glb_path,
                    Path(output_directory),
                    temp_files,
                    cfg,
                ),
                timeout=cfg.stage3_timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise Stage3ExportError(
                "Stage 3 Failed: Blender mesh processing timeout"
            ) from exc

        elapsed = time.perf_counter() - pipeline_t0
        logger.info(
            "Pipeline complete: output=%s elapsed=%.2fs",
            final_path,
            elapsed,
        )
        return str(final_path)

    except Stage1IsolationError:
        raise
    except Stage2ReconstructionError:
        raise
    except Stage3ExportError:
        raise
    except Exception as exc:
        raise Stage2ReconstructionError(
            f"Stage 2 Failed: Unexpected pipeline error — {exc}"
        ) from exc
    finally:
        for path in reversed(temp_files):
            try:
                if path.is_dir():
                    for child in sorted(path.rglob("*"), reverse=True):
                        if child.is_file():
                            child.unlink(missing_ok=True)
                    path.rmdir()
                elif path.is_file():
                    path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Failed to remove temp path %s: %s", path, exc)
