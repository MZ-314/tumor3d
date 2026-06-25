"""Local module detection — lesion bbox intersects registered modules."""

from __future__ import annotations

import numpy as np

from pipeline.modular.types import ModularContext


def _lesion_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows, cols = np.where(mask)
    if rows.size == 0:
        return None
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def _module_centroid_2d(module_id: str, organ_mask: np.ndarray) -> tuple[float, float]:
    """Map module id to approximate 2D region on anchor slice."""
    h, w = organ_mask.shape
    rows, cols = np.where(organ_mask)
    if rows.size == 0:
        return w / 2, h / 2
    cy, cx = float(rows.mean()), float(cols.mean())
    offsets = {
        "FrontalLobe": (0.0, 0.15 * h),
        "TemporalLobe": (0.2 * w, -0.1 * h),
        "ParietalLobe": (0.0, 0.05 * h),
        "OccipitalLobe": (0.0, -0.2 * h),
        "Hippocampus": (0.15 * w, -0.05 * h),
        "Amygdala": (0.12 * w, -0.08 * h),
        "Ventricles": (0.0, 0.0),
        "WhiteMatter": (0.0, 0.0),
        "CorpusCallosum": (0.0, 0.05 * h),
        "brain_shell": (0.0, 0.0),
    }
    dx, dy = offsets.get(module_id, (0.0, 0.0))
    return cx + dx, cy + dy


def run_module_detection(ctx: ModularContext) -> list[str]:
    affected: list[str] = []
    if ctx.lesion_mask_2d is None or not ctx.lesion_mask_2d.any():
        ctx.affected_module_ids = []
        return []

    bbox = _lesion_bbox(ctx.lesion_mask_2d)
    if bbox is None:
        ctx.affected_module_ids = []
        return []

    r0, r1, c0, c1 = bbox
    lcx = (c0 + c1) / 2.0
    lcy = (r0 + r1) / 2.0

    best_id = "FrontalLobe"
    best_dist = float("inf")
    for mod in ctx.modules:
        if mod.module_id == "brain_shell":
            continue
        mx, my = _module_centroid_2d(mod.module_id, ctx.organ_mask_2d)
        dist = (mx - lcx) ** 2 + (my - lcy) ** 2
        if dist < best_dist:
            best_dist = dist
            best_id = mod.module_id

    affected.append(best_id)
    if ctx.graph:
        for edge in ctx.graph.edges:
            if edge.source_id == best_id and edge.target_id not in affected:
                affected.append(edge.target_id)
            if edge.target_id == best_id and edge.source_id not in affected:
                affected.append(edge.source_id)

    ctx.affected_module_ids = affected
    return affected
