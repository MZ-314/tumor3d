"""Orchestrate modular brain reconstruction engines."""

from __future__ import annotations

import numpy as np

from pipeline.modular.anatomical_extraction import run_anatomical_extraction
from pipeline.modular.atlas_library import run_atlas_library
from pipeline.modular.consensus import run_consensus
from pipeline.modular.image_processing import run_image_processing
from pipeline.modular.mesh_optimize import run_mesh_optimize
from pipeline.modular.module_assembly import run_module_assembly
from pipeline.modular.module_detection import run_module_detection
from pipeline.modular.module_morph import run_local_morph
from pipeline.modular.multi_model_ai import run_multi_model_ai
from pipeline.modular.neighbor_propagation import run_neighbor_propagation
from pipeline.modular.optimizer import run_optimizer
from pipeline.modular.registration import run_registration
from pipeline.modular.spatial_mapping import run_spatial_mapping
from pipeline.modular.structure_replacement import run_structure_replacement
from pipeline.modular.types import ModularContext
from pipeline.modular.volume_completion import run_volume_completion
from pipeline.modular.volumetric import run_volumetric
from pipeline.reconstruct.context import PipelineState
from shared.schemas.pydantic.pipeline import ReconstructionBlueprint


def _lesion_mask_2d(state: PipelineState, anchor_z: int) -> np.ndarray | None:
    if state.segmentation is None or not state.segmentation.lesions:
        return None
    lesion = state.segmentation.lesions[0]
    return np.asarray(lesion.mask[anchor_z], dtype=bool)


def build_modular_context(state: PipelineState) -> ModularContext:
    if state.slice_volume is None or state.scan_context is None:
        raise RuntimeError("slice_volume and scan_context required")
    if state.organ_mask_2d is None:
        raise RuntimeError("organ_mask_2d required for modular reconstruction")

    anchor_indices = state.scan_context.anchor_slice_indices
    anchor_z = anchor_indices[0] if anchor_indices else 0

    return ModularContext(
        reconstruction_id=state.reconstruction_id,
        work_dir=state.work_dir,
        slice_volume=state.slice_volume,
        organ_mask_2d=np.asarray(state.organ_mask_2d, dtype=bool),
        anatomical_map=state.anatomical_map,
        scan_context=state.scan_context,
        anchor_z=anchor_z,
        atlas_warp=state.atlas_warp,
        blueprint=state.blueprint,
        lesion_mask_2d=_lesion_mask_2d(state, anchor_z),
    )


def run_modular_perception(ctx: ModularContext) -> None:
    run_image_processing(ctx)
    run_multi_model_ai(ctx)
    run_consensus(ctx)
    run_anatomical_extraction(ctx)
    run_spatial_mapping(ctx)


def run_modular_atlas_block(ctx: ModularContext) -> None:
    run_atlas_library(ctx)
    run_registration(ctx)
    run_structure_replacement(ctx)


def run_modular_local_block(ctx: ModularContext) -> None:
    run_module_detection(ctx)
    run_local_morph(ctx)
    run_neighbor_propagation(ctx)
    run_optimizer(ctx)


async def run_modular_completion_block(ctx: ModularContext) -> None:
    if ctx.blueprint is None:
        z, h, w = ctx.slice_volume.data.shape
        ctx.blueprint = ReconstructionBlueprint(
            volume_shape_zyx=[max(z, 48), h, w],
            anchor_slice_indices=[ctx.anchor_z],
            synthesis_strategy="modular_volume_completion",
            locked_lesion_planes=[ctx.anchor_z],
            organ_extent_zyx=[max(z, 48), h, w],
        )
    await run_volume_completion(ctx)
    run_volumetric(ctx)


def run_modular_assembly_block(ctx: ModularContext) -> None:
    run_module_assembly(ctx)
    run_mesh_optimize(ctx)


def apply_modular_context_to_state(state: PipelineState, ctx: ModularContext) -> None:
    if ctx.anatomical_map is not None:
        state.anatomical_map = ctx.anatomical_map
    if ctx.atlas_warp is not None:
        state.atlas_warp = ctx.atlas_warp
    if ctx.blueprint is not None:
        state.blueprint = ctx.blueprint
    if ctx.synthesis is not None:
        state.synthesis = ctx.synthesis
    if ctx.output_volume is not None:
        state.output_volume = ctx.output_volume
    if ctx.assembly is not None:
        state.module_assembly = ctx.assembly
    if ctx.pose is not None:
        state.scan_context = state.scan_context.model_copy(
            update={"mri_view": ctx.pose.mri_view}
        )
