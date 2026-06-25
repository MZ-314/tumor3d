"""Training data: sample parallel slices from 3D brain volumes."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def load_brain_volume_nifti(path: Path, *, max_dim: int | None = 128) -> np.ndarray:
    import nibabel as nib

    data = nib.load(str(path)).get_fdata().astype(np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected 3D volume at {path}, got {data.shape}")
    data = data - data.min()
    data = data / (data.max() or 1.0)
    if max_dim is not None and max(data.shape) > max_dim:
        scale = max_dim / max(data.shape)
        data = zoom(data, scale, order=1).astype(np.float32)
    return data


def _resize_plane(plane: np.ndarray, size: int) -> np.ndarray:
    h, w = plane.shape
    if h == size and w == size:
        return plane.astype(np.float32)
    return zoom(plane, (size / h, size / w), order=1).astype(np.float32)


def _brain_mask_from_plane(plane: np.ndarray) -> np.ndarray:
    return (plane > np.percentile(plane, 35)).astype(np.float32)


class ParallelSliceDataset(Dataset):
    """
    Precomputed (anchor, mask, z_offset) → target slice pairs from 3D volumes.

    All slices are parallel (same axis) — trains the conditional generator.
    """

    def __init__(
        self,
        volume_paths: list[Path],
        *,
        plane_size: int = 128,
        samples_per_volume: int = 192,
        axis: int | None = None,
        max_volume_dim: int | None = 128,
    ) -> None:
        self.plane_size = plane_size
        self.inputs: list[torch.Tensor] = []
        self.targets: list[torch.Tensor] = []

        for vpath in volume_paths:
            logger.info("Loading volume %s …", vpath.name)
            vol = load_brain_volume_nifti(vpath, max_dim=max_volume_dim)
            axes = [axis] if axis is not None else [0, 1, 2]
            per_axis = max(samples_per_volume // len(axes), 1)
            for ax in axes:
                depth = vol.shape[ax]
                half = max(depth // 2, 1)
                for _ in range(per_axis):
                    anchor_i = int(np.random.randint(0, depth))
                    target_i = int(np.clip(anchor_i + np.random.randint(-12, 13), 0, depth - 1))
                    anchor = np.take(vol, anchor_i, axis=ax)
                    target = np.take(vol, target_i, axis=ax)
                    mask = _brain_mask_from_plane(anchor)
                    anchor_r = _resize_plane(anchor, plane_size)
                    mask_r = _resize_plane(mask, plane_size)
                    target_r = _resize_plane(target, plane_size)
                    offset = float((target_i - anchor_i) / half)
                    offset_plane = np.full(
                        (1, plane_size, plane_size),
                        offset,
                        dtype=np.float32,
                    )
                    inp = np.concatenate(
                        [anchor_r[None], mask_r[None], offset_plane],
                        axis=0,
                    )
                    self.inputs.append(torch.from_numpy(inp))
                    self.targets.append(torch.from_numpy(target_r[None]))

        logger.info("Dataset ready: %d slice pairs", len(self.inputs))

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]
