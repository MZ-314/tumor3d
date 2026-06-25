"""Training data: sample parallel slices from 3D brain volumes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset


def load_brain_volume_nifti(path: Path) -> np.ndarray:
    import nibabel as nib

    data = nib.load(str(path)).get_fdata().astype(np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected 3D volume at {path}, got {data.shape}")
    data = data - data.min()
    data = data / (data.max() or 1.0)
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
  Sample (anchor, mask, z_offset) → target slice pairs from a 3D volume.

    All slices are parallel (same axis) — trains the conditional generator.
    """

    def __init__(
        self,
        volume_paths: list[Path],
        *,
        plane_size: int = 128,
        samples_per_volume: int = 256,
        axis: int | None = None,
    ) -> None:
        self.plane_size = plane_size
        self.samples: list[tuple[np.ndarray, int, int, int]] = []
        for vpath in volume_paths:
            vol = load_brain_volume_nifti(vpath)
            axes = [axis] if axis is not None else [0, 1, 2]
            for ax in axes:
                d = vol.shape[ax]
                for _ in range(samples_per_volume // len(axes)):
                    anchor_i = int(np.random.randint(0, d))
                    target_i = int(np.clip(anchor_i + np.random.randint(-12, 13), 0, d - 1))
                    self.samples.append((vol, ax, anchor_i, target_i))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        vol, axis, anchor_i, target_i = self.samples[idx]
        anchor = np.take(vol, anchor_i, axis=axis)
        target = np.take(vol, target_i, axis=axis)
        mask = _brain_mask_from_plane(anchor)
        anchor_r = _resize_plane(anchor, self.plane_size)
        mask_r = _resize_plane(mask, self.plane_size)
        target_r = _resize_plane(target, self.plane_size)
        half = max(vol.shape[axis] // 2, 1)
        offset = float((target_i - anchor_i) / half)
        offset_plane = np.full((1, self.plane_size, self.plane_size), offset, dtype=np.float32)
        inp = np.concatenate(
            [anchor_r[None], mask_r[None], offset_plane],
            axis=0,
        )
        return torch.from_numpy(inp), torch.from_numpy(target_r[None])
