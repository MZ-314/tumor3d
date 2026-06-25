"""Image processing — normalize anchor slice."""

from __future__ import annotations

import numpy as np

from pipeline.modular.types import ModularContext


def run_image_processing(ctx: ModularContext) -> None:
    vol = ctx.slice_volume
    anchor = ctx.anchor_z
    sl = vol.data[anchor].astype(np.float32)
    lo, hi = float(sl.min()), float(sl.max())
    if hi > lo:
        sl = (sl - lo) / (hi - lo)
    vol.data[anchor] = sl
    norm_path = ctx.work_dir / "anchor_normalized.npy"
    np.save(norm_path, sl)
