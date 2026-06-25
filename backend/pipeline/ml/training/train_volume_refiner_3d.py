"""Train 3D volume refiner (coarse slab → full brain volume)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter, zoom
from torch.utils.data import DataLoader, Dataset

from config_pipeline import ATLAS_BRAIN_TEMPLATE, ML_VOLUME_MODEL_DIR
from pipeline.ml.brain_envelope import build_extruded_mask_3d
from pipeline.ml.models.volume_refiner_3d import BrainVolumeRefiner3D
from pipeline.ml.training.dataset import load_brain_volume_nifti

logger = logging.getLogger(__name__)

REFINER_SIZE = 64


def _to_cube(vol: np.ndarray, size: int = REFINER_SIZE) -> np.ndarray:
    z, h, w = vol.shape
    return zoom(vol, (size / z, size / h, size / w), order=1).astype(np.float32)


def _coarse_from_anchor(vol: np.ndarray, anchor_z: int) -> np.ndarray:
    """Simulate parallel-slice slab: only anchor sharp, off-slice heavily smeared."""
    coarse = vol.copy()
    for z in range(vol.shape[0]):
        if z == anchor_z:
            continue
        t = 1.0 - abs(z - anchor_z) / max(anchor_z, 1)
        coarse[z] = vol[anchor_z] * (0.55 + 0.45 * t) + vol[z] * 0.25
    coarse = gaussian_filter(coarse, sigma=(2.2, 0.8, 0.8))
    coarse[anchor_z] = vol[anchor_z]
    return coarse.astype(np.float32)


class VolumeRefinerDataset(Dataset):
    def __init__(self, volume_paths: list[Path], *, cube_size: int = REFINER_SIZE) -> None:
        self.items: list[tuple[np.ndarray, np.ndarray]] = []
        for vpath in volume_paths:
            logger.info("Loading %s for 3D refiner …", vpath.name)
            vol = _to_cube(load_brain_volume_nifti(vpath), cube_size)
            for _ in range(48):
                anchor_z = int(np.random.randint(cube_size // 4, 3 * cube_size // 4))
                mask2d = vol[anchor_z] > np.percentile(vol[anchor_z], 35)
                envelope = build_extruded_mask_3d(vol.shape, mask2d, anchor_z)
                coarse = _coarse_from_anchor(vol, anchor_z)
                coarse = coarse * envelope
                self.items.append(
                    (
                        np.stack([coarse, envelope], axis=0).astype(np.float32),
                        vol.astype(np.float32),
                    )
                )
        logger.info("3D refiner dataset: %d samples", len(self.items))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        inp, target = self.items[idx]
        return torch.from_numpy(inp), torch.from_numpy(target[None])


def train(
    volume_paths: list[Path],
    *,
    output_path: Path,
    epochs: int = 30,
    batch_size: int = 4,
    device: str | None = None,
) -> Path:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = VolumeRefinerDataset(volume_paths)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = BrainVolumeRefiner3D().to(dev)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()

    steps = len(loader)
    print(f"3D refiner training on {dev}: {epochs} epochs × {steps} batches", flush=True)

    model.train()
    for epoch in range(epochs):
        running = 0.0
        for step, (inp, target) in enumerate(loader, start=1):
            inp = inp.to(dev)
            target = target.to(dev)
            optim.zero_grad()
            pred = model(inp)
            loss = loss_fn(pred, target)
            loss.backward()
            optim.step()
            running += float(loss.item())
            if step == 1 or step == steps:
                print(
                    f"  refiner epoch {epoch + 1}/{epochs} batch {step}/{steps} loss={loss.item():.4f}",
                    flush=True,
                )
        logger.info("refiner epoch %d avg_loss=%.4f", epoch + 1, running / max(steps, 1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"version": "ml_volume_refiner_v1", "state_dict": model.state_dict()}, output_path)
    print(f"Saved 3D refiner → {output_path}", flush=True)
    return output_path


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--volumes", nargs="*", type=Path, default=[])
    parser.add_argument("--output", type=Path, default=ML_VOLUME_MODEL_DIR / "volume_refiner_3d.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    volumes = list(args.volumes) or (
        [ATLAS_BRAIN_TEMPLATE] if ATLAS_BRAIN_TEMPLATE.is_file() else []
    )
    if not volumes:
        raise SystemExit("No training volumes")

    train(
        volumes,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
