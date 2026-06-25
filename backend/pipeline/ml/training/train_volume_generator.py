"""Train conditional slice generator on 3D brain MRI volumes."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config_pipeline import ATLAS_BRAIN_TEMPLATE, ML_VOLUME_MODEL_DIR
from pipeline.ml.models.conditional_unet import ConditionalSliceUNet
from pipeline.ml.training.dataset import ParallelSliceDataset

logger = logging.getLogger(__name__)


def train(
    volume_paths: list[Path],
    *,
    output_path: Path,
    epochs: int = 40,
    batch_size: int = 8,
    samples_per_volume: int = 192,
    lr: float = 1e-3,
    device: str | None = None,
) -> Path:
    if not volume_paths:
        raise ValueError("At least one 3D volume path is required for training")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Building training dataset on {dev} …", flush=True)
    dataset = ParallelSliceDataset(volume_paths, samples_per_volume=samples_per_volume)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    steps = len(loader)
    print(
        f"Training {epochs} epochs × {steps} batches (batch_size={batch_size}) — "
        f"expect ~{max(2, epochs * steps // 120)}–{max(5, epochs * steps // 40)} min on GPU",
        flush=True,
    )

    model = ConditionalSliceUNet().to(dev)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()

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
            if step == 1 or step % max(steps // 4, 1) == 0 or step == steps:
                print(
                    f"  epoch {epoch + 1}/{epochs} batch {step}/{steps} loss={loss.item():.4f}",
                    flush=True,
                )
        avg = running / max(steps, 1)
        logger.info("epoch %d/%d avg_loss=%.4f", epoch + 1, epochs, avg)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "ml_slice_generator_v1",
        "state_dict": model.state_dict(),
    }
    torch.save(payload, output_path)
    print(f"Saved volume generator → {output_path}", flush=True)
    logger.info("Saved volume generator to %s", output_path)
    return output_path


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Train brain slice-to-volume generator")
    parser.add_argument(
        "--volumes",
        nargs="*",
        type=Path,
        default=[],
        help="NIfTI volumes (default: brain atlas template)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ML_VOLUME_MODEL_DIR / "volume_generator.pt",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--samples", type=int, default=192, help="Training pairs per volume")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    volumes = list(args.volumes)
    if not volumes and ATLAS_BRAIN_TEMPLATE.is_file():
        volumes = [ATLAS_BRAIN_TEMPLATE]
    if not volumes:
        raise SystemExit(
            f"No training volumes. Provide --volumes or run setup_brain_atlas.py "
            f"(template expected at {ATLAS_BRAIN_TEMPLATE})"
        )

    train(
        volumes,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        samples_per_volume=args.samples,
        device=args.device,
    )


if __name__ == "__main__":
    main()
