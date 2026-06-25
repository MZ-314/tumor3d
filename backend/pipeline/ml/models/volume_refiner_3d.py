"""3D U-Net refiner: coarse slab → volumetric brain (Phase 6b)."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class _Conv3dBlock(nn.Module):
    def __init__(self, inc: int, outc: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(inc, outc, 3, padding=1),
            nn.InstanceNorm3d(outc),
            nn.ReLU(inplace=True),
            nn.Conv3d(outc, outc, 3, padding=1),
            nn.InstanceNorm3d(outc),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BrainVolumeRefiner3D(nn.Module):
    """
    Refine a coarse single-slice-expanded volume into a full 3D brain.

    Input (B, 2, D, H, W): coarse intensity + 3D brain envelope.
    Output (B, 1, D, H, W): refined MRI-like volume.
    """

    def __init__(self, base: int = 16) -> None:
        super().__init__()
        self.enc1 = _Conv3dBlock(2, base)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = _Conv3dBlock(base, base * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.bottleneck = _Conv3dBlock(base * 2, base * 4)
        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _Conv3dBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
        self.dec1 = _Conv3dBlock(base * 2, base)
        self.out = nn.Conv3d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.out(d1))


def load_volume_refiner_checkpoint(
    path: Path,
    *,
    device: torch.device,
) -> tuple[BrainVolumeRefiner3D, str]:
    payload = torch.load(path, map_location=device, weights_only=False)
    version = "ml_volume_refiner_v1"
    if isinstance(payload, dict):
        version = str(payload.get("version", version))
        state = payload["state_dict"]
    else:
        state = payload
    model = BrainVolumeRefiner3D()
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, version
