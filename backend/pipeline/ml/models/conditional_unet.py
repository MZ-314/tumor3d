"""Conditional 2.5D U-Net: predict off-slice MRI parallel to anchor plane."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    def __init__(self, inc: int, outc: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(inc, outc, 3, padding=1),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True),
            nn.Conv2d(outc, outc, 3, padding=1),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConditionalSliceUNet(nn.Module):
    """
    Inputs (B, 3, H, W): anchor slice, organ mask, z-offset plane.
    Output (B, 1, H, W): predicted parallel slice.
    """

    def __init__(self, base: int = 32) -> None:
        super().__init__()
        self.enc1 = _ConvBlock(3, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = _ConvBlock(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = _ConvBlock(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = _ConvBlock(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = _ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.out(d1))


def load_volume_generator_checkpoint(
    path: Path,
    *,
    device: torch.device,
) -> tuple[ConditionalSliceUNet, str]:
    payload = torch.load(path, map_location=device, weights_only=False)
    version = "ml_slice_generator_v1"
    if isinstance(payload, dict):
        version = str(payload.get("version", version))
        state = payload["state_dict"]
    else:
        state = payload
    model = ConditionalSliceUNet()
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, version
