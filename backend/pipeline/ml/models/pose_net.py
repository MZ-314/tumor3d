"""Lightweight pose regression CNN (axis + depth along stack)."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class PoseNet(nn.Module):
    """Predict imaging plane axis (3-way) and normalized slice depth."""

    def __init__(self, base: int = 16) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base * 2, base * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.axis_head = nn.Linear(base * 4, 3)
        self.depth_head = nn.Sequential(nn.Linear(base * 4, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.encoder(x).flatten(1)
        return self.axis_head(feat), self.depth_head(feat)


def load_pose_checkpoint(path: Path, *, device: torch.device) -> PoseNet | None:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except Exception:
        return None
    model = PoseNet()
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
