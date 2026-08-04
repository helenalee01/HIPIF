"""Tier 3: Full physics-informed neural network."""
from __future__ import annotations
import torch
import torch.nn as nn
from ..config import HIPIFConfig


class Tier3PINN(nn.Module):
    def __init__(self, cfg: HIPIFConfig | None = None,
                 input_dim: int | None = None, width: int | None = None):
        super().__init__()
        if cfg is not None:
            input_dim = input_dim or cfg.input_dim
            width = width or cfg.width
        else:
            input_dim = input_dim or 8
            width = width or 118
        self.net = nn.Sequential(
            nn.Linear(input_dim, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, width // 2), nn.ReLU(),
            nn.Linear(width // 2, 1),
        )
        self.input_dim = input_dim; self.width = width

    def forward(self, x): return self.net(x).squeeze(-1)

    @property
    def n_params(self): return sum(p.numel() for p in self.parameters())
