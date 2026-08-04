"""Uncertainty head (paper Eq. 12)."""
from __future__ import annotations
import torch
import torch.nn as nn
from ..config import HIPIFConfig


class UncertaintyHead(nn.Module):
    def __init__(self, cfg: HIPIFConfig, n_dropout_passes: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cfg = cfg; self.n_passes = n_dropout_passes
        h = cfg.uncertainty_hidden
        self.var_head = nn.Sequential(
            nn.Linear(cfg.width // 2, h), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h, 1), nn.Softplus(),
        )

    def forward(self, hidden): return self.var_head(hidden).squeeze(-1)

    @staticmethod
    def combine(c_phys, c_unc, c_temp, c_hist,
                wp=0.4, wu=0.3, wt=0.2, wh=0.1):
        s = wp + wu + wt + wh
        return (wp * c_phys + wu * c_unc + wt * c_temp + wh * c_hist) / s
