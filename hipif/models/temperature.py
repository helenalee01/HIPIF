"""Physics-informed temperature reconstruction (paper Eq. 7)."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from ..config import HIPIFConfig, KELVIN_OFFSET


def isa_ambient(altitude_m):
    return 288.15 - 0.0065 * np.asarray(altitude_m, dtype=float) - KELVIN_OFFSET


def physics_temperature(voltage, current, altitude_m, R_int=0.05):
    T_amb = isa_ambient(altitude_m)
    Q_gen = (np.asarray(current, dtype=float) ** 2) * R_int
    return T_amb + Q_gen * 0.5


class TemperatureReconstructor(nn.Module):
    def __init__(self, cfg: HIPIFConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.temp_residual_hidden
        self.residual = nn.Sequential(
            nn.Linear(cfg.input_dim + 1, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
            nn.Linear(h, 1),
        )

    def forward(self, x, T_phys):
        z = torch.cat([x, T_phys.unsqueeze(-1)], dim=-1)
        return T_phys + self.residual(z).squeeze(-1)

    def residual_only(self, x, T_phys):
        z = torch.cat([x, T_phys.unsqueeze(-1)], dim=-1)
        return self.residual(z).squeeze(-1)
