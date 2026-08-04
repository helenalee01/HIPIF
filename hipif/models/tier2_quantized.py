"""Tier 2: Quantized hybrid model with INT8 fake-quantization."""
from __future__ import annotations
import torch
import torch.nn as nn
from .tier3_pinn import Tier3PINN


class _FakeQuantLinear(nn.Module):
    def __init__(self, base: nn.Linear):
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.weight = nn.Parameter(base.weight.detach().clone())
        self.bias = nn.Parameter(base.bias.detach().clone()) if base.bias is not None else None

    def _quantize(self, w):
        scale = w.abs().max().clamp(min=1e-8) / 127.0
        q = torch.round(w / scale).clamp(-128, 127)
        return q * scale

    def forward(self, x):
        return torch.nn.functional.linear(x, self._quantize(self.weight), self.bias)


class Tier2Quantized(nn.Module):
    def __init__(self, base: Tier3PINN):
        super().__init__()
        layers = []
        linears = [m for m in base.net if isinstance(m, nn.Linear)]
        keep_fp32 = {0, len(linears) - 1}
        i_lin = 0
        for m in base.net:
            if isinstance(m, nn.Linear):
                if i_lin in keep_fp32:
                    layers.append(m)
                else:
                    layers.append(_FakeQuantLinear(m))
                i_lin += 1
            else:
                layers.append(m)
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x).squeeze(-1)

    @property
    def memory_mb_estimate(self) -> float:
        b = 0
        for m in self.net:
            if isinstance(m, _FakeQuantLinear):
                b += m.weight.numel() * 1
                if m.bias is not None: b += m.bias.numel() * 4
            elif isinstance(m, nn.Linear):
                b += m.weight.numel() * 4
                if m.bias is not None: b += m.bias.numel() * 4
        return b / (1024 * 1024)
