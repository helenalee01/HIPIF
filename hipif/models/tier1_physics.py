"""Tier 1: Analytical physics-only estimator."""
from __future__ import annotations
import numpy as np
from ..config import HIPIFConfig, KELVIN_OFFSET, R_GAS


class Tier1Physics:
    def __init__(self, cfg: HIPIFConfig, soh_initial: float = 100.0,
                 total_fade_pct: float = 17.0):
        self.cfg = cfg
        self.soh_initial = soh_initial
        self.total_fade = total_fade_pct
        self.nparams = 0

    def fit(self, *args, **kwargs): return self

    def predict_per_cycle(self, cycle, T_C, I_A, max_cycle):
        cycle = np.asarray(cycle, dtype=float)
        T_K = np.asarray(T_C, dtype=float) + KELVIN_OFFSET
        I_abs = np.abs(np.asarray(I_A, dtype=float))
        T_ref = 25.0 + KELVIN_OFFSET
        base = self.total_fade / max(max_cycle, 1)
        temp_factor = np.exp(
            -self.cfg.E_a / R_GAS * (1.0 / np.clip(T_K, 220.0, 350.0) - 1.0 / T_ref)
        )
        crate_factor = 1.0 + 0.5 * np.clip(I_abs / 20.0, 0.0, 0.6)
        per_step = base * temp_factor * crate_factor
        order = np.argsort(cycle)
        cum = np.zeros_like(per_step)
        cum[order] = np.cumsum(per_step[order])
        return np.clip(self.soh_initial - cum, 50.0, 100.0)

    def predict(self, X, cycle=None):
        feat = list(self.cfg.feat_cols)
        idx_T = feat.index("T_mean"); idx_I = feat.index("I_mean")
        if cycle is None:
            # cycle_norm no longer in features; assume cycle index = row index
            cycle = np.arange(len(X))
        max_c = max(int(np.asarray(cycle).max()), 1)
        return self.predict_per_cycle(cycle, X[:, idx_T], X[:, idx_I], max_c)
