"""Deterministic synthetic data generators (fallback)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from ..config import HIPIFConfig


def make_synthetic_evtol(cfg: HIPIFConfig, n_cells: int = 12,
                         n_cycles: int = 200, samples_per_cycle: int = 100):
    rng = np.random.default_rng(cfg.seed)
    rows = []
    for cid in range(n_cells):
        T_amb = 22.0 + rng.uniform(-3, 5)
        soh = 100.0
        for cyc in range(1, n_cycles + 1):
            I_mean = rng.uniform(0.8, 1.6)
            T_mean = T_amb + I_mean ** 2 * 0.5 + rng.normal(0, 0.5)
            fade = (17.0 / n_cycles) * (1.0 + (T_mean - 25.0) / 100.0) \
                                     * (1.0 + (I_mean / 20.0) * 0.5)
            soh = max(60.0, soh - fade)
            for _ in range(samples_per_cycle):
                rows.append({
                    "cell_id": f"SYN{cid:02d}", "cycle": cyc,
                    "V": rng.uniform(3.0, 4.2),
                    "I_A": I_mean + rng.normal(0, 0.05),
                    "I_mA": (I_mean + rng.normal(0, 0.05)) * 1000,
                    "T": T_mean + rng.normal(0, 0.2),
                    "QD_mAh": -1, "SoH": soh,
                })
    return pd.DataFrame(rows)


def make_synthetic_field(cfg: HIPIFConfig, n_flights: int = 12,
                         samples_per_flight: int = 720):
    rng = np.random.default_rng(cfg.seed + 1)
    packs = ["Old-A", "Old-B", "Mid-A", "Mid-B", "Mid-C", "Mid-D", "Mid-E",
             "New-A", "New-B", "New-C", "New-D", "New-E"]
    soh_map = {p: rng.uniform(83.0, 96.0) for p in packs}
    rows = []
    for f, pack in enumerate(packs[:n_flights]):
        for t in range(samples_per_flight):
            alt = 30 + 50 * np.sin(t / 60) + rng.normal(0, 3)
            V = 22.5 + rng.normal(0, 0.4) - t * 1e-4
            I = 8 + 4 * abs(np.sin(t / 30)) + rng.normal(0, 0.5)
            T_amb = 288.15 - 0.0065 * alt - 273.15
            T_phys = T_amb + I ** 2 * 0.05 * 0.5
            base = soh_map[pack]
            soh_ref = max(base - t * 1e-4, base - 1)
            rows.append({"battery_pack": pack, "V": V, "I_A": I,
                         "Alt": alt, "T_phys": T_phys,
                         "SoH_reference": soh_ref})
    return pd.DataFrame(rows)
