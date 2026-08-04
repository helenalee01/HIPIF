"""CALCE CS2 loader — cache-based, fail-fast (P7).

The cycle cache `calce_cs2_cycles.csv` is built by scripts/prepare_calce.py
from the raw Arbin xlsx archives. Ah_cycle provenance: the cache `capacity`
column is the Arbin-integrated per-cycle discharge Ah (coulomb counting of
observable telemetry) and is used as the single-discharge throughput for I4.
NOTE: on full-discharge lab cycling this equals measured capacity, so I4 is
highly informative here — stated in the paper and isolated in ablation E5.
`t_cycle_s` and `I_rms` are approximated from cache statistics when the cache
predates v2 (duration = Ah / |I_mean| * 3600 for constant-current discharge);
rebuild the cache with prepare_calce.py for exact values.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from ..config import HIPIFConfig
from ..features.schema import add_causal_features

CALCE_CELLS = ("CS2_33", "CS2_34", "CS2_35", "CS2_36", "CS2_37", "CS2_38")
NOMINAL_AH = 1.1


def load_calce_cs2(cfg: HIPIFConfig, verbose: bool = True) -> pd.DataFrame:
    cache = cfg.data_dir / "CALCE_CS2" / "calce_cs2_cycles.csv"
    if not cache.exists():
        raise FileNotFoundError(
            f"[CALCE] cycle cache not found: {cache}\n"
            f"Run: python scripts/prepare_calce.py --data-dir {cfg.data_dir}\n"
            f"(synthetic fallback is forbidden in the results pipeline, P7)")
    df = pd.read_csv(cache)
    df["SoH"] = df["SoH"].clip(50, 105)
    if "Ah_cycle" not in df.columns:
        df["Ah_cycle"] = df["capacity"].astype(float)
    if "I_rms" not in df.columns:
        df["I_rms"] = df["I_mean"].abs() + df["I_std"].fillna(0.0)
    if "t_cycle_s" not in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["t_cycle_s"] = (df["Ah_cycle"]
                               / df["I_mean"].abs().clip(lower=1e-3) * 3600.0)
    if "T_amb" not in df.columns:
        df["T_amb"] = 25.0   # registry-documented room-temperature cycling
    df = add_causal_features(df)
    df = df[df["cell_id"].isin(CALCE_CELLS)].reset_index(drop=True)
    if verbose:
        print(f"[CALCE] {len(df)} cycles, {df['cell_id'].nunique()} cells")
    return df
