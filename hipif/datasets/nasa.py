"""NASA PCoE loader — real data only, fail-fast (P7).

Per discharge cycle, computes from the raw time series:
  V/I statistics, I_rms, coulomb-counted throughput Ah_cycle = trapz(|I|,t)/3600,
  duration t_cycle_s, and (post-hoc only) T_mean/T_max and SoH from Capacity.
Measured temperature columns are retained in the frame for POST-HOC
verification only; hipif.features.schema blocks them from the target path.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from ..config import HIPIFConfig
from ..features.schema import add_causal_features

NASA_CELLS = ("B0005", "B0006", "B0007", "B0018")
NOMINAL_AH = 2.0


def load_nasa_pcoe(cfg: HIPIFConfig, verbose: bool = True) -> pd.DataFrame:
    nasa_dir = cfg.data_dir / "NASA_PCoE"
    missing = [c for c in NASA_CELLS if not (nasa_dir / f"{c}.mat").exists()]
    if missing:
        raise FileNotFoundError(
            f"[NASA] missing {missing} under {nasa_dir} — synthetic fallback "
            f"is forbidden in the results pipeline (P7)")
    import scipy.io as sio
    rows = []
    for cell in NASA_CELLS:
        mat = sio.loadmat(str(nasa_dir / f"{cell}.mat"), simplify_cells=True)
        idx = 0
        for c in mat[cell]["cycle"]:
            if c.get("type") != "discharge":
                continue
            d = c["data"]
            V = np.asarray(d["Voltage_measured"], dtype=float)
            I = np.asarray(d["Current_measured"], dtype=float)
            T = np.asarray(d["Temperature_measured"], dtype=float)
            t = np.asarray(d["Time"], dtype=float)
            cap = float(d.get("Capacity", np.nan))
            if not np.isfinite(cap) or cap <= 0 or len(V) < 5:
                continue
            idx += 1
            ah = float(np.trapezoid(np.abs(I), t) / 3600.0)
            rows.append({
                "cell_id": cell, "cycle": idx,
                "V_mean": float(V.mean()), "V_min": float(V.min()),
                "V_max": float(V.max()),
                "I_mean": float(I.mean()), "I_std": float(I.std()),
                "I_abs_mean": float(np.abs(I).mean()),
                "I_rms": float(np.sqrt((I ** 2).mean())),
                "Ah_cycle": ah, "t_cycle_s": float(t[-1] - t[0]),
                "T_amb": float(c.get("ambient_temperature", 24.0)),
                # post-hoc only:
                "T_mean": float(T.mean()), "T_max": float(T.max()),
                "SoH": cap / NOMINAL_AH * 100.0,
            })
    df = pd.DataFrame(rows)
    df["SoH"] = df["SoH"].clip(50, 105)
    df = add_causal_features(df)
    if verbose:
        print(f"[NASA] {len(df)} discharge cycles, "
              f"{df['cell_id'].nunique()} cells")
    return df
