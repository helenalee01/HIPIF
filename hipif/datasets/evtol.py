"""eVTOL aging dataset loader (paper VII-A).

PATCH 2026-04-29 (Option C):
  Earlier patch defined SoH = 100 * Q_discharge(c) / Q_initial with Q_initial
  taken as the mean of cycles 2..6. The dataset interleaves "mission profile"
  cycles (partial discharges) with "characterization" cycles (full discharges)
  and the early cycles include both, so Q_initial under-counts true fresh
  capacity. The result was SoH ranges like [21, 195]% which after the [50,
  105] clip collapsed every label toward the upper bound — every NN baseline
  trained on those targets converged to a saturated constant.

  This loader now:
    * Anchors Q_initial to the *peak* discharge over an early window (it is a
      better proxy for a true full-discharge characterization cycle than a
      mean over heterogeneous cycles).
    * Drops cycles whose discharge is below 50% of that anchor (partial
      mission cycles polluted the SoH definition with values like 0.21).
    * Caps SoH at 100% by construction so labels live in a physical range
      that the [50, 100] prediction clip can represent.
"""
from __future__ import annotations
import glob, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from ..config import HIPIFConfig
warnings.filterwarnings("ignore")

# Window over which we hunt for a true full-discharge characterization cycle.
# Skip cycle 1 (forming effects) and look at the first 30 cycles for the
# largest QD value, which represents a fresh-cell full discharge.
_INITIAL_WINDOW = 30
# Drop a cycle if its discharge is below this fraction of Q_initial — those
# are partial mission profiles, not capacity measurements. 0.3 keeps all
# cycles whose discharge looks within plausible aging range (LFP loses about
# 50% capacity at end-of-life), and excludes the very-partial flights.
_FULL_DISCHARGE_FRAC = 0.3
# Half-width of the capacity-envelope window used for v2 labels.
_ENVELOPE_WIN = 15


def load_evtol_lab(cfg: HIPIFConfig, verbose: bool = True) -> pd.DataFrame:
    lab_dir = cfg.data_dir / "eVTOL"
    files = sorted(glob.glob(str(lab_dir / "VAH*.csv")))
    files = [f for f in files if "impedance" not in f.lower()]
    if not files:
        raise FileNotFoundError(
            f"[eVTOL] no VAH*.csv under {lab_dir} — synthetic fallback is "
            f"forbidden in the results pipeline (P7)")

    all_dfs = []
    for fp in files:
        cell = Path(fp).stem
        df = pd.read_csv(fp)
        df["cell_id"] = cell
        df.rename(columns={"Ecell_V": "V", "I_mA": "I_mA",
                           "Temperature__C": "T", "cycleNumber": "cycle",
                           "QDischarge_mA_h": "QD_mAh"}, inplace=True)
        df["I_A"] = df["I_mA"] / 1000.0
        all_dfs.append(df)
    data = pd.concat(all_dfs, ignore_index=True)

    # Compute SoH per (cell_id, cycle) from actual QDischarge_mA_h
    soh_map = {}
    valid_keys: set = set()
    for cell_id in data["cell_id"].unique():
        cell_df = data[data["cell_id"] == cell_id]
        per_cycle_q = cell_df.groupby("cycle")["QD_mAh"].max()
        per_cycle_q = per_cycle_q[per_cycle_q > 0].dropna()
        if len(per_cycle_q) == 0:
            if verbose:
                print(f"[eVTOL] {cell_id}: no valid QDischarge -- skipping")
            continue
        early = per_cycle_q[per_cycle_q.index >= 2].iloc[:_INITIAL_WINDOW]
        q_initial = float(early.max()) if len(early) else float(per_cycle_q.max())
        # If even the early-window peak looks low compared to the global peak,
        # the cell didn't have a characterization cycle in the early window —
        # use the global peak as the freshest reference.
        global_peak = float(per_cycle_q.max())
        if global_peak > 1.05 * q_initial:
            q_initial = global_peak
        if q_initial <= 0:
            continue
        full_threshold = _FULL_DISCHARGE_FRAC * q_initial
        # v2 label fix (WP1): a partial mission discharge is NOT a capacity
        # measurement. The label at cycle c is the capacity ENVELOPE — the
        # maximum QD within a +/- _ENVELOPE_WIN cycle window, which tracks the
        # interleaved full-discharge characterization cycles — divided by
        # Q_initial. Feature rows are still restricted to cycles above the
        # partial-discharge threshold.
        cyc_idx = per_cycle_q.index.to_numpy()
        q_vals = per_cycle_q.to_numpy(dtype=float)
        for cyc, q in per_cycle_q.items():
            if q < full_threshold:
                continue
            win = np.abs(cyc_idx - cyc) <= _ENVELOPE_WIN
            ref_q = float(q_vals[win].max())
            soh_val = float(min(100.0 * ref_q / q_initial, 100.0))
            soh_map[(cell_id, cyc)] = soh_val
            valid_keys.add((cell_id, cyc))
        if verbose:
            cell_vals = [v for (cid, _), v in soh_map.items() if cid == cell_id]
            if cell_vals:
                print(f"[eVTOL] {cell_id}: Q_init={q_initial:.0f} mAh, "
                      f"SoH range [{min(cell_vals):.1f}, {max(cell_vals):.1f}]%, "
                      f"n_cycles={len(cell_vals)}")

    # Keep only rows belonging to (cell, cycle) pairs we accepted as full-discharge
    keep_mask = np.array([
        (cid, c) in valid_keys
        for cid, c in zip(data["cell_id"].values, data["cycle"].values)
    ])
    data = data.loc[keep_mask].reset_index(drop=True)
    soh_arr = np.array([
        soh_map[(cid, c)]
        for cid, c in zip(data["cell_id"].values, data["cycle"].values)
    ])
    data["SoH"] = np.clip(soh_arr, 50.0, 100.0)
    return data


def aggregate_cycles(df: pd.DataFrame) -> pd.DataFrame:
    """Cycle-level features (v2): causal V/I statistics + coulomb-counted
    throughput + duration. Measured T retained for SOURCE temperature-head
    supervision and post-hoc verification only (never a model input)."""
    from ..features.schema import add_causal_features
    rows = []
    for cell_id in df["cell_id"].unique():
        cdf = df[df["cell_id"] == cell_id]
        for cyc in cdf["cycle"].dropna().unique():
            s = cdf[cdf["cycle"] == cyc].sort_values("time_s")
            if len(s) < 5:
                continue
            I = s["I_A"].to_numpy(float)
            t = s["time_s"].to_numpy(float)
            # Discharge-only coulomb count (observable BMS counter). trapz of
            # |I| would double-count charge + discharge within one cycle.
            ah = float(np.nanmax(s["QD_mAh"].to_numpy(float)) / 1000.0)
            if not np.isfinite(ah) or ah <= 0:
                ah = float(np.trapezoid(np.clip(-I, 0, None), t) / 3600.0)
            rows.append({
                "cell_id": cell_id, "cycle": int(cyc),
                "V_mean": float(s["V"].mean()), "V_min": float(s["V"].min()),
                "V_max": float(s["V"].max()),
                "I_mean": float(I.mean()), "I_std": float(I.std()),
                "I_abs_mean": float(np.abs(I).mean()),
                "I_rms": float(np.sqrt((I ** 2).mean())),
                "Ah_cycle": ah, "t_cycle_s": float(t[-1] - t[0]),
                "T_amb": 23.0,   # chamber setpoint (registry provenance)
                "T_mean": float(s["T"].mean()), "T_max": float(s["T"].max()),
                "SoH": float(s["SoH"].iloc[0]),
            })
    return add_causal_features(pd.DataFrame(rows))
