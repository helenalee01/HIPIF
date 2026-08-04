#!/usr/bin/env python3
"""Rebuild the CALCE CS2 cycle cache from raw Arbin xlsx archives (WP1/P7).

Reads Data/CALCE_CS2/CS2_*.zip, aggregates per (cell, Cycle_Index):
V/I statistics, I_rms, coulomb-counted discharge Ah_cycle, duration
t_cycle_s, SoH = Ah_cycle / 1.1 * 100. Writes calce_cs2_cycles.csv.
Skips work if the cache already exists (use --force to rebuild)."""
from __future__ import annotations
import argparse, glob, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

NOMINAL_AH = 1.1


def process_cell(zpath: Path, tmp: Path, verbose=True) -> list[dict]:
    cell = zpath.stem
    rows = []
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".xlsx")]
        z.extractall(tmp)
    cyc_offset = 0
    for n in sorted(names):
        try:
            xl = pd.ExcelFile(tmp / n)
            sheet = next(s for s in xl.sheet_names if "channel" in s.lower())
            df = xl.parse(sheet)
        except Exception as e:
            if verbose:
                print(f"  [skip] {n}: {e}")
            continue
        need = {"Cycle_Index", "Current(A)", "Voltage(V)", "Test_Time(s)",
                "Discharge_Capacity(Ah)"}
        if not need.issubset(df.columns):
            continue
        for cyc, s in df.groupby("Cycle_Index"):
            I = s["Current(A)"].to_numpy(float)
            V = s["Voltage(V)"].to_numpy(float)
            t = s["Test_Time(s)"].to_numpy(float)
            dc = s["Discharge_Capacity(Ah)"].to_numpy(float)
            ah = float(np.nanmax(dc) - np.nanmin(dc))
            if not np.isfinite(ah) or ah < 0.1 * NOMINAL_AH or len(s) < 10:
                continue
            rows.append({
                "cell_id": cell, "cycle": int(cyc) + cyc_offset,
                "V_mean": float(np.nanmean(V)), "V_min": float(np.nanmin(V)),
                "V_max": float(np.nanmax(V)),
                "I_mean": float(np.nanmean(I)), "I_std": float(np.nanstd(I)),
                "I_abs_mean": float(np.nanmean(np.abs(I))),
                "I_rms": float(np.sqrt(np.nanmean(I ** 2))),
                "Ah_cycle": ah, "t_cycle_s": float(t[-1] - t[0]),
                "T_amb": 25.0, "T_mean": 25.0, "T_max": 25.0,  # room-temp doc
                "capacity": ah, "SoH": ah / NOMINAL_AH * 100.0,
            })
        if rows:
            cyc_offset = max(r["cycle"] for r in rows if r["cell_id"] == cell)
    if verbose and rows:
        print(f"  {cell}: {len([r for r in rows if r['cell_id']==cell])} cycles")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./Data")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    d = Path(a.data_dir) / "CALCE_CS2"
    cache = d / "calce_cs2_cycles.csv"
    if cache.exists() and not a.force:
        print(f"cache exists: {cache} (use --force to rebuild)")
        return
    tmp = d / "_extract"; tmp.mkdir(exist_ok=True)
    rows = []
    for zp in sorted(glob.glob(str(d / "CS2_*.zip"))):
        print(f"[calce] {zp}")
        rows += process_cell(Path(zp), tmp)
    if not rows:
        raise SystemExit("no cycles parsed — check raw archives (P7)")
    df = pd.DataFrame(rows).sort_values(["cell_id", "cycle"])
    # renumber cycles consecutively per cell
    df["cycle"] = df.groupby("cell_id").cumcount() + 1
    df.to_csv(cache, index=False)
    print(f"wrote {cache}: {len(df)} cycles, {df.cell_id.nunique()} cells")


if __name__ == "__main__":
    main()
