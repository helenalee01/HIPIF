#!/usr/bin/env python3
"""exp_dod_curve.py — prefix-truncation stress test (paper Sec. 6.7, Fig. 4, Table S7).

Re-integrates both benchmarks' RAW current traces over only the first
fraction f of each discharge-segment duration and evaluates the coulomb
estimator 100*Q_f/Q_nom against the full-discharge label:

  NASA PCoE : per-cycle discharge records from the released .mat files
              (fields Time / Current_measured / Capacity); trapezoidal
              integration of |I| over the retained prefix; label = the
              stored Capacity field (SoH = 100*C_t/2.0).
  CALCE CS2 : raw Arbin xlsx archives (Data/CALCE_CS2/CS2_*.zip); discharge
              rows with I < -0.01 A grouped by Cycle_Index; the f=1.0
              integration of the same segment defines the label
              (SoH = 100*Q_full/1.1) — the definitional identity of
              paper Table 2, so CALCE at f=1.0 is exact by construction.

Evaluation follows the shared convention: labels and estimates clipped to
[50, 100] independently per sample; unweighted mean of per-cell MAEs
(mean +/- SD across cells reported, as in Table S7). `tightness` is the
mean floor-to-truth ratio using the repair operator's coulomb floor
100*Q_f/(Q_nom*(1+tau)), tau = 0.05.

Per-cell integration results are cached under results/dod_cache/ so the
slow CALCE xlsx pass runs once; pass --force to rebuild.

Usage:
  python scripts/exp_dod_curve.py --data-dir ./Data
Outputs results/dod_curve.csv (per cell x f) and a printed Table-S7 summary.
"""
from __future__ import annotations
import argparse, io, zipfile
from pathlib import Path

import numpy as np
import pandas as pd

FRACTIONS = [1.0, 0.8, 0.6, 0.4, 0.2]
QNOM = {"nasa": 2.0, "calce": 1.1}
TAU = 0.05
CLIP_LO, CLIP_HI = 50.0, 100.0
# Fixed full-discharge-feature reference (direct transfer + PAVA, features
# NOT prefix-recomputed) — the dashed line of Fig. 4; from the factorial
# decomposition "mono only" row (Table 6 / S6).
FIXED_REF = {"nasa": 14.49, "calce": 23.70}


# ------------------------------------------------------------------ NASA
def nasa_cell(mat_path: Path, fractions) -> pd.DataFrame:
    import scipy.io as sio
    name = mat_path.stem
    m = sio.loadmat(str(mat_path), simplify_cells=True)
    cycles = m[name]["cycle"]
    rows, cyc_idx = [], 0
    for c in cycles:
        if not isinstance(c, dict) or c.get("type") != "discharge":
            continue
        cyc_idx += 1
        d = c.get("data", {})
        t = np.atleast_1d(np.asarray(d.get("Time", []), float)).ravel()
        i = np.abs(np.atleast_1d(
            np.asarray(d.get("Current_measured", []), float)).ravel())
        cap = d.get("Capacity", None)
        if cap is None or np.size(cap) == 0 or t.size < 5 or t.size != i.size:
            continue                      # paper Sec. 4.2 segment rules
        cap = float(np.ravel(cap)[0])
        dur = t[-1] - t[0]
        if dur <= 0:
            continue
        for f in fractions:
            mask = t <= t[0] + f * dur
            q = np.trapz(i[mask], t[mask]) / 3600.0 if mask.sum() >= 2 else 0.0
            rows.append({"cell": name, "cycle": cyc_idx, "f": f,
                         "q_f": q, "label_soh": 100.0 * cap / QNOM["nasa"]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ CALCE
def _colmap(columns):
    out = {}
    for c in columns:
        key = str(c).lower().split("(")[0].strip().replace(" ", "_")
        out.setdefault(key, c)
    return out


def calce_cell(zip_path: Path, fractions) -> pd.DataFrame:
    cell = zip_path.stem
    rows, cyc_offset = [], 0
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(n for n in zf.namelist()
                       if n.lower().endswith((".xlsx", ".xls"))
                       and not Path(n).name.startswith("~"))
        for n in names:
            with zf.open(n) as fh:
                xls = pd.ExcelFile(io.BytesIO(fh.read()))
            file_max_cyc = 0
            for sheet in xls.sheet_names:
                if "channel" not in sheet.lower():
                    continue
                df = xls.parse(sheet)
                cols = _colmap(df.columns)
                if not all(k in cols for k in
                           ("current", "test_time", "cycle_index")):
                    continue
                cur = pd.to_numeric(df[cols["current"]], errors="coerce")
                tt = pd.to_numeric(df[cols["test_time"]], errors="coerce")
                ci = pd.to_numeric(df[cols["cycle_index"]], errors="coerce")
                ok = cur.notna() & tt.notna() & ci.notna()
                cur, tt, ci = cur[ok], tt[ok], ci[ok].astype(int)
                if len(ci):
                    file_max_cyc = max(file_max_cyc, int(ci.max()))
                dis = cur < -0.01          # paper Sec. 4.2 discharge rule
                sub = pd.DataFrame({"t": tt[dis], "i": -cur[dis],
                                    "cyc": ci[dis]})
                for cyc, g in sub.groupby("cyc"):
                    g = g.sort_values("t")
                    t = g["t"].to_numpy(float)
                    ii = g["i"].to_numpy(float)
                    if t.size < 5:
                        continue
                    dur = t[-1] - t[0]
                    if dur <= 0:
                        continue
                    per_f = {}
                    for f in fractions:
                        m = t <= t[0] + f * dur
                        per_f[f] = (np.trapz(ii[m], t[m]) / 3600.0
                                    if m.sum() >= 2 else 0.0)
                    label = 100.0 * per_f[1.0] / QNOM["calce"]
                    for f in fractions:
                        rows.append({"cell": cell,
                                     "cycle": cyc_offset + int(cyc),
                                     "f": f, "q_f": per_f[f],
                                     "label_soh": label})
            cyc_offset += file_max_cyc
    return pd.DataFrame(rows)


# ------------------------------------------------------------- evaluation
def summarize(df: pd.DataFrame, qnom: float):
    df = df.copy()
    df["cc"] = np.clip(100.0 * df["q_f"] / qnom, CLIP_LO, CLIP_HI)
    df["y"] = np.clip(df["label_soh"], CLIP_LO, CLIP_HI)
    df["floor"] = 100.0 * df["q_f"] / (qnom * (1.0 + TAU))
    df["err"] = (df["cc"] - df["y"]).abs()
    df["tight"] = df["floor"] / df["y"]
    per_cell = (df.groupby(["cell", "f"])
                  .agg(mae=("err", "mean"), tightness=("tight", "mean"),
                       n_cycles=("err", "size"))
                  .reset_index())
    agg = (per_cell.groupby("f")
                   .agg(mae_mean=("mae", "mean"), mae_sd=("mae", "std"),
                        tightness=("tightness", "mean"))
                   .reset_index()
                   .sort_values("f", ascending=False))
    return per_cell, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./Data")
    ap.add_argument("--out", default="results/dod_curve.csv")
    ap.add_argument("--cache-dir", default="results/dod_cache")
    ap.add_argument("--force", action="store_true",
                    help="rebuild per-cell caches")
    args = ap.parse_args()

    data = Path(args.data_dir)
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    fractions = sorted(set(FRACTIONS) | {1.0}, reverse=True)

    jobs = []
    for p in sorted((data / "NASA_PCoE").glob("B*.mat")):
        jobs.append(("nasa", p.stem, lambda p=p: nasa_cell(p, fractions)))
    for p in sorted((data / "CALCE_CS2").glob("CS2_*.zip")):
        jobs.append(("calce", p.stem, lambda p=p: calce_cell(p, fractions)))
    if not jobs:
        raise SystemExit(f"no NASA .mat or CALCE zips found under {data}")

    frames = []
    for ds, cell, fn in jobs:
        cpath = cache / f"{ds}_{cell}.csv"
        if cpath.exists() and not args.force:
            d = pd.read_csv(cpath)
            print(f"[cache] {ds}/{cell}: {len(d)} rows")
        else:
            print(f"[integrate] {ds}/{cell} ...", flush=True)
            d = fn()
            d.to_csv(cpath, index=False)
        d["dataset"] = ds
        frames.append(d)
    full = pd.concat(frames, ignore_index=True)

    all_percell = []
    for ds in ("nasa", "calce"):
        sub = full[full["dataset"] == ds]
        if sub.empty:
            continue
        per_cell, agg = summarize(sub, QNOM[ds])
        per_cell.insert(0, "dataset", ds)
        all_percell.append(per_cell)
        print(f"\n=== {ds.upper()}  (Table S7 column) ===")
        print(f"{'f':>4} {'CC MAE (pp)':>16} {'tightness':>10}")
        for _, r in agg.iterrows():
            sd = 0.0 if np.isnan(r.mae_sd) else r.mae_sd
            print(f"{r.f:>4.1f} {r.mae_mean:>8.2f} +/- {sd:<5.2f}"
                  f" {r.tightness:>9.2f}")
        print(f"fixed reference (direct transfer + PAVA, "
              f"full-discharge features): {FIXED_REF[ds]:.2f} pp")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(all_percell, ignore_index=True).to_csv(out, index=False)
    print(f"\nwritten -> {out}")


if __name__ == "__main__":
    main()
