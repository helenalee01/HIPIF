#!/usr/bin/env python3
"""exp_factorial_repair.py — factorial repair-operator ablation (review R5).
Renamed from exp_factorial_projection.py to match the manuscript terminology.

Applies every combination of {monotone (PAVA), rate cap, coulomb floor} —
with the [50,100] clip always applied last, since any SoH output must lie in
physical bounds — to the STORED direct-transfer predictions, isolating each
operator's contribution without any adaptation confound.

Usage:
  python scripts/exp_factorial_repair.py — factorial repair-operator ablation (renamed from exp_factorial_projection.py to match the manuscript terminology) --data-dir ./Data \
      --pred-csv results/primary2/predictions_per_sample.csv
Outputs results/factorial_projection.csv and a printed summary.
"""
from __future__ import annotations
import argparse, itertools, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hipif.config import HIPIFConfig, ChemistryParams
from hipif.datasets import load_nasa_pcoe, load_calce_cs2
from hipif.adaptation.projector import _pava_nonincreasing

CHEM = {"nasa": "Li-ion", "calce": "LCO"}


def repair(raw, cyc, T, ah, chem, mono, rate, floor):
    z = raw.astype(float).copy()
    if mono:
        z = _pava_nonincreasing(z)
    if rate:
        k = np.maximum(chem.k_max(T), 1e-4)
        for j in range(1, len(z)):
            dc = max(cyc[j] - cyc[j - 1], 1.0)
            lo, hi = z[j - 1] - k[j] * dc, z[j - 1] + (0.0 if mono else k[j] * dc)
            z[j] = min(max(z[j], lo), hi if not mono else z[j])
            if mono and z[j] < lo:
                z[j] = min(lo, z[j - 1])
    if floor:
        f = ah / chem.Q_nom * 100.0 / (1 + chem.energy_tol) * (1 + 1e-6)
        z = np.maximum(z, f)
        for j in range(len(z) - 2, -1, -1):   # restore monotone by raising
            if mono:
                z[j] = max(z[j], z[j + 1])
    return np.clip(z, chem.soh_min, chem.soh_max)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./Data")
    ap.add_argument("--pred-csv", required=True)
    ap.add_argument("--out", default="results/factorial_projection.csv")
    a = ap.parse_args()

    cfg = HIPIFConfig(data_dir=Path(a.data_dir))
    data = {"nasa": load_nasa_pcoe(cfg, verbose=False),
            "calce": load_calce_cs2(cfg, verbose=False)}
    pr = pd.read_csv(a.pred_csv)
    pr = pr[pr.method == "direct"]

    rows = []
    for ds, dfp in pr.groupby("dataset"):
        chem = ChemistryParams.from_registry(CHEM[ds])
        df = data[ds]
        for (cell, seed), p in dfp.groupby(["cell", "seed"]):
            p = p.sort_values("cycle")
            d = df[df.cell_id == cell].sort_values("cycle")
            if len(p) != len(d):
                continue
            raw = p.pred_raw.to_numpy(); y = p.y_true.to_numpy()
            cyc = p.cycle.to_numpy(float)
            ah = d.Ah_cycle.to_numpy()
            T = chem.ambient_C + d.I_rms.to_numpy() ** 2 * chem.R_int / chem.hA
            for mono, rate, floor in itertools.product([0, 1], repeat=3):
                z = repair(raw, cyc, T, ah, chem, mono, rate, floor)
                name = "+".join([n for n, on in
                                 (("mono", mono), ("rate", rate),
                                  ("floor", floor)) if on]) or "clip_only"
                rows.append(dict(dataset=ds, cell=cell, seed=seed,
                                 combo=name, mono=mono, rate=rate,
                                 floor=floor,
                                 mae=float(np.abs(z - y).mean())))
    res = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out, index=False)
    print(res.groupby(["dataset", "combo"]).mae.agg(["mean", "std"])
             .round(2).sort_values("mean"))


if __name__ == "__main__":
    main()
