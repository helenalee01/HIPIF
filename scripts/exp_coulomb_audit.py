#!/usr/bin/env python3
"""exp_coulomb_audit.py — the coulomb-tautology audit (paper Sec. 4, Table 2;
CC rows of Tables 4, S3, S4; clipping analysis of Table S2).

Computes, per target dataset, the training-free estimator
    SoH_hat_t = 100 * Q_t / Q_nom
against the benchmark label, before and after the shared evaluation clipping
(labels and estimator clipped to [50, 100] independently per sample), plus the
monotone-smoothed variant (per-cell non-increasing PAVA applied to the raw
count). Aggregation follows the paper's unified convention: unweighted mean of
per-cell MAEs.

No model, labels, or adaptation are involved: Q_t is the per-cycle delivered
charge already present in the cycle cache (`Ah_cycle`), i.e. the cycler
discharge-capacity channel on CALCE (definitional identity with the label
numerator) and the trapezoidal reintegration of the raw current trace on NASA
(implementation-independent comparison; see paper Sec. 4.1).

Usage:
  python scripts/exp_coulomb_audit.py --data-dir ./Data
Outputs results/coulomb_audit.csv and a printed per-cell / per-dataset summary.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hipif.config import HIPIFConfig
from hipif.datasets import load_nasa_pcoe, load_calce_cs2
from hipif.adaptation.projector import _pava_nonincreasing

CHEM = {"nasa": "Li-ion", "calce": "LCO"}
CLIP_LO, CLIP_HI = 50.0, 100.0


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def audit_dataset(name: str, df: pd.DataFrame, q_nom: float) -> pd.DataFrame:
    rows = []
    for cell, sub in df.groupby("cell_id", sort=True):
        sub = sub.sort_values("cycle") if "cycle" in sub.columns else sub
        y_raw = sub["SoH"].to_numpy(float)
        cc_raw = 100.0 * sub["Ah_cycle"].to_numpy(float) / q_nom
        y_clip = np.clip(y_raw, CLIP_LO, CLIP_HI)
        cc_clip = np.clip(cc_raw, CLIP_LO, CLIP_HI)
        cc_mono = np.clip(_pava_nonincreasing(cc_raw.copy()), CLIP_LO, CLIP_HI)
        corr = float(np.corrcoef(cc_raw, y_clip)[0, 1])
        rows.append({
            "dataset": name, "cell": cell, "n_cycles": len(sub),
            "mae_unclipped": mae(cc_raw, y_clip),          # Table S2 left column
            "mae_clipped": mae(cc_clip, y_clip),           # evaluation convention
            "mae_mono_smoothed": mae(cc_mono, y_clip),     # Table 4 second row
            "corr": corr,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./Data")
    ap.add_argument("--out", default="results/coulomb_audit.csv")
    args = ap.parse_args()

    frames = []
    for name, loader in (("nasa", load_nasa_pcoe), ("calce", load_calce_cs2)):
        cfg = HIPIFConfig(chemistry=CHEM[name], data_dir=Path(args.data_dir))
        df = loader(cfg)
        q_nom = cfg.chem.Q_nom
        d = audit_dataset(name, df, q_nom)
        frames.append(d)
        print(f"\n=== {name.upper()}  (Q_nom = {q_nom} Ah) ===")
        print(d.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(f"cell-level mean MAE  unclipped {d.mae_unclipped.mean():.2f}  "
              f"clipped {d.mae_clipped.mean():.2f}  "
              f"mono-smoothed {d.mae_mono_smoothed.mean():.2f} pp   "
              f"corr range [{d['corr'].min():.4f}, {d['corr'].max():.4f}]")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames).to_csv(out, index=False)
    print(f"\nwritten -> {out}")


if __name__ == "__main__":
    main()
