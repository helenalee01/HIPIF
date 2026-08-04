"""Cell-cluster statistics + uncertainty calibration metrics (4.4 / WP5).

The generalisation unit is the held-out TARGET CELL; seeds are repeated
measurements within a cell. Method comparisons therefore use hierarchical
(cluster) bootstrap over cells (outer) and seeds (inner), plus exact paired
sign/permutation tests as small-n sensitivity, plus effect sizes.
"""
from __future__ import annotations
from itertools import product
import numpy as np
import pandas as pd


# ---------------------------------------------------------- paired inference
def paired_cell_seed(df: pd.DataFrame, method_a: str, method_b: str,
                     metric: str = "mae") -> pd.DataFrame:
    """df columns: method, cell, seed, <metric>. Returns wide per (cell,seed)
    paired frame with delta = a - b (negative => a better for error metrics)."""
    a = df[df.method == method_a].set_index(["cell", "seed"])[metric]
    b = df[df.method == method_b].set_index(["cell", "seed"])[metric]
    j = pd.concat({"a": a, "b": b}, axis=1).dropna()
    j["delta"] = j["a"] - j["b"]
    return j.reset_index()


def hierarchical_bootstrap_ci(paired: pd.DataFrame, n_boot: int = 10000,
                              seed: int = 0, alpha: float = 0.05):
    """Cluster bootstrap: resample cells with replacement; within each drawn
    cell resample its seeds; statistic = mean over cells of per-cell mean
    delta. Returns (point, lo, hi)."""
    rng = np.random.default_rng(seed)
    cells = paired["cell"].unique()
    per_cell = {c: paired.loc[paired.cell == c, "delta"].to_numpy()
                for c in cells}
    point = float(np.mean([v.mean() for v in per_cell.values()]))
    stats = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.choice(cells, size=len(cells), replace=True)
        vals = [rng.choice(per_cell[c], size=len(per_cell[c]),
                           replace=True).mean() for c in draw]
        stats[b] = np.mean(vals)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def exact_paired_sign_test(paired: pd.DataFrame) -> float:
    """Exact two-sided sign-flip permutation on per-cell mean deltas
    (feasible for the small cell counts here)."""
    d = paired.groupby("cell")["delta"].mean().to_numpy()
    n = len(d)
    obs = abs(d.mean())
    if n > 20:   # fall back to Monte-Carlo
        rng = np.random.default_rng(1)
        flips = rng.choice([-1, 1], size=(20000, n))
        stats = np.abs((flips * d).mean(axis=1))
        return float((stats >= obs - 1e-12).mean())
    count = 0; total = 2 ** n
    for signs in product([-1, 1], repeat=n):
        if abs(np.mean(np.array(signs) * d)) >= obs - 1e-12:
            count += 1
    return count / total


def cliff_delta(paired: pd.DataFrame) -> float:
    """Cliff's delta on per-cell mean deltas vs 0 (paired one-sample form:
    fraction(neg) - fraction(pos) => negative favours method A on error)."""
    d = paired.groupby("cell")["delta"].mean().to_numpy()
    return float((np.sum(d < 0) - np.sum(d > 0)) / max(len(d), 1))


def compare_methods(df: pd.DataFrame, method_a: str, method_b: str,
                    metric: str = "mae", n_boot: int = 10000) -> dict:
    p = paired_cell_seed(df, method_a, method_b, metric)
    if p.empty:
        return {"error": "no paired rows"}
    point, lo, hi = hierarchical_bootstrap_ci(p, n_boot=n_boot)
    return {"method_a": method_a, "method_b": method_b, "metric": metric,
            "delta_point": point, "delta_ci_lo": lo, "delta_ci_hi": hi,
            "sign_test_p": exact_paired_sign_test(p),
            "cliff_delta": cliff_delta(p),
            "n_cells": int(p["cell"].nunique()),
            "n_pairs": int(len(p)),
            "a_better_ci_excludes_0": bool(hi < 0)}


# -------------------------------------------------- interval calibration
def interval_from_ensemble(preds: np.ndarray, alpha: float = 0.10):
    """90% normal-approximation interval from a seed ensemble
    (preds: [n_members, n_samples])."""
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    mu = preds.mean(0); sd = preds.std(0, ddof=1) + 1e-6
    return mu, mu - z * sd, mu + z * sd


def picp(y, lo, hi) -> float:
    return float(np.mean((y >= lo) & (y <= hi)))


def mpiw(lo, hi) -> float:
    return float(np.mean(hi - lo))


def wis(y, lo, hi, alpha: float = 0.10) -> float:
    """Winkler interval score (lower is better)."""
    w = hi - lo
    below = (lo - y) * (y < lo)
    above = (y - hi) * (y > hi)
    return float(np.mean(w + (2 / alpha) * (below + above)))
