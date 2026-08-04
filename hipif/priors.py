"""Label-free chemistry-parameter refinement from ADAPTATION cells only (P3/P4).

v1 hard-coded per-dataset fade lines and blend ratios are removed (A2).
The only remaining role of this module is H1 calibration support:
refining the kinetic envelope k0 from the source model's own direct-transfer
predictions on the adaptation cells — no target labels, no held-out cell,
no EOL information (A3).
"""
from __future__ import annotations
import numpy as np


def estimate_k0_labelfree(pred: np.ndarray, cycles, cell_ids,
                          headroom: float = 1.5,
                          lo: float = 0.05, hi: float = 2.0) -> dict:
    """Envelope on |dSoH_pred/dcycle| over adaptation cells.

    Per cell: least-squares slope of direct-transfer prediction on cycle
    index (uses only observed cycles of adaptation cells — causal, no max
    cycle of any held-out cell). k0_hat = headroom * max per-cell |slope|,
    clamped to [lo, hi] pp/cycle.
    Returns dict with value + provenance for the run manifest."""
    pred = np.asarray(pred, dtype=float)
    cycles = np.asarray(cycles, dtype=float)
    cell_ids = np.asarray(cell_ids)
    slopes = []
    for cid in np.unique(cell_ids):
        m = cell_ids == cid
        if m.sum() < 5:
            continue
        c = cycles[m]; p = pred[m]
        c = c - c.mean()
        denom = float((c ** 2).sum())
        if denom <= 0:
            continue
        slopes.append(abs(float((c * (p - p.mean())).sum() / denom)))
    if not slopes:
        raise ValueError("estimate_k0_labelfree: no adaptation cell with >=5 "
                         "samples (fail-fast, P7)")
    k0 = float(np.clip(headroom * max(slopes), lo, hi))
    return {"k0_pp_per_cycle": k0, "floor_applied": bool(k0 <= lo + 1e-12),
            "provenance": ("labelfree_envelope: headroom "
                           f"{headroom} x max per-adaptation-cell |LS slope| "
                           f"of direct-transfer predictions; slopes="
                           f"{[round(s, 4) for s in slopes]}")}
