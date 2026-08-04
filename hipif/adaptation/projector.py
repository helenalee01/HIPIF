"""Constrained projection Pi_F onto the feasible set (WP2 / 5.1).

Minimal-change principle: given a teacher trajectory per cell (ordered by
cycle), return the closest trajectory (L2, via pool-adjacent-violators)
satisfying:
  I1 non-increasing (up to mono_eps handled by strict PAVA — projection is
     conservative: fully monotone),
  I2 decrement rate cap  z[j] >= z[j-1] - k_max(T_hat[j]) * dcycle,
  I3 bounds [soh_min, soh_max].
One operator for every dataset — no dataset-name branching (P3).
"""
from __future__ import annotations
import numpy as np
from ..config import ChemistryParams


def _pava_nonincreasing(y: np.ndarray) -> np.ndarray:
    """L2 isotonic regression, non-increasing, via PAVA on -y."""
    z = -np.asarray(y, dtype=float)
    n = len(z)
    level = z.copy(); weight = np.ones(n)
    # pool adjacent violators for non-decreasing fit of z
    vals = []; wts = []
    for i in range(n):
        v, w = z[i], 1.0
        while vals and vals[-1] > v:
            pv, pw = vals.pop(), wts.pop()
            v = (v * w + pv * pw) / (w + pw)
            w = w + pw
        vals.append(v); wts.append(w)
    out = np.empty(n)
    k = 0
    for v, w in zip(vals, wts):
        out[k:k + int(w)] = v
        k += int(w)
    return -out


def project_feasible(soh: np.ndarray, cell_ids, cycles,
                     T_hat_C: np.ndarray, chem: ChemistryParams,
                     ah_cycle=None) -> np.ndarray:
    """Project teacher predictions onto the feasible set, per cell.
    Order: PAVA (I1) -> decrement-rate cap (I2) -> coulomb lower bound (I4,
    when Ah_cycle given: SoH >= Ah/Q_nom*100/(1+tol)) -> backward max pass
    (restores I1 by only RAISING values, preserving the I4 floor and never
    increasing any decrement) -> bounds clip (I3)."""
    soh = np.asarray(soh, dtype=float)
    z = soh.copy()
    cell_ids = np.asarray(cell_ids)
    cycles = np.asarray(cycles, dtype=float)
    k_max = np.maximum(chem.k_max(T_hat_C), 1e-4)
    ah = None if ah_cycle is None else np.asarray(ah_cycle, dtype=float)
    for cid in np.unique(cell_ids):
        m = np.where(cell_ids == cid)[0]
        idxs = m[np.argsort(cycles[m])]
        seq = _pava_nonincreasing(soh[idxs])              # I1
        for j in range(1, len(idxs)):                     # I2 decrement cap
            dc = max(cycles[idxs[j]] - cycles[idxs[j - 1]], 1.0)
            floor = seq[j - 1] - k_max[idxs[j]] * dc
            if seq[j] < floor:
                seq[j] = min(floor, seq[j - 1])
        if ah is not None:                                # I4 coulomb floor
            # tiny relative lift keeps float32 round-trips strictly feasible
            f = (ah[idxs] / chem.Q_nom * 100.0 / (1.0 + chem.energy_tol)
                 * (1.0 + 1e-6))
            f = np.where(np.isfinite(f), f, -np.inf)
            seq = np.maximum(seq, f)
            for j in range(len(idxs) - 2, -1, -1):        # restore I1
                seq[j] = max(seq[j], seq[j + 1])
        z[idxs] = seq
    return np.clip(z, chem.soh_min, chem.soh_max)         # I3
