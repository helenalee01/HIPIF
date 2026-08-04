"""Physics constraints I1-I4, v2 specification (WP3 / 5.2).

Units and windows
-----------------
I1 Monotonic trend    : SoH[j] - SoH[j-1] <= mono_eps  [pp], per cell,
                        samples ordered by cycle. mono_eps (default 0.5 pp)
                        absorbs capacity-test repeatability noise.
I2 Kinetic rate bound : |SoH[j]-SoH[j-1]| / (cycle[j]-cycle[j-1])
                        <= k(T_hat[j]) [pp per cycle], with
                        k(T) = k0 * exp(-Ea/R * (1/T_K - 1/T_ref_K)).
                        k0 [pp/cycle at 298.15 K] and Ea [J/mol] come from the
                        chemistry registry (provenance recorded).
I3 Bounds             : SoH in [soh_min, soh_max] = [50, 100] %.
I4 Energy consistency : single-discharge window — Ah delivered in cycle j
                        (coulomb-counted from telemetry) must satisfy
                        Ah_cycle[j] <= Q_nom * SoH[j]/100 * (1 + tol).
                        Missing Ah_cycle while I4 is active is a HARD FAIL
                        (P7); constraints silently skipping is forbidden.

Persistence filter: a violation is charged only if it persists for
`persistence_window` consecutive samples, applied PER CELL (v1 bug: the
filter ran across concatenated cells).

Reporting (P6): report BOTH raw-prediction PVR (primary feasibility metric)
and projected-output PVR (operational, trivially ~0 by construction).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from .config import HIPIFConfig, ChemistryParams, KELVIN_OFFSET, R_GAS


class MissingConstraintInput(RuntimeError):
    pass


def _per_cell_ordered(cell_ids, cycles, n):
    """Yield (cell_id, ordered_indices) per cell; falls back to one group."""
    if cell_ids is None:
        yield None, np.arange(n)
        return
    cell_ids = np.asarray(cell_ids)
    cycles = np.asarray(cycles) if cycles is not None else np.arange(n)
    for cid in np.unique(cell_ids):
        m = np.where(cell_ids == cid)[0]
        yield cid, m[np.argsort(cycles[m])]


def monotonicity_indicator(soh, chem: ChemistryParams,
                           cell_ids=None, cycles=None) -> np.ndarray:
    soh = np.asarray(soh, dtype=float); n = len(soh)
    ok = np.ones(n, dtype=bool)
    for _, idxs in _per_cell_ordered(cell_ids, cycles, n):
        for j in range(1, len(idxs)):
            if soh[idxs[j]] > soh[idxs[j - 1]] + chem.mono_eps:
                ok[idxs[j]] = False
    return ok


def arrhenius_indicator(soh, temperature_C, chem: ChemistryParams,
                        cell_ids=None, cycles=None) -> np.ndarray:
    """Rate bound in pp PER CYCLE with explicit Delta-cycle."""
    soh = np.asarray(soh, dtype=float); n = len(soh)
    if temperature_C is None:
        raise MissingConstraintInput(
            "I2 active but reconstructed temperature T_hat is None (P7)")
    k_max = np.maximum(chem.k_max(temperature_C), 1e-4)  # pp/cycle floor
    cyc = (np.asarray(cycles, dtype=float) if cycles is not None
           else np.arange(n, dtype=float))
    ok = np.ones(n, dtype=bool)
    for _, idxs in _per_cell_ordered(cell_ids, cyc, n):
        for j in range(1, len(idxs)):
            dc = max(cyc[idxs[j]] - cyc[idxs[j - 1]], 1.0)
            rate = abs(soh[idxs[j]] - soh[idxs[j - 1]]) / dc
            if rate > k_max[idxs[j]]:
                ok[idxs[j]] = False
    return ok


def bounds_indicator(soh, chem: ChemistryParams) -> np.ndarray:
    soh = np.asarray(soh, dtype=float)
    return (soh >= chem.soh_min) & (soh <= chem.soh_max)


def energy_indicator(soh, ah_cycle, chem: ChemistryParams) -> np.ndarray:
    """Single-discharge window: delivered Ah <= usable capacity * (1+tol).
    ah_cycle is coulomb-counted from V/I telemetry (observable), NOT from a
    capacity label. NOTE (documented in the paper): on full-discharge lab
    cycles this bound is highly informative because delivered Ah approaches
    true capacity — this is physics-as-supervision from observable telemetry,
    and its contribution is isolated in the constraint ablation (E5)."""
    if ah_cycle is None:
        raise MissingConstraintInput(
            "I4 active but Ah_cycle is None — loaders must coulomb-count "
            "per-cycle throughput from telemetry (hard fail, P7)")
    soh = np.asarray(soh, dtype=float)
    ah = np.asarray(ah_cycle, dtype=float)
    cap = chem.Q_nom * soh / 100.0
    ok = ah <= cap * (1.0 + chem.energy_tol)
    ok |= ~np.isfinite(ah)          # non-finite telemetry rows: no evidence
    return ok


@dataclass
class PhysicsConstraints:
    chem: ChemistryParams

    def raw_mask(self, soh, temperature_C=None, ah_cycle=None,
                 cell_ids=None, cycles=None,
                 which: Tuple[str, ...] = ("I1", "I2", "I3", "I4")) -> np.ndarray:
        """Per-constraint AND, before persistence filtering."""
        n = len(soh)
        ok = np.ones(n, dtype=bool)
        if "I1" in which:
            ok &= monotonicity_indicator(soh, self.chem, cell_ids, cycles)
        if "I2" in which:
            ok &= arrhenius_indicator(soh, temperature_C, self.chem,
                                      cell_ids, cycles)
        if "I3" in which:
            ok &= bounds_indicator(soh, self.chem)
        if "I4" in which:
            ok &= energy_indicator(soh, ah_cycle, self.chem)
        return ok

    def feasibility_mask(self, soh, temperature_C=None, ah_cycle=None,
                         cell_ids=None, cycles=None,
                         which=("I1", "I2", "I3", "I4")) -> np.ndarray:
        ok = self.raw_mask(soh, temperature_C, ah_cycle, cell_ids, cycles, which)
        return self._persistence_filter(ok, cell_ids, cycles)

    def _persistence_filter(self, ok, cell_ids, cycles) -> np.ndarray:
        """Forgive violation runs shorter than persistence_window,
        applied independently within each cell (v2 fix)."""
        w = self.chem.persistence_window
        if w <= 1:
            return ok
        out = ok.copy()
        n = len(ok)
        for _, idxs in _per_cell_ordered(cell_ids, cycles, n):
            viol = ~ok[idxs]
            i = 0
            while i < len(idxs):
                if viol[i]:
                    j = i
                    while j < len(idxs) and viol[j]:
                        j += 1
                    if (j - i) < w:
                        out[idxs[i:j]] = True
                    i = j
                else:
                    i += 1
        return out


def compute_pvr(soh, chem: ChemistryParams, temperature_C=None, ah_cycle=None,
                cell_ids=None, cycles=None,
                which=("I1", "I2", "I3", "I4"), persistence=True) -> float:
    pc = PhysicsConstraints(chem)
    if persistence:
        ok = pc.feasibility_mask(soh, temperature_C, ah_cycle,
                                 cell_ids, cycles, which)
    else:
        ok = pc.raw_mask(soh, temperature_C, ah_cycle, cell_ids, cycles, which)
    return float((~ok).mean() * 100.0)


def constraint_residuals(soh, chem: ChemistryParams, temperature_C=None,
                         ah_cycle=None, cell_ids=None, cycles=None) -> dict:
    """Violation magnitudes (not just binary), for P6 reporting."""
    soh = np.asarray(soh, dtype=float); n = len(soh)
    cyc = (np.asarray(cycles, dtype=float) if cycles is not None
           else np.arange(n, dtype=float))
    r1 = np.zeros(n); r2 = np.zeros(n)
    if temperature_C is not None:
        k_max = np.maximum(chem.k_max(temperature_C), 1e-4)
    for _, idxs in _per_cell_ordered(cell_ids, cyc, n):
        for j in range(1, len(idxs)):
            up = soh[idxs[j]] - soh[idxs[j - 1]] - chem.mono_eps
            r1[idxs[j]] = max(up, 0.0)
            if temperature_C is not None:
                dc = max(cyc[idxs[j]] - cyc[idxs[j - 1]], 1.0)
                rate = abs(soh[idxs[j]] - soh[idxs[j - 1]]) / dc
                r2[idxs[j]] = max(rate - k_max[idxs[j]], 0.0)
    r3 = np.maximum(soh - chem.soh_max, 0) + np.maximum(chem.soh_min - soh, 0)
    out = {"I1_mean_pp": float(r1.mean()), "I1_max_pp": float(r1.max()),
           "I2_mean_pp_per_cyc": float(r2.mean()),
           "I2_max_pp_per_cyc": float(r2.max()),
           "I3_mean_pp": float(r3.mean()), "I3_max_pp": float(r3.max())}
    if ah_cycle is not None:
        ah = np.asarray(ah_cycle, dtype=float)
        cap = chem.Q_nom * soh / 100.0 * (1 + chem.energy_tol)
        r4 = np.maximum(ah - cap, 0)
        r4 = np.where(np.isfinite(r4), r4, 0.0)
        out["I4_mean_Ah"] = float(r4.mean()); out["I4_max_Ah"] = float(r4.max())
    return out


def compute_pvr_breakdown(soh, chem, temperature_C=None, ah_cycle=None,
                          cell_ids=None, cycles=None) -> dict:
    out = {}
    for name in ("I1", "I2", "I3", "I4"):
        try:
            out[name] = compute_pvr(soh, chem, temperature_C, ah_cycle,
                                    cell_ids, cycles, which=(name,))
        except MissingConstraintInput:
            out[name] = float("nan")
    out["all"] = compute_pvr(soh, chem, temperature_C, ah_cycle,
                             cell_ids, cycles)
    return out


class PhysicsLoss(nn.Module):
    """Soft auxiliary physics loss used DURING gradient steps (the hard
    filter operates at the data level; this term only stabilises between
    refinement iterations, cf. lambda_phys sensitivity)."""

    def __init__(self, cfg: HIPIFConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, soh_pred, prev_soh=None, T_K=None,
                which: Tuple[str, ...] = ("I1", "I2", "I3", "I4")):
        chem = self.cfg.chem
        loss = soh_pred.new_zeros(())
        if "I3" in which:
            loss = loss + (torch.relu(soh_pred - chem.soh_max).mean()
                           + torch.relu(chem.soh_min - soh_pred).mean())
        if "I1" in which and prev_soh is not None:
            loss = loss + torch.relu(soh_pred - prev_soh - chem.mono_eps).mean()
        if "I2" in which and prev_soh is not None and T_K is not None:
            k_max = chem.k0 * torch.exp(
                -chem.E_a / R_GAS
                * (1.0 / torch.clamp(T_K, 220.0, 350.0) - 1.0 / chem.T_ref_K))
            d = torch.abs(soh_pred - prev_soh)
            loss = loss + torch.relu(d - k_max).mean()
        return loss
