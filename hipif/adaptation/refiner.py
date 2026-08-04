"""Unified physics-constrained label-free adaptation (WP2 / 5.1).

Single algorithm for every target dataset:

  teacher y_bar = f_teacher(x)                        (EMA of student)
  T_hat        = g_teacher(x, T_phys)                 (latent temperature)
  raw PVR      = PVR(y_bar | I1-I4, T_hat, Ah)        (logged, P6)
  z            = Pi_F(y_bar | T_hat)                  (projector)
  keep         = feasibility_mask(y_bar)              (hard gate on RAW preds)
  student step : lambda_target * L(f(x_keep), z_keep)
                 + lambda_source * L(f(x_s), y_s)     (source anchor)
                 + lambda_phys * soft physics + lambda_temp * temp reg
  teacher     <- EMA(teacher, student)

Rejected samples are excluded from the target regression loss; the source
anchor always applies. No dataset-specific fade lines, blend ratios, or
thresholds (P3). Hyperparameters are frozen from source validation (P4).
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import torch

from ..config import HIPIFConfig
from ..constraints import PhysicsConstraints, compute_pvr
from .projector import project_feasible
from ..training.trainer import HIPIFTrainer


@dataclass
class RefineLog:
    iteration: int
    raw_pvr: float
    accept_rate: float
    drift: float
    total_loss: float


@dataclass
class TargetPack:
    """Everything adaptation may see about the target (unlabeled)."""
    X: np.ndarray               # model features (schema-checked)
    T_phys: np.ndarray          # analytical thermal prior [C]
    cell_ids: np.ndarray
    cycles: np.ndarray
    ah_cycle: Optional[np.ndarray] = None   # coulomb-counted throughput [Ah]


@dataclass
class SourcePack:
    X: np.ndarray
    y: np.ndarray
    T_phys: np.ndarray


class UnifiedRefiner:
    def __init__(self, cfg: HIPIFConfig, trainer: HIPIFTrainer,
                 source: SourcePack,
                 use_gate: bool = True, use_projection: bool = True,
                 use_anchor: bool = True):
        self.cfg = cfg
        self.trainer = trainer
        self.source = source
        self.use_gate = use_gate
        self.use_projection = use_projection
        self.use_anchor = use_anchor
        self.constraints = PhysicsConstraints(cfg.chem)
        self.log: List[RefineLog] = []
        # EMA teacher = frozen copies of the student networks
        self.t_soh = copy.deepcopy(trainer.soh_net).eval()
        self.t_temp = (copy.deepcopy(trainer.temp_net).eval()
                       if trainer.temp_net is not None else None)

    # ---- teacher inference -------------------------------------------------
    @torch.no_grad()
    def _teacher_predict(self, X, T_phys):
        tr = self.cfg
        t = self.trainer
        Xn = (np.nan_to_num(X.astype(np.float32)) - t.x_mean) / t.x_std
        Xt = torch.FloatTensor(Xn).to(t.device)
        soh = self.t_soh(Xt).cpu().numpy() * t.y_std + t.y_mean
        if self.t_temp is not None:
            Tpn = (T_phys.astype(np.float32) - t.t_mean) / t.t_std
            Tpt = torch.FloatTensor(Tpn).to(t.device)
            T_hat = self.t_temp(Xt, Tpt).cpu().numpy() * t.t_std + t.t_mean
        else:
            T_hat = T_phys.copy()
        return soh, T_hat

    def _ema_update(self):
        m = self.cfg.ema_momentum
        with torch.no_grad():
            for pt, ps in zip(self.t_soh.parameters(),
                              self.trainer.soh_net.parameters()):
                pt.mul_(m).add_(ps, alpha=1 - m)
            if self.t_temp is not None:
                for pt, ps in zip(self.t_temp.parameters(),
                                  self.trainer.temp_net.parameters()):
                    pt.mul_(m).add_(ps, alpha=1 - m)

    # ---- main loop ---------------------------------------------------------
    def refine(self, tgt: TargetPack, n_iter: Optional[int] = None,
               verbose: bool = False) -> List[RefineLog]:
        cfg = self.cfg
        n_iter = n_iter or cfg.n_refine_iters
        prev_pseudo = None
        anchor = self.source if self.use_anchor else None
        for it in range(n_iter):
            pseudo, T_hat = self._teacher_predict(tgt.X, tgt.T_phys)
            raw_pvr = compute_pvr(pseudo, cfg.chem, T_hat, tgt.ah_cycle,
                                  tgt.cell_ids, tgt.cycles,
                                  which=cfg.active_constraints)
            z = (project_feasible(pseudo, tgt.cell_ids, tgt.cycles,
                                  T_hat, cfg.chem, tgt.ah_cycle)
                 if self.use_projection else pseudo.copy())
            if self.use_gate:
                keep = self.constraints.feasibility_mask(
                    pseudo, T_hat, tgt.ah_cycle, tgt.cell_ids, tgt.cycles,
                    which=cfg.active_constraints)
            else:
                keep = np.ones(len(pseudo), dtype=bool)
            accept = keep.mean() * 100.0
            drift = (float("nan") if prev_pseudo is None
                     else float(np.mean(np.abs(pseudo - prev_pseudo))))
            prev_pseudo = pseudo.copy()
            if keep.sum() < 10:
                self.log.append(RefineLog(it, raw_pvr, accept, drift,
                                          float("nan")))
                if verbose:
                    print(f"  iter {it}: only {int(keep.sum())} feasible — stop")
                break
            res = self.trainer.fit_adapt(
                tgt.X[keep], z[keep].astype(np.float32),
                T_phys_t=tgt.T_phys[keep], anchor=anchor,
                epochs=cfg.epochs_refine)
            self._ema_update()
            self.log.append(RefineLog(it, raw_pvr, accept, drift,
                                      res.final_loss))
            if verbose:
                print(f"  iter {it}: rawPVR={raw_pvr:.2f}% accept={accept:.1f}%"
                      f" drift={drift:.2e} loss={res.final_loss:.4f}")
            if not np.isnan(drift) and drift < cfg.drift_eps and it >= 3:
                break
        return self.log

    # ---- deployment output -------------------------------------------------
    @torch.no_grad()
    def predict(self, X, T_phys, cell_ids, cycles, project: bool = True,
                ah_cycle=None):
        """Final student predictions. Returns (raw, projected, T_hat)."""
        raw = self.trainer.predict_soh(X)
        T_hat = self.trainer.predict_temperature(X, T_phys)
        proj = (project_feasible(raw, cell_ids, cycles, T_hat, self.cfg.chem,
                                 ah_cycle)
                if project else raw.copy())
        return raw, proj, T_hat

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame([{
            "Iter": r.iteration, "raw_PVR (%)": r.raw_pvr,
            "Accept (%)": r.accept_rate, "Label Drift": r.drift,
            "L_total": r.total_loss} for r in self.log])
