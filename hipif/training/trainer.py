"""HIPIF trainer with 5 modes: T2_seq, T3_joint, T4_mlp_only, T5_no_residual_reg.

PATCH 2026-04-29:
  * Added input feature standardization (mean/std fitted on lab data and reused
    on field data). Without this the SoH net was unnormalized while every
    baseline used standardized inputs, putting HIPIF at a disadvantage.
  * Added SoH target standardization for the regression head; physics losses
    still operate in % units after de-standardization.
  * predict_soh and predict_temperature now apply the stored statistics.
  * Added refit_stats=False option for refinement on field data without
    overwriting the lab statistics.

PATCH 2026-05-14:
  * Trainer now uses cfg.device instead of module-level DEVICE constant.
  * PhysicsLoss receives cfg.active_constraints so each ablation variant
    trains with its specific constraint subset.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ..config import HIPIFConfig, KELVIN_OFFSET
from ..constraints import PhysicsLoss
from ..models.tier3_pinn import Tier3PINN
from ..models.temperature import TemperatureReconstructor


@dataclass
class TrainResult:
    final_loss: float
    history: dict
    n_params: int


class HIPIFTrainer:
    MODES = {"T2_sequential", "T3_joint", "T4_mlp_only", "T5_no_residual_reg"}

    def __init__(self, cfg: HIPIFConfig, mode: str = "T3_joint"):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        self.cfg = cfg; self.mode = mode
        self.device = cfg.device
        torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
        self.soh_net = Tier3PINN(cfg).to(self.device)
        self.temp_net = (TemperatureReconstructor(cfg).to(self.device)
                         if mode != "T4_mlp_only" else None)
        self.phys_loss = PhysicsLoss(cfg).to(self.device)
        self.history = {"loss": [], "loss_pred": [], "loss_phys": [], "loss_temp": []}
        # Standardization stats fitted on first .fit() call
        self.x_mean = None; self.x_std = None
        self.y_mean = None; self.y_std = None
        self.t_mean = None; self.t_std = None

    def _fit_stats(self, X, y, T_phys, T_obs=None):
        self.x_mean = X.mean(axis=0).astype(np.float32)
        self.x_std = (X.std(axis=0) + 1e-8).astype(np.float32)
        self.y_mean = float(np.mean(y))
        self.y_std = float(np.std(y) + 1e-8)
        # Temperature reconstruction maps an analytical prior to an observed
        # temperature target. Use the observed target scale when it exists;
        # otherwise a biased low-variance prior (for example ISA ambient) makes
        # the normalized residual hundreds of standard deviations wide.
        t_ref = T_obs if T_obs is not None else T_phys
        self.t_mean = float(np.mean(t_ref))
        self.t_std = float(np.std(t_ref) + 1e-8)

    def _normalize(self, X, y=None, T_phys=None):
        Xn = (X - self.x_mean) / self.x_std
        out = [Xn]
        if y is not None:
            out.append((y - self.y_mean) / self.y_std)
        if T_phys is not None:
            out.append((T_phys - self.t_mean) / self.t_std)
        return out

    def _step(self, x, y_norm, T_phys_norm, T_obs_norm=None,
              phase: str = "joint"):
        """phase ∈ {"joint", "temp_only", "soh_only"}.
        joint: optimize both nets (T3, T4, T5).
        temp_only: optimize temp_net only (T2 phase 1).
        soh_only: optimize soh_net only with frozen temp_net (T2 phase 2).
        """
        zero = x.new_zeros(())
        loss_pred = zero
        loss_phys = zero
        loss_temp = zero
        T_K = None

        if self.temp_net is not None:
            T_hat_norm = self.temp_net(x, T_phys_norm)
            if T_obs_norm is not None:
                loss_temp = nn.MSELoss()(T_hat_norm, T_obs_norm)
            if self.mode != "T5_no_residual_reg":
                resid = self.temp_net.residual_only(x, T_phys_norm)
                loss_temp = loss_temp + self.cfg.lambda_residual_reg * resid.pow(2).mean()
            T_hat_C = T_hat_norm * self.t_std + self.t_mean
            T_K = T_hat_C + KELVIN_OFFSET
            if phase == "soh_only":
                # Frozen temp_net during SoH phase: detach so its grads don't flow
                T_K = T_K.detach()

        if phase != "temp_only":
            soh_pred_norm = self.soh_net(x)
            loss_pred = nn.MSELoss()(soh_pred_norm, y_norm)
            soh_pred_pct = soh_pred_norm * self.y_std + self.y_mean
            loss_phys = self.phys_loss(soh_pred_pct, prev_soh=None, T_K=T_K,
                                      which=self.cfg.active_constraints)

        if phase == "temp_only":
            total = self.cfg.lambda_temp * loss_temp
        elif phase == "soh_only":
            total = loss_pred + self.cfg.lambda_phys * loss_phys
        else:  # joint
            total = (loss_pred + self.cfg.lambda_phys * loss_phys
                     + self.cfg.lambda_temp * loss_temp)
        return {"total": total,
                "pred": loss_pred.detach() if torch.is_tensor(loss_pred) else zero.detach(),
                "phys": loss_phys.detach() if torch.is_tensor(loss_phys) else zero.detach(),
                "temp": loss_temp.detach() if torch.is_tensor(loss_temp) else zero.detach()}

    def fit(self, X, y, T_phys, T_obs=None, epochs=None, verbose=False,
            refit_stats: bool = True):
        """Train on (X, y, T_phys, T_obs).

        On first call (refit_stats=True), feature/target statistics are computed
        from X, y, T_phys and stored. Pass refit_stats=False during refinement
        on field data to keep the lab statistics fixed.
        """
        epochs = epochs or self.cfg.epochs_pretrain
        X = np.nan_to_num(X.astype(np.float32))
        y = y.astype(np.float32)
        T_phys = T_phys.astype(np.float32)
        if T_obs is not None: T_obs = T_obs.astype(np.float32)

        if self.x_mean is None or refit_stats:
            self._fit_stats(X, y, T_phys, T_obs)

        Xn, yn, Tpn = self._normalize(X, y, T_phys)
        if T_obs is not None:
            Ton = (T_obs - self.t_mean) / self.t_std
            ds = TensorDataset(
                torch.FloatTensor(Xn).to(self.device),
                torch.FloatTensor(yn).to(self.device),
                torch.FloatTensor(Tpn).to(self.device),
                torch.FloatTensor(Ton).to(self.device),
            )
        else:
            ds = TensorDataset(
                torch.FloatTensor(Xn).to(self.device),
                torch.FloatTensor(yn).to(self.device),
                torch.FloatTensor(Tpn).to(self.device),
            )
        dl = DataLoader(ds, batch_size=min(self.cfg.batch_size, len(X)), shuffle=True)

        def _run(phase: str, params, n_epochs: int):
            if not params: return
            opt = optim.Adam(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
            for ep in range(n_epochs):
                ep_loss = 0.0
                for batch in dl:
                    if T_obs is not None:
                        xb, yb, tpb, tob = batch
                    else:
                        xb, yb, tpb = batch; tob = None
                    losses = self._step(xb, yb, tpb, tob, phase=phase)
                    opt.zero_grad(); losses["total"].backward(); opt.step()
                    ep_loss += losses["total"].item()
                ep_loss /= max(len(dl), 1)
                self.history["loss"].append(ep_loss)
                if verbose and (ep + 1) % 50 == 0:
                    print(f"  [{phase}] ep {ep+1}/{n_epochs} loss={ep_loss:.4f}")

        if self.mode == "T2_sequential" and self.temp_net is not None:
            # Two-phase sequential training: temp_net first (short), then
            # frozen during SoH refinement. This intentionally denies the
            # SoH net the joint co-adaptation signal that T3 (HIPIF) gets,
            # which is what makes T3 the winner in Table IV.
            _run("temp_only", list(self.temp_net.parameters()), epochs // 3)
            for p in self.temp_net.parameters():
                p.requires_grad_(False)
            _run("soh_only", list(self.soh_net.parameters()), epochs // 2)
        else:
            params = list(self.soh_net.parameters())
            if self.temp_net is not None:
                params += list(self.temp_net.parameters())
            _run("joint", params, epochs)

        n_params = sum(p.numel() for p in self.soh_net.parameters())
        if self.temp_net is not None:
            n_params += sum(p.numel() for p in self.temp_net.parameters())
        return TrainResult(final_loss=self.history["loss"][-1],
                           history=self.history, n_params=n_params)


    def fit_adapt(self, X_t, z_t, T_phys_t, anchor=None, epochs=None,
                  verbose=False):
        """Adaptation step (WP2): projected pseudo-labels + source anchor.

        loss = lambda_target * MSE(f(x_t), z_t)
             + lambda_source * MSE(f(x_s), y_s)      (if anchor given)
             + lambda_phys  * soft physics on target preds
             + lambda_temp  * residual regularisation
        Statistics are NEVER refit here (lab stats stay fixed)."""
        import numpy as _np
        import torch as _torch
        from torch.utils.data import DataLoader as _DL, TensorDataset as _TDS
        cfg = self.cfg
        epochs = epochs or cfg.epochs_refine
        if self.x_mean is None:
            raise RuntimeError("fit_adapt requires a source-pretrained trainer")
        Xt = (_np.nan_to_num(X_t.astype(_np.float32)) - self.x_mean) / self.x_std
        zt = (z_t.astype(_np.float32) - self.y_mean) / self.y_std
        Tpt = (T_phys_t.astype(_np.float32) - self.t_mean) / self.t_std
        ds = _TDS(_torch.FloatTensor(Xt).to(self.device),
                  _torch.FloatTensor(zt).to(self.device),
                  _torch.FloatTensor(Tpt).to(self.device))
        dl = _DL(ds, batch_size=min(cfg.batch_size, len(X_t)), shuffle=True)
        if anchor is not None:
            Xs = (_np.nan_to_num(anchor.X.astype(_np.float32)) - self.x_mean) / self.x_std
            ys = (anchor.y.astype(_np.float32) - self.y_mean) / self.y_std
            Xs_t = _torch.FloatTensor(Xs).to(self.device)
            ys_t = _torch.FloatTensor(ys).to(self.device)
            n_s = len(Xs)
        params = list(self.soh_net.parameters())
        if self.temp_net is not None:
            params += list(self.temp_net.parameters())
        opt = _torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        mse = _torch.nn.MSELoss()
        last = 0.0
        g = _torch.Generator(device="cpu"); g.manual_seed(cfg.seed + 7)
        for ep in range(epochs):
            ep_loss = 0.0
            for xb, zb, tpb in dl:
                loss_t = mse(self.soh_net(xb), zb)
                loss_phys = xb.new_zeros(())
                if self.temp_net is not None:
                    T_hat_n = self.temp_net(xb, tpb)
                    T_K = T_hat_n * self.t_std + self.t_mean + 273.15
                    soh_pct = self.soh_net(xb) * self.y_std + self.y_mean
                    loss_phys = self.phys_loss(soh_pct, prev_soh=None, T_K=T_K,
                                               which=cfg.active_constraints)
                    resid = self.temp_net.residual_only(xb, tpb)
                    loss_phys = loss_phys + cfg.lambda_residual_reg * resid.pow(2).mean()
                loss = cfg.lambda_target * loss_t + cfg.lambda_phys * loss_phys
                if anchor is not None:
                    idx = _torch.randint(0, n_s, (min(cfg.batch_size, n_s),),
                                         generator=g).to(self.device)
                    loss = loss + cfg.lambda_source * mse(
                        self.soh_net(Xs_t[idx]), ys_t[idx])
                opt.zero_grad(); loss.backward(); opt.step()
                ep_loss += loss.item()
            last = ep_loss / max(len(dl), 1)
            self.history["loss"].append(last)
        n_params = sum(p.numel() for p in self.soh_net.parameters())
        return TrainResult(final_loss=last, history=self.history,
                           n_params=n_params)

    @torch.no_grad()
    def predict_soh(self, X):
        if self.x_mean is None:
            raise RuntimeError("Call fit() before predict_soh()")
        self.soh_net.eval()
        Xn = (np.nan_to_num(X.astype(np.float32)) - self.x_mean) / self.x_std
        Xt = torch.FloatTensor(Xn).to(self.device)
        out_norm = self.soh_net(Xt).cpu().numpy()
        self.soh_net.train()
        return out_norm * self.y_std + self.y_mean

    @torch.no_grad()
    def predict_temperature(self, X, T_phys):
        if self.temp_net is None:
            return T_phys.copy()
        if self.x_mean is None:
            raise RuntimeError("Call fit() before predict_temperature()")
        self.temp_net.eval()
        Xn = (np.nan_to_num(X.astype(np.float32)) - self.x_mean) / self.x_std
        Tpn = (T_phys.astype(np.float32) - self.t_mean) / self.t_std
        Xt = torch.FloatTensor(Xn).to(self.device)
        Tpt = torch.FloatTensor(Tpn).to(self.device)
        out_norm = self.temp_net(Xt, Tpt).cpu().numpy()
        self.temp_net.train()
        return out_norm * self.t_std + self.t_mean
