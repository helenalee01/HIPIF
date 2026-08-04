"""Valid label-free transfer baselines (E2 / WP4).

Fairness protocol: identical schema features, identical splits, identical
source pretraining budget and seeds as HIPIF. Method cards in docstrings.
"""
from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn as nn

from ..config import HIPIFConfig
from ..training.trainer import HIPIFTrainer
from ..adaptation.refiner import SourcePack


# ---------------------------------------------------------------- Direct
def direct_transfer(trainer: HIPIFTrainer, X_t: np.ndarray) -> np.ndarray:
    """Source model applied unchanged (E1)."""
    return trainer.predict_soh(X_t)


# ---------------------------------------------------------------- CORAL
def coral_fit(X_s: np.ndarray, X_t_adapt: np.ndarray, eps: float = 1e-3):
    """Linear CORAL (Sun & Saenko 2016). Transform parameters are estimated
    ONLY on adaptation cells (inductive, P4); apply the returned function to
    any target features, including the held-out cell."""
    Xs = np.nan_to_num(X_s.astype(np.float64))
    Xt = np.nan_to_num(X_t_adapt.astype(np.float64))
    mu_s, mu_t = Xs.mean(0), Xt.mean(0)
    Cs = np.cov(Xs, rowvar=False) + eps * np.eye(Xs.shape[1])
    Ct = np.cov(Xt, rowvar=False) + eps * np.eye(Xt.shape[1])
    def _sqrt(C, inv=False):
        w, V = np.linalg.eigh(C)
        w = np.clip(w, 1e-8, None)
        w = 1 / np.sqrt(w) if inv else np.sqrt(w)
        return (V * w) @ V.T
    A = _sqrt(Ct, inv=True) @ _sqrt(Cs)
    def transform(X):
        return ((np.nan_to_num(X.astype(np.float64)) - mu_t) @ A
                + mu_s).astype(np.float32)
    return transform


def coral_align(X_s, X_t):
    """Convenience: fit on X_t and transform X_t (transductive form —
    use coral_fit for the inductive protocol)."""
    return coral_fit(X_s, X_t)(X_t)


# ------------------------------------------------- Confidence self-training
class ConfidenceSelfTrainer:
    """Ensemble-confidence pseudo-labeling (French et al. 2017 style):
    K students pretrained on source with different seeds; iteratively fit on
    target samples whose ensemble std < tau (pp). No physics, no anchor —
    the canonical accuracy-only self-training the paper argues against."""

    def __init__(self, cfg: HIPIFConfig, source: SourcePack,
                 k: int = 3, tau_pp: float = 1.0, n_iter: int = 6,
                 pretrain_epochs: int | None = None):
        self.cfg = cfg; self.tau = tau_pp; self.n_iter = n_iter
        self.members: list[HIPIFTrainer] = []
        ep = pretrain_epochs or cfg.epochs_pretrain
        for i in range(k):
            c = copy.deepcopy(cfg); c.seed = cfg.seed + 101 * i
            m = HIPIFTrainer(c, mode="T4_mlp_only")
            m.fit(source.X, source.y, T_phys=source.T_phys, T_obs=None,
                  epochs=ep, verbose=False)
            self.members.append(m)

    def adapt_predict(self, X_t: np.ndarray) -> np.ndarray:
        for _ in range(self.n_iter):
            preds = np.stack([m.predict_soh(X_t) for m in self.members])
            mu, sd = preds.mean(0), preds.std(0)
            keep = sd < self.tau
            if keep.sum() < 10:
                break
            for m in self.members:
                m.fit(X_t[keep], mu[keep].astype(np.float32),
                      T_phys=np.zeros(int(keep.sum()), np.float32),
                      T_obs=None, epochs=self.cfg.epochs_refine,
                      verbose=False, refit_stats=False)
        return np.stack([m.predict_soh(X_t) for m in self.members]).mean(0)


# ---------------------------------------------------------------- DANN
class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam; return x.view_as(x)
    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class DANNRegressor:
    """Domain-adversarial NN (Ganin et al. 2016): shared encoder, SoH
    regressor on labeled source, gradient-reversal domain head on
    source-vs-target(adaptation cells). No physics."""

    def __init__(self, cfg: HIPIFConfig, lam_adv: float = 0.1,
                 hidden: int = 64):
        self.cfg = cfg; self.lam = lam_adv
        torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
        d = cfg.input_dim
        self.enc = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU()
                                 ).to(cfg.device)
        self.reg = nn.Linear(hidden, 1).to(cfg.device)
        self.dom = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(),
                                 nn.Linear(32, 1)).to(cfg.device)
        self.stats = None

    def fit(self, X_s, y_s, X_t, epochs: int | None = None):
        cfg = self.cfg
        epochs = epochs or cfg.epochs_pretrain
        mu = X_s.mean(0); sd = X_s.std(0) + 1e-8
        ym, ys = float(y_s.mean()), float(y_s.std() + 1e-8)
        self.stats = (mu, sd, ym, ys)
        Xs = torch.FloatTensor((X_s - mu) / sd).to(cfg.device)
        Ys = torch.FloatTensor((y_s - ym) / ys).to(cfg.device)
        Xt = torch.FloatTensor((X_t - mu) / sd).to(cfg.device)
        opt = torch.optim.Adam([*self.enc.parameters(), *self.reg.parameters(),
                                *self.dom.parameters()], lr=cfg.lr,
                               weight_decay=cfg.weight_decay)
        bce = nn.BCEWithLogitsLoss(); mse = nn.MSELoss()
        bs = cfg.batch_size
        g = torch.Generator(); g.manual_seed(cfg.seed)
        for _ in range(epochs):
            i_s = torch.randint(0, len(Xs), (bs,), generator=g)
            i_t = torch.randint(0, len(Xt), (bs,), generator=g)
            hs, ht = self.enc(Xs[i_s]), self.enc(Xt[i_t])
            loss = mse(self.reg(hs).squeeze(-1), Ys[i_s])
            h_all = torch.cat([_GRL.apply(hs, self.lam),
                               _GRL.apply(ht, self.lam)])
            dlab = torch.cat([torch.zeros(bs), torch.ones(bs)]).to(cfg.device)
            loss = loss + bce(self.dom(h_all).squeeze(-1), dlab)
            opt.zero_grad(); loss.backward(); opt.step()
        return self

    @torch.no_grad()
    def predict(self, X):
        mu, sd, ym, ys = self.stats
        Xn = torch.FloatTensor((np.nan_to_num(X) - mu) / sd).to(self.cfg.device)
        return (self.reg(self.enc(Xn)).squeeze(-1).cpu().numpy() * ys + ym)
