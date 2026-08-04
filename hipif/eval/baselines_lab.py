"""Supervised lab baselines with structurally valid architectures (E0 / WP4).

v1 defects fixed: LSTM/GRU now consume real sliding windows (seq_len=8),
the Transformer consumes multi-token sequences, and the fake UKF (identical
to EKF) is REMOVED pending a genuine sigma-point implementation — its Table
row is retired rather than mislabeled. No calibration ensembles on top of
any model; splits are cell-level (group LOCO), never row-random.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

SEQ_LEN = 8


def build_windows(df, feat_cols, seq_len: int = SEQ_LEN):
    """Per-cell sliding windows over cycle-ordered features.
    Returns X [n, seq_len, d], y [n], cell_ids [n]."""
    Xs, ys, cs = [], [], []
    for cid in df["cell_id"].unique():
        s = df[df["cell_id"] == cid].sort_values("cycle")
        F = s[list(feat_cols)].to_numpy(np.float32)
        Y = s["SoH"].to_numpy(np.float32)
        for i in range(len(s)):
            lo = max(0, i - seq_len + 1)
            w = F[lo:i + 1]
            if len(w) < seq_len:                       # left-pad by repeat
                w = np.vstack([np.repeat(w[:1], seq_len - len(w), 0), w])
            Xs.append(w); ys.append(Y[i]); cs.append(cid)
    return (np.stack(Xs), np.asarray(ys, np.float32), np.asarray(cs))


class _SeqBase:
    def __init__(self, d_in, device, seed, lr=1e-3, epochs=150, batch=64):
        torch.manual_seed(seed); np.random.seed(seed)
        self.device = device; self.lr = lr; self.epochs = epochs
        self.batch = batch; self.stats = None

    def _norm_fit(self, X, y):
        self.stats = (X.mean((0, 1)), X.std((0, 1)) + 1e-8,
                      float(y.mean()), float(y.std() + 1e-8))

    def _norm(self, X):
        mu, sd, _, _ = self.stats
        return (X - mu) / sd

    def fit(self, X, y):
        self._norm_fit(X, y)
        mu, sd, ym, ys = self.stats
        Xt = torch.FloatTensor(self._norm(X)).to(self.device)
        Yt = torch.FloatTensor((y - ym) / ys).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        mse = nn.MSELoss()
        n = len(Xt); g = torch.Generator(); g.manual_seed(0)
        for _ in range(self.epochs):
            idx = torch.randperm(n, generator=g)
            for i in range(0, n, self.batch):
                b = idx[i:i + self.batch]
                opt.zero_grad()
                loss = mse(self.net(Xt[b]), Yt[b])
                loss.backward(); opt.step()
        return self

    @torch.no_grad()
    def predict(self, X):
        _, _, ym, ys = self.stats
        Xt = torch.FloatTensor(self._norm(X)).to(self.device)
        return self.net(Xt).cpu().numpy() * ys + ym


class LSTMReg(_SeqBase):
    """Zhang et al. 2020-style LSTM: 1 layer, 64 hidden, last state -> linear."""
    def __init__(self, d_in, device, seed, **kw):
        super().__init__(d_in, device, seed, **kw)
        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.rnn = nn.LSTM(d_in, 64, batch_first=True)
                s.head = nn.Linear(64, 1)
            def forward(s, x):
                o, _ = s.rnn(x)
                return s.head(o[:, -1]).squeeze(-1)
        self.net = Net().to(device)


class GRUReg(_SeqBase):
    """Ma et al. 2021-style GRU."""
    def __init__(self, d_in, device, seed, **kw):
        super().__init__(d_in, device, seed, **kw)
        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.rnn = nn.GRU(d_in, 64, batch_first=True)
                s.head = nn.Linear(64, 1)
            def forward(s, x):
                o, _ = s.rnn(x)
                return s.head(o[:, -1]).squeeze(-1)
        self.net = Net().to(device)


class TransformerReg(_SeqBase):
    """Liu et al. 2023-style encoder: multi-token, 2 layers, 4 heads,
    learned positional embedding, mean pooling."""
    def __init__(self, d_in, device, seed, d_model=64, **kw):
        super().__init__(d_in, device, seed, **kw)
        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.embed = nn.Linear(d_in, d_model)
                s.pos = nn.Parameter(torch.zeros(1, SEQ_LEN, d_model))
                layer = nn.TransformerEncoderLayer(
                    d_model, nhead=4, dim_feedforward=128,
                    batch_first=True, dropout=0.1)
                s.enc = nn.TransformerEncoder(layer, num_layers=2)
                s.head = nn.Linear(d_model, 1)
            def forward(s, x):
                h = s.enc(s.embed(x) + s.pos[:, :x.shape[1]])
                return s.head(h.mean(1)).squeeze(-1)
        self.net = Net().to(device)


def coulomb_counting(df) -> np.ndarray:
    """Classical baseline (Plett 2015): SoH proxy = per-cycle coulomb-counted
    discharge Ah / nameplate — the physically-defined estimator, not a
    voltage regression as in v1."""
    return np.asarray(df["Ah_cycle"], float)  # caller scales by Q_nom
