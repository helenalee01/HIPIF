#!/usr/bin/env python3
"""run_all.py — single entry point for every primary table/figure (P8 / 7.x).

Stages (plan 4.3):
  e0  source lab Group-LOCO (eVTOL, supervised baselines sanity/upper bound)
  e1  direct transfer (no adaptation)                       [Track A]
  e2  valid UDA baselines: CORAL, confidence self-training, DANN
  e3  HIPIF full (gate + projection + anchor, H1 calibration)  <- main result
  e4  contribution split: no_gate / no_anchor / no_projection / direct_proj
  e5  constraint ablation: drop each of I1..I4 from the gate
  e6  calibration scope: H0 zero-shot / H1 two-param / H2 thermal-extended
  e7  uncertainty: 90% ensemble intervals -> PICP / MPIW / WIS

Usage (GPU laptop):
  python scripts/run_all.py --data-dir ./Data --stages core --seeds 8
  python scripts/run_all.py --data-dir ./Data --stages e3 --quick   # smoke

Every run writes results/<run_id>/{*.csv, manifest.json} and refreshed
paper/generated/macros.tex. No number in the paper may bypass this pipeline.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, platform, subprocess, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hipif.config import HIPIFConfig
from hipif.features import schema
from hipif.splits import target_loco, source_group_folds, assert_disjoint, mask_cells
from hipif.constraints import compute_pvr, compute_pvr_breakdown, constraint_residuals
from hipif.adaptation.projector import project_feasible
from hipif.adaptation.refiner import UnifiedRefiner, TargetPack, SourcePack
from hipif.training.trainer import HIPIFTrainer
from hipif.priors import estimate_k0_labelfree
from hipif.eval.metrics import compute_mae, compute_rmse, compute_r2
from hipif.eval import statistics as st
from hipif.eval.baselines_transfer import (
    direct_transfer, coral_fit, ConfidenceSelfTrainer, DANNRegressor)
from hipif.eval.baselines_lab import (
    build_windows, LSTMReg, GRUReg, TransformerReg, SEQ_LEN)
from hipif.datasets import load_evtol_lab, aggregate_cycles, load_nasa_pcoe, load_calce_cs2

DEFAULT_SEEDS = [42, 123, 2024, 7, 99, 256, 1234, 2718]
TARGETS = {"nasa": "Li-ion", "calce": "LCO"}


# --------------------------------------------------------------- utilities
def sha256_head(path: Path, mb: int = 8) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(mb * 1024 * 1024))
    return h.hexdigest()[:16]


def git_commit() -> str:
    try:
        c = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                           text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True,
                               timeout=5).stdout.strip()
        return c + ("+dirty" if dirty else "")
    except Exception:
        return "no-git"


def t_phys_prior(df: pd.DataFrame, chem) -> np.ndarray:
    """Lumped steady-state thermal prior from OBSERVABLE telemetry:
    T_phys = T_amb + I_rms^2 * R_int / hA  (registry parameters, provenance
    in manifest). Never uses measured cell temperature."""
    i_rms = df["I_rms"].to_numpy(float)
    return (chem.ambient_C + (i_rms ** 2) * chem.R_int / chem.hA
            ).astype(np.float32)


def metrics_row(y, yhat) -> dict:
    return {"mae": compute_mae(y, yhat), "rmse": compute_rmse(y, yhat),
            "r2": compute_r2(y, yhat)}


class Ctx:
    """Loaded data + per-seed source-model cache."""

    def __init__(self, args):
        self.args = args
        base = HIPIFConfig()
        self.data_dir = Path(args.data_dir)
        self.device = (torch.device(args.device) if args.device
                       else base.device)
        print(f"[setup] device={self.device}")
        cfg0 = self.cfg("LFP", DEFAULT_SEEDS[0])
        agg_cache = self.data_dir / "eVTOL" / "_agg_cache_v2.csv"
        if agg_cache.exists():
            self.src = pd.read_csv(agg_cache)
        else:
            self.src = aggregate_cycles(load_evtol_lab(cfg0, verbose=False))
            self.src.to_csv(agg_cache, index=False)
        print(f"[data] eVTOL source: {len(self.src)} cycles, "
              f"{self.src.cell_id.nunique()} cells")
        self.tgt = {}
        wanted = [d.strip() for d in args.datasets.split(",") if d.strip()]
        if not args.skip_targets:
            if "nasa" in wanted:
                self.tgt["nasa"] = load_nasa_pcoe(cfg0)
            if "calce" in wanted:
                self.tgt["calce"] = load_calce_cs2(cfg0)
        self._source_cache: dict = {}
        # feature audits for the manifest
        self.audits = {"source": schema.audit(self.src, "source_lab")}
        for k, v in self.tgt.items():
            self.audits[k] = schema.audit(v, "target")

    def cfg(self, chemistry: str, seed: int, **kw) -> HIPIFConfig:
        c = HIPIFConfig(seed=seed, chemistry=chemistry,
                        data_dir=Path(self.args.data_dir), **kw)
        c.device = getattr(self, "device", c.device)
        if self.args.quick:
            c.epochs_pretrain = 60
            c.epochs_refine = 10
            c.n_refine_iters = 5
        return c

    def source_pack(self) -> SourcePack:
        chem = self.cfg("LFP", 0).chem
        return SourcePack(
            X=schema.build_model_matrix(self.src, role="source_lab"),
            y=self.src["SoH"].to_numpy(np.float32),
            T_phys=t_phys_prior(self.src, chem))

    def source_trainer(self, seed: int) -> HIPIFTrainer:
        """Pretrained source model, cached per seed (frozen hyperparams, P4).
        Source supervision may use measured T as the temperature-head TARGET
        (T_obs), never as an input feature."""
        if seed in self._source_cache:
            return self._source_cache[seed]
        cfg = self.cfg("LFP", seed)
        tr = HIPIFTrainer(cfg, mode="T3_joint")
        sp = self.source_pack()
        T_obs = self.src["T_mean"].to_numpy(np.float32)   # source lab only
        tr.fit(sp.X, sp.y, T_phys=sp.T_phys, T_obs=T_obs,
               epochs=cfg.epochs_pretrain, verbose=False)
        self._source_cache[seed] = tr
        return tr


# --------------------------------------------------------------- stage E0
def run_e0(ctx: Ctx, seeds) -> pd.DataFrame:
    """Source lab Group-LOCO. Lab role may observe measured temperature, so
    the lab feature set = MODEL_FEATURES + (T_mean, T_max)."""
    lab_feats = list(schema.MODEL_FEATURES) + ["T_mean", "T_max"]
    df = ctx.src
    rows = []
    lab_seeds = seeds[: max(1, min(3, len(seeds)))]   # 3 seeds suffice for E0
    for seed in lab_seeds:
        for train_cells, test_cells in source_group_folds(
                df.cell_id, n_folds=None, seed=42):
            m_tr = mask_cells(df.cell_id, train_cells)
            m_te = mask_cells(df.cell_id, test_cells)
            tr_df, te_df = df[m_tr], df[m_te]
            y_te = te_df["SoH"].to_numpy(np.float32)
            # sequence baselines
            Xw_tr, yw_tr, _ = build_windows(tr_df, lab_feats)
            Xw_te, yw_te, _ = build_windows(te_df, lab_feats)
            for name, cls in (("LSTM", LSTMReg), ("GRU", GRUReg),
                              ("Transformer", TransformerReg)):
                ep = 40 if ctx.args.quick else 150
                mdl = cls(len(lab_feats), ctx.device, seed, epochs=ep)
                mdl.fit(Xw_tr, yw_tr)
                rows.append({"stage": "e0", "dataset": "evtol",
                             "method": name, "cell": test_cells[0],
                             "seed": seed,
                             **metrics_row(yw_te, mdl.predict(Xw_te))})
            # HIPIF supervised (flat lab features + physics loss)
            cfg = ctx.cfg("LFP", seed, input_dim=len(lab_feats))
            t = HIPIFTrainer(cfg, mode="T3_joint")
            Xf_tr = tr_df[lab_feats].to_numpy(np.float32)
            Xf_te = te_df[lab_feats].to_numpy(np.float32)
            chem = cfg.chem
            t.fit(Xf_tr, tr_df["SoH"].to_numpy(np.float32),
                  T_phys=t_phys_prior(tr_df, chem),
                  T_obs=tr_df["T_mean"].to_numpy(np.float32),
                  epochs=cfg.epochs_pretrain)
            # I3 bounds are part of HIPIF's deployed output stage, so the
            # lab variant is evaluated with the same [soh_min, soh_max] clip
            # (raw MLP extrapolation on an out-of-range cell is unbounded).
            p_lab = np.clip(t.predict_soh(Xf_te), chem.soh_min, chem.soh_max)
            rows.append({"stage": "e0", "dataset": "evtol",
                         "method": "HIPIF_lab", "cell": test_cells[0],
                         "seed": seed, **metrics_row(y_te, p_lab)})
            # coulomb counting reference
            cc = te_df["Ah_cycle"].to_numpy(float) / chem.Q_nom * 100.0
            rows.append({"stage": "e0", "dataset": "evtol",
                         "method": "CoulombCount", "cell": test_cells[0],
                         "seed": seed, **metrics_row(y_te, np.clip(cc, 50, 100))})
    return pd.DataFrame(rows)


# ------------------------------------------------- target fold machinery
def fold_packs(ctx: Ctx, ds: str, adapt_cells, held: str, chem):
    df = ctx.tgt[ds]
    m_ad = mask_cells(df.cell_id, adapt_cells)
    m_te = mask_cells(df.cell_id, [held])
    assert_disjoint(df.cell_id[m_ad].unique(), df.cell_id[m_te].unique())
    def pack(mask) -> TargetPack:
        sub = df[mask]
        return TargetPack(
            X=schema.build_model_matrix(sub, role="target"),
            T_phys=t_phys_prior(sub, chem),
            cell_ids=sub["cell_id"].to_numpy(),
            cycles=sub["cycle"].to_numpy(float),
            ah_cycle=schema.constraint_inputs(sub, "target")["Ah_cycle"])
    y_te = schema.posthoc_frame(df[m_te])["SoH"].to_numpy(np.float32)
    return pack(m_ad), pack(m_te), y_te


def h1_chem_cfg(ctx: Ctx, ds: str, seed: int, adapt_pack: TargetPack):
    """H1 two-parameter calibration: registry Ea/Qnom for the target
    chemistry + label-free k0 envelope from adaptation cells (P3/P4)."""
    tr = ctx.source_trainer(seed)
    pred = tr.predict_soh(adapt_pack.X)
    reg_k0 = ctx.cfg(TARGETS[ds], seed).chem.k0    # literature envelope floor
    est = estimate_k0_labelfree(pred, adapt_pack.cycles,
                                adapt_pack.cell_ids, lo=reg_k0)
    cfg = ctx.cfg(TARGETS[ds], seed, k0_override=est["k0_pp_per_cycle"])
    return cfg, est


def eval_pred(name, ds, held, seed, y, raw, chem, pack: TargetPack,
              proj=None, extra=None) -> dict:
    raw_pvr = compute_pvr(raw, chem, None, pack.ah_cycle, pack.cell_ids,
                          pack.cycles, which=("I1", "I3", "I4"))
    row = {"stage": "", "dataset": ds, "method": name, "cell": held,
           "seed": seed, **metrics_row(y, proj if proj is not None else raw),
           "mae_raw": compute_mae(y, raw), "raw_pvr": raw_pvr}
    if proj is not None:
        row["proj_pvr"] = compute_pvr(proj, chem, None, pack.ah_cycle,
                                      pack.cell_ids, pack.cycles,
                                      which=("I1", "I3", "I4"))
    if extra:
        row.update(extra)
    return row


def run_transfer_stages(ctx: Ctx, seeds, stages):
    rows = []
    pred_rows = []           # per-sample dump (manifest 7.3)
    pred_store = []          # for e7: (ds, cell, seed, cycles, y, proj)

    def dump(stage, ds, method, held, seed, pack, y, raw, proj=None):
        for i in range(len(y)):
            pred_rows.append({
                "stage": stage, "dataset": ds, "method": method,
                "cell": held, "seed": seed,
                "cycle": float(pack.cycles[i]), "y_true": float(y[i]),
                "pred_raw": float(raw[i]),
                "pred_proj": (float(proj[i]) if proj is not None
                              else np.nan)})
    sp = ctx.source_pack()
    for ds in ctx.tgt:
        cells = sorted(ctx.tgt[ds].cell_id.unique())
        for seed in seeds:
            tr_src = ctx.source_trainer(seed)
            for adapt_cells, held in target_loco(cells):
                ad, te, y = fold_packs(ctx, ds, adapt_cells, held,
                                       ctx.cfg("LFP", seed).chem)
                cfg_h1, k0_est = h1_chem_cfg(ctx, ds, seed, ad)
                chem = cfg_h1.chem
                # recompute T_phys with target thermal registry (H1 keeps
                # source thermal per 5.3 -> use source chem for prior;
                # thermal-extended handled in e6)
                if "e1" in stages:
                    raw = direct_transfer(tr_src, te.X)
                    r = eval_pred("direct", ds, held, seed, y, raw, chem, te)
                    r["stage"] = "e1"; rows.append(r)
                    dump("e1", ds, "direct", held, seed, te, y, raw)
                if "e2" in stages:
                    tf = coral_fit(sp.X, ad.X)        # fit on adapt cells only
                    raw = tr_src.predict_soh(tf(te.X))
                    r = eval_pred("coral", ds, held, seed, y, raw, chem, te)
                    r["stage"] = "e2"; rows.append(r)
                    stt = ConfidenceSelfTrainer(
                        ctx.cfg("LFP", seed), sp, k=3,
                        pretrain_epochs=(60 if ctx.args.quick else 150),
                        n_iter=(2 if ctx.args.quick else 6))
                    stt.adapt_predict(ad.X)           # adapt on N-1 cells
                    raw = np.stack([m.predict_soh(te.X)
                                    for m in stt.members]).mean(0)
                    r = eval_pred("self_train", ds, held, seed, y, raw,
                                  chem, te)
                    r["stage"] = "e2"; rows.append(r)
                    dann = DANNRegressor(ctx.cfg("LFP", seed)).fit(
                        sp.X, sp.y, ad.X,
                        epochs=(60 if ctx.args.quick else 300))
                    raw = dann.predict(te.X)
                    r = eval_pred("dann", ds, held, seed, y, raw, chem, te)
                    r["stage"] = "e2"; rows.append(r)
                variants = []
                if "e3" in stages:
                    variants.append(("hipif", dict()))
                if "e4" in stages:
                    variants += [("no_gate", dict(use_gate=False)),
                                 ("no_anchor", dict(use_anchor=False)),
                                 ("no_projection", dict(use_projection=False))]
                for name, flags in variants:
                    t2 = copy.deepcopy(tr_src); t2.cfg = cfg_h1
                    ref = UnifiedRefiner(cfg_h1, t2, sp, **flags)
                    ref.refine(ad, verbose=False)
                    raw, proj, _ = ref.predict(te.X, te.T_phys, te.cell_ids,
                                               te.cycles, ah_cycle=te.ah_cycle)
                    acc = (ref.log[-1].accept_rate if ref.log else np.nan)
                    r = eval_pred(name, ds, held, seed, y, raw, chem, te,
                                  proj=proj,
                                  extra={"accept_rate": acc,
                                         "k0_labelfree":
                                             k0_est["k0_pp_per_cycle"],
                                         "n_refine_iters": len(ref.log)})
                    r["stage"] = "e3" if name == "hipif" else "e4"
                    rows.append(r)
                    dump(r["stage"], ds, name, held, seed, te, y, raw, proj)
                    if name == "hipif":
                        pred_store.append((ds, held, seed, te.cycles.copy(),
                                           y.copy(), proj.copy()))
                if "e4" in stages:
                    raw = direct_transfer(tr_src, te.X)
                    T_hat = tr_src.predict_temperature(te.X, te.T_phys)
                    proj = project_feasible(raw, te.cell_ids, te.cycles,
                                            T_hat, chem, te.ah_cycle)
                    r = eval_pred("direct_proj", ds, held, seed, y, raw,
                                  chem, te, proj=proj)
                    r["stage"] = "e4"; rows.append(r)
                if "e5" in stages:
                    for drop in ("I1", "I2", "I3", "I4"):
                        which = tuple(c for c in ("I1", "I2", "I3", "I4")
                                      if c != drop)
                        cfg_ab = copy.deepcopy(cfg_h1)
                        cfg_ab.active_constraints = which
                        t2 = copy.deepcopy(tr_src); t2.cfg = cfg_ab
                        ref = UnifiedRefiner(cfg_ab, t2, sp)
                        ref.refine(ad, verbose=False)
                        raw, proj, T_hat = ref.predict(
                            te.X, te.T_phys, te.cell_ids, te.cycles,
                            ah_cycle=te.ah_cycle)
                        bd = compute_pvr_breakdown(raw, chem, T_hat,
                                                   te.ah_cycle, te.cell_ids,
                                                   te.cycles)
                        r = eval_pred(f"drop_{drop}", ds, held, seed, y, raw,
                                      chem, te, proj=proj,
                                      extra={f"pvr_{k}": v
                                             for k, v in bd.items()})
                        r["stage"] = "e5"; rows.append(r)
                if "e6" in stages:
                    for scope in ("H0", "H1", "H2"):
                        if scope == "H0":
                            cfg_s = ctx.cfg("LFP", seed)     # source params
                            te_s, ad_s = te, ad
                        elif scope == "H1":
                            cfg_s = cfg_h1                    # Ea/Qnom + k0
                            te_s, ad_s = te, ad
                        else:                                  # H2: + thermal
                            cfg_s = copy.deepcopy(cfg_h1)
                            chem_t = cfg_s.chem               # registry thermal
                            df_ds = ctx.tgt[ds]
                            ad_s = TargetPack(ad.X, t_phys_prior(
                                df_ds[mask_cells(df_ds.cell_id,
                                                 adapt_cells)], chem_t),
                                ad.cell_ids, ad.cycles, ad.ah_cycle)
                            te_s = TargetPack(te.X, t_phys_prior(
                                df_ds[mask_cells(df_ds.cell_id, [held])],
                                chem_t), te.cell_ids, te.cycles, te.ah_cycle)
                        t2 = copy.deepcopy(tr_src); t2.cfg = cfg_s
                        ref = UnifiedRefiner(cfg_s, t2, sp)
                        ref.refine(ad_s, verbose=False)
                        raw, proj, _ = ref.predict(te_s.X, te_s.T_phys,
                                                   te_s.cell_ids, te_s.cycles,
                                                   ah_cycle=te_s.ah_cycle)
                        r = eval_pred(f"calib_{scope}", ds, held, seed, y,
                                      raw, cfg_s.chem, te_s, proj=proj)
                        r["stage"] = "e6"; rows.append(r)
    return pd.DataFrame(rows), pred_store, pd.DataFrame(pred_rows)


def run_e8(ctx: Ctx, seeds) -> pd.DataFrame:
    """Backbone robustness (E8): is projection-dominance an artefact of the
    MLP backbone? Train GRU/Transformer on SOURCE using target-deployable
    causal windows (MODEL_FEATURES only — no measured T), direct-transfer to
    the held-out target cell, then apply the SAME projector Pi_F.
    T_hat for the I2 rate cap uses the analytic thermal prior (observable)."""
    from hipif.eval.baselines_lab import build_windows, GRUReg, TransformerReg
    feats = list(schema.MODEL_FEATURES)
    rows = []
    src = ctx.src
    Xw_s, yw_s, _ = build_windows(src, feats)
    for ds in ctx.tgt:
        df = ctx.tgt[ds]
        cells = sorted(df.cell_id.unique())
        for seed in seeds:
            for name, cls in (("GRU", GRUReg), ("Transformer", TransformerReg)):
                ep = 40 if ctx.args.quick else 150
                mdl = cls(len(feats), ctx.device, seed, epochs=ep)
                mdl.fit(Xw_s, yw_s)
                for adapt_cells, held in target_loco(cells):
                    chem = h1_chem_cfg(ctx, ds, seed,
                                       fold_packs(ctx, ds, adapt_cells, held,
                                                  ctx.cfg("LFP", seed).chem)[0]
                                       )[0].chem
                    sub = df[mask_cells(df.cell_id, [held])]
                    schema.assert_no_forbidden(feats, role="target")
                    Xw_t, _, _ = build_windows(sub, feats)
                    y = schema.posthoc_frame(sub)["SoH"].to_numpy(np.float32)
                    raw = mdl.predict(Xw_t)
                    T_hat = t_phys_prior(sub, chem)     # analytic, observable
                    proj = project_feasible(
                        raw, sub["cell_id"].to_numpy(),
                        sub["cycle"].to_numpy(float), T_hat, chem,
                        schema.constraint_inputs(sub, "target")["Ah_cycle"])
                    for tag, p in (("direct", raw), ("direct_proj", proj)):
                        rows.append({
                            "stage": "e8", "dataset": ds,
                            "method": f"{name}_{tag}", "cell": held,
                            "seed": seed, **metrics_row(y, p),
                            "raw_pvr": compute_pvr(
                                raw, chem, None,
                                schema.constraint_inputs(sub, "target")["Ah_cycle"],
                                sub["cell_id"].to_numpy(),
                                sub["cycle"].to_numpy(float),
                                which=("I1", "I3", "I4"))})
    return pd.DataFrame(rows)


def run_e7(pred_store, out_dir: Path) -> pd.DataFrame:
    """90% ensemble intervals over seeds -> PICP/MPIW/WIS per cell."""
    rows = []
    by_key = {}
    for ds, cell, seed, cyc, y, proj in pred_store:
        by_key.setdefault((ds, cell), []).append((seed, cyc, y, proj))
    for (ds, cell), items in by_key.items():
        if len(items) < 3:
            continue
        ref = items[0][1]
        P = np.stack([p for _, c, _, p in items if len(c) == len(ref)])
        y = items[0][2]
        mu, lo, hi = st.interval_from_ensemble(P, alpha=0.10)
        rows.append({"stage": "e7", "dataset": ds, "cell": cell,
                     "n_members": len(P),
                     "picp90": st.picp(y, lo, hi),
                     "mpiw": st.mpiw(lo, hi),
                     "wis": st.wis(y, lo, hi, 0.10)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------- reporting
def aggregate(percell: pd.DataFrame) -> pd.DataFrame:
    num = percell.select_dtypes(float).columns
    g = (percell.groupby(["stage", "dataset", "method"])[num]
         .agg(["mean", "std"]).round(4))
    g.columns = ["_".join(c) for c in g.columns]
    return g.reset_index()


def stats_tables(percell: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    out = []
    for ds in percell.dataset.unique():
        sub = percell[(percell.dataset == ds)
                      & percell.stage.isin(["e1", "e2", "e3", "e4"])]
        methods = [m for m in sub.method.unique() if m != "hipif"]
        if "hipif" not in sub.method.values:
            continue
        for m in methods:
            res = st.compare_methods(sub, "hipif", m, "mae", n_boot=n_boot)
            res["dataset"] = ds
            out.append(res)
    return pd.DataFrame(out)


def write_macros(agg: pd.DataFrame, statsdf: pd.DataFrame, path: Path):
    lines = ["% AUTO-GENERATED by scripts/run_all.py — do not edit (P8)"]
    def mac(name, val):
        lines.append(f"\\newcommand{{\\{name}}}{{{val}}}")
    for _, r in agg.iterrows():
        key = f"{r['dataset']}{r['method']}".replace("_", "")
        if "mae_mean" in agg.columns and not pd.isna(r.get("mae_mean")):
            mac(f"mae{key}", f"{r['mae_mean']:.2f}")
        if "raw_pvr_mean" in agg.columns and not pd.isna(r.get("raw_pvr_mean")):
            mac(f"rawpvr{key}", f"{r['raw_pvr_mean']:.1f}")
    for _, r in statsdf.iterrows():
        key = f"{r['dataset']}{r['method_b']}".replace("_", "")
        mac(f"dmae{key}",
            f"{r['delta_point']:.2f}\\,[{r['delta_ci_lo']:.2f},"
            f"{r['delta_ci_hi']:.2f}]")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./Data")
    ap.add_argument("--stages", default="core",
                    help="comma list of e0..e7, or 'core'(e1,e2,e3,e4,e7) "
                         "or 'all'")
    ap.add_argument("--seeds", default="8",
                    help="count (<=8) or comma list of seed values")
    ap.add_argument("--quick", action="store_true",
                    help="smoke mode: short epochs, 2 seeds, small bootstrap")
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--skip-targets", action="store_true")
    ap.add_argument("--datasets", default="nasa,calce",
                    help="comma subset of nasa,calce")
    args = ap.parse_args()

    if "," in args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    elif int(args.seeds) <= 16:          # small int = seed COUNT
        seeds = DEFAULT_SEEDS[: int(args.seeds)]
    else:                                 # large int = a literal seed value
        seeds = [int(args.seeds)]
    if args.quick:
        seeds = seeds[:2]
        args.n_boot = min(args.n_boot, 2000)
    stages = {"core": ["e1", "e2", "e3", "e4", "e7"],
              "all": [f"e{i}" for i in range(9)]}.get(
        args.stages, args.stages.split(","))

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    out = Path("results") / run_id
    out.mkdir(parents=True, exist_ok=True)
    ctx = Ctx(args)

    data_files = list(Path(args.data_dir).rglob("*.mat")) \
        + list(Path(args.data_dir).rglob("*_cycles.csv")) \
        + sorted(Path(args.data_dir).rglob("VAH*.csv"))[:3]
    manifest = {
        "run_id": run_id, "git": git_commit(),
        "python": sys.version.split()[0],
        "torch": torch.__version__, "device": str(ctx.device),
        "platform": platform.platform(),
        "stages": stages, "seeds": seeds, "quick": args.quick,
        "config": ctx.cfg("LFP", seeds[0]).as_dict(),
        "feature_audit": ctx.audits,
        "data_checksums": {str(p.name): sha256_head(p) for p in data_files},
        "start": time.strftime("%F %T"),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    frames = []
    if "e0" in stages:
        print("[stage] e0 source lab Group-LOCO ...")
        e0 = run_e0(ctx, seeds); e0.to_csv(out / "e0_lab_loco.csv", index=False)
        frames.append(e0)
    if "e8" in stages and not args.skip_targets:
        print("[stage] e8 backbone robustness ...")
        e8 = run_e8(ctx, seeds)
        e8.to_csv(out / "e8_backbone.csv", index=False)
        frames.append(e8)
    tstages = [s for s in stages if s in
               ("e1", "e2", "e3", "e4", "e5", "e6")]
    pred_store = []
    if tstages and not args.skip_targets:
        print(f"[stage] target LOCO stages {tstages} ...")
        tf, pred_store, preds = run_transfer_stages(ctx, seeds, tstages)
        tf.to_csv(out / "target_loco_percell.csv", index=False)
        if not preds.empty:
            preds.to_csv(out / "predictions_per_sample.csv", index=False)
        frames.append(tf)
    if "e7" in stages and pred_store:
        e7 = run_e7(pred_store, out)
        if e7.empty:
            print("[e7] skipped: needs >=3 seeds of e3 in the same run "
                  "for ensemble intervals")
        else:
            e7.to_csv(out / "e7_uncertainty.csv", index=False)
            print(e7.to_string(index=False))
    percell = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not percell.empty:
        agg = aggregate(percell)
        agg.to_csv(out / "aggregate.csv", index=False)
        sdf = stats_tables(percell, args.n_boot)
        if not sdf.empty:
            sdf.to_csv(out / "statistics.csv", index=False)
            print(sdf[["dataset", "method_b", "delta_point", "delta_ci_lo",
                       "delta_ci_hi", "sign_test_p",
                       "a_better_ci_excludes_0"]].to_string(index=False))
        write_macros(agg, sdf if not sdf.empty else pd.DataFrame(),
                     Path("paper/generated/macros.tex"))
        print(f"[done] results -> {out}")
    manifest["end"] = time.strftime("%F %T")
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
