"""Feature schema, whitelist, and leakage assertions (WP1 / P1 / P2 / 3.1).

Roles
-----
target        : deployment-condition inputs. FORBIDDEN columns may not enter
                the model matrix, the adaptation objective, or the constraint
                evaluation (except via reconstructed T_hat).
source_lab    : source-domain supervised training. Measured temperature is
                allowed as the *temperature-head training target* (T_obs) but
                NOT as a model input feature, so that the trained network is
                deployable on temperature-free targets.
posthoc_eval  : evaluation namespace only (labels, measured T for verifying
                the temperature reconstruction). Never imported by adaptation.
"""
from __future__ import annotations
from typing import Iterable, Sequence
import numpy as np
import pandas as pd

# NN input features — identical for source training and target inference.
# All causally observable from BMS telemetry at the current cycle.
MODEL_FEATURES: tuple = (
    "V_mean", "V_min", "V_max",     # voltage signature statistics [V]
    "I_mean", "I_std", "I_abs_mean",  # current statistics [A]
    "cycle_log",                    # log1p(cycle index) — causal counter
)

# Inputs used ONLY by physics-constraint evaluation / thermal prior,
# not by the NN. Causally observable (coulomb counting, wall clock, RMS).
CONSTRAINT_INPUTS: tuple = ("Ah_cycle", "t_cycle_s", "I_rms")

# Hard-forbidden anywhere in the target path (A1/A3): measured internal
# temperature, EOL-normalised age, labels and label proxies.
FORBIDDEN_TARGET: tuple = (
    "T_mean", "T_max", "T_min", "Temperature_measured",
    "cycle_norm", "max_cycle", "SoH", "capacity", "QD_mAh",
)

# Post-hoc evaluation-only columns.
POSTHOC_ONLY: tuple = ("SoH", "T_mean", "T_max", "capacity")


class LeakageError(RuntimeError):
    pass


def assert_no_forbidden(cols: Iterable[str], role: str) -> None:
    if role != "target":
        return
    bad = sorted(set(cols) & set(FORBIDDEN_TARGET))
    if bad:
        raise LeakageError(
            f"forbidden column(s) {bad} entered the target path "
            f"(P1 whitelist violation)")


def build_model_matrix(df: pd.DataFrame, role: str) -> np.ndarray:
    """Extract the NN input matrix. Same columns for every role (P2:
    the deployed network never sees measured temperature)."""
    assert_no_forbidden(MODEL_FEATURES, role)
    missing = [c for c in MODEL_FEATURES if c not in df.columns]
    if missing:
        raise LeakageError(
            f"dataset is missing required causal features {missing}; "
            f"loader must provide them (fail-fast, P7)")
    X = df.loc[:, list(MODEL_FEATURES)].to_numpy(dtype=np.float32)
    return np.nan_to_num(X)


def constraint_inputs(df: pd.DataFrame, role: str) -> dict:
    """Ah_cycle / t_cycle_s / I_rms for I2 (dt) and I4. Missing columns
    raise (P7): silent skip of a declared-active constraint is forbidden."""
    out = {}
    for c in CONSTRAINT_INPUTS:
        if c not in df.columns:
            raise LeakageError(
                f"constraint input '{c}' missing from {role} dataframe; "
                f"loaders must compute it from raw telemetry (fail-fast, P7)")
        out[c] = df[c].to_numpy(dtype=np.float64)
    return out


def posthoc_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluation-only view (labels + measured T). Call sites must live in
    evaluation code; adaptation modules must not import this function."""
    cols = [c for c in POSTHOC_ONLY if c in df.columns]
    return df.loc[:, ["cell_id", "cycle"] + cols].copy()


def audit(df: pd.DataFrame, role: str) -> dict:
    """Feature audit record for the run manifest (7.3)."""
    present_forbidden = sorted(set(df.columns) & set(FORBIDDEN_TARGET))
    return {
        "role": role,
        "model_features": list(MODEL_FEATURES),
        "constraint_inputs": list(CONSTRAINT_INPUTS),
        "forbidden_columns_present_in_frame": present_forbidden,
        "note": ("forbidden columns may exist in the loaded frame for "
                 "post-hoc evaluation, but assert_no_forbidden() blocks them "
                 "from the model/constraint path when role='target'"),
        "n_rows": int(len(df)),
        "cells": sorted(map(str, df["cell_id"].unique())),
    }


def add_causal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive causal features every loader must provide."""
    df = df.copy()
    df["cycle_log"] = np.log1p(df["cycle"].astype(float))
    if "I_abs_mean" not in df.columns:
        df["I_abs_mean"] = df["I_mean"].abs()
    return df
