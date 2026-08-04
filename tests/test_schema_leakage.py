import numpy as np, pandas as pd, pytest
from hipif.features import schema


def _frame():
    return pd.DataFrame({
        "cell_id": ["a"] * 5, "cycle": range(1, 6),
        "V_mean": 3.7, "V_min": 3.0, "V_max": 4.1,
        "I_mean": -1.0, "I_std": 0.1, "I_abs_mean": 1.0,
        "cycle_log": np.log1p(np.arange(1, 6)),
        "Ah_cycle": 1.0, "t_cycle_s": 3600.0, "I_rms": 1.0,
        "T_mean": 30.0, "T_max": 33.0, "SoH": 95.0, "cycle_norm": 0.1,
    })


def test_forbidden_blocked_in_target_role():
    with pytest.raises(schema.LeakageError):
        schema.assert_no_forbidden(["V_mean", "T_mean"], role="target")
    with pytest.raises(schema.LeakageError):
        schema.assert_no_forbidden(["cycle_norm"], role="target")


def test_model_matrix_excludes_measured_temperature():
    X = schema.build_model_matrix(_frame(), role="target")
    assert X.shape[1] == len(schema.MODEL_FEATURES)
    assert "T_mean" not in schema.MODEL_FEATURES
    assert "T_max" not in schema.MODEL_FEATURES
    assert "cycle_norm" not in schema.MODEL_FEATURES


def test_missing_causal_feature_fails_fast():
    df = _frame().drop(columns=["cycle_log"])
    with pytest.raises(schema.LeakageError):
        schema.build_model_matrix(df, role="target")


def test_constraint_inputs_fail_fast():
    df = _frame().drop(columns=["Ah_cycle"])
    with pytest.raises(schema.LeakageError):
        schema.constraint_inputs(df, "target")
