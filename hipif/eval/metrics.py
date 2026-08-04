"""Evaluation metrics."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_mae(y, yhat): return float(mean_absolute_error(y, yhat))
def compute_rmse(y, yhat): return float(np.sqrt(mean_squared_error(y, yhat)))
def compute_r2(y, yhat): return float(r2_score(y, yhat))
