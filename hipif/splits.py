"""Split generators: source GroupCV, target LOCO, causal rolling-origin (WP1).

The held-out target cell is excluded from adaptation, prior estimation,
early stopping, and hyperparameter selection (P4). `assert_disjoint` is the
runtime guard; tests/test_splits.py verifies it.
"""
from __future__ import annotations
from typing import Iterator, List, Sequence, Tuple
import numpy as np


class SplitLeakageError(RuntimeError):
    pass


def assert_disjoint(adapt_cells: Sequence[str], test_cells: Sequence[str]) -> None:
    inter = set(map(str, adapt_cells)) & set(map(str, test_cells))
    if inter:
        raise SplitLeakageError(
            f"adaptation/test cell overlap {sorted(inter)} (P4 violation)")


def target_loco(cell_ids: Sequence[str]) -> Iterator[Tuple[List[str], str]]:
    """Leave-one-cell-out over target cells.
    Yields (adapt_cells, held_out_cell); adapt_cells are UNLABELED-use only."""
    cells = sorted(set(map(str, cell_ids)))
    if len(cells) < 2:
        raise SplitLeakageError(
            f"LOCO needs >=2 target cells, got {cells} (fail-fast, P7)")
    for held in cells:
        adapt = [c for c in cells if c != held]
        assert_disjoint(adapt, [held])
        yield adapt, held


def source_group_folds(cell_ids: Sequence[str], n_folds: int | None = None,
                       seed: int = 42) -> Iterator[Tuple[List[str], List[str]]]:
    """Group (cell-level) CV over source cells. n_folds=None -> LOCO."""
    cells = sorted(set(map(str, cell_ids)))
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(cells))
    k = len(cells) if n_folds is None else min(n_folds, len(cells))
    folds = [order[i::k] for i in range(k)]
    for f in folds:
        train = [c for c in cells if c not in f]
        assert_disjoint(train, f)
        yield train, list(f)


def rolling_origin(cycles: np.ndarray, n_origins: int = 5,
                   min_prefix_frac: float = 0.3, horizon_frac: float = 0.1):
    """Causal same-cell protocol (Track B): prefix -> future horizon.
    Yields (prefix_mask, horizon_mask) boolean arrays over samples of ONE cell,
    ordered by cycle. Never exposes the full-trajectory slope or max cycle."""
    order = np.argsort(cycles)
    n = len(cycles)
    h = max(int(round(horizon_frac * n)), 1)
    starts = np.linspace(int(min_prefix_frac * n), n - h - 1, n_origins).astype(int)
    for s in starts:
        pre = np.zeros(n, bool); hor = np.zeros(n, bool)
        pre[order[:s]] = True
        hor[order[s:s + h]] = True
        yield pre, hor


def mask_cells(cell_ids: Sequence[str], keep: Sequence[str]) -> np.ndarray:
    keep = set(map(str, keep))
    return np.array([str(c) in keep for c in cell_ids], dtype=bool)
