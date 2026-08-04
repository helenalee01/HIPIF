import numpy as np, pytest
from hipif.splits import (target_loco, source_group_folds, assert_disjoint,
                          rolling_origin, SplitLeakageError)


def test_loco_disjoint_and_complete():
    cells = ["c1", "c2", "c3", "c4"]
    held_all = []
    for adapt, held in target_loco(cells):
        assert held not in adapt
        assert set(adapt) | {held} == set(cells)
        held_all.append(held)
    assert sorted(held_all) == sorted(cells)


def test_assert_disjoint_raises():
    with pytest.raises(SplitLeakageError):
        assert_disjoint(["a", "b"], ["b"])


def test_loco_needs_two_cells():
    with pytest.raises(SplitLeakageError):
        list(target_loco(["only"]))


def test_group_folds_cover_all_cells_once():
    cells = [f"c{i}" for i in range(7)]
    seen = []
    for train, test in source_group_folds(cells, n_folds=None):
        assert not set(train) & set(test)
        seen += test
    assert sorted(seen) == sorted(cells)


def test_rolling_origin_causal():
    cyc = np.arange(100)
    for pre, hor in rolling_origin(cyc, n_origins=3):
        assert cyc[pre].max() < cyc[hor].min()   # horizon strictly future
