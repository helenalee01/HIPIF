import numpy as np, pytest
from hipif.config import ChemistryParams
from hipif.constraints import (PhysicsConstraints, monotonicity_indicator,
                               arrhenius_indicator, energy_indicator,
                               compute_pvr, MissingConstraintInput)
from hipif.adaptation.projector import project_feasible

CHEM = ChemistryParams.from_registry("Li-ion")


def test_i1_respects_cell_boundaries():
    # jump up ACROSS cells must not be flagged; within-cell jump must be
    soh = np.array([80.0, 79.0, 95.0, 96.5])
    cells = np.array(["a", "a", "b", "b"])
    cyc = np.array([1, 2, 1, 2])
    ok = monotonicity_indicator(soh, CHEM, cells, cyc)
    assert ok[2]          # first sample of cell b: cross-cell jump ignored
    assert not ok[3]      # +1.5pp within cell b > mono_eps


def test_i2_uses_delta_cycle():
    # 2pp drop over 10 cycles = 0.2 pp/cyc (ok); over 1 cycle (violation)
    chem = ChemistryParams.from_registry("Li-ion")
    chem.k0 = 0.3
    soh = np.array([100.0, 98.0]); T = np.array([25.0, 25.0])
    ok_far = arrhenius_indicator(soh, T, chem, cycles=np.array([0, 10]))
    ok_near = arrhenius_indicator(soh, T, chem, cycles=np.array([0, 1]))
    assert ok_far[1] and not ok_near[1]


def test_i2_requires_temperature():
    with pytest.raises(MissingConstraintInput):
        arrhenius_indicator(np.array([100.0, 99.0]), None, CHEM)


def test_i4_hard_fails_without_ah():
    with pytest.raises(MissingConstraintInput):
        energy_indicator(np.array([90.0]), None, CHEM)


def test_i4_flags_impossible_capacity():
    # 1.9 Ah delivered but SoH claims 80% of 2.0 Ah = 1.6 Ah usable
    ok = energy_indicator(np.array([80.0, 97.0]),
                          np.array([1.9, 1.9]), CHEM)
    assert not ok[0] and ok[1]


def test_persistence_filter_per_cell():
    chem = ChemistryParams.from_registry("Li-ion")
    pc = PhysicsConstraints(chem)
    # cell a: 2-sample violation run (< window 3) forgiven;
    # cell b: 3-sample run charged
    soh = np.array([100, 101, 101.8, 99,   100, 101, 102, 103.5])
    cells = np.array(["a"] * 4 + ["b"] * 4)
    cyc = np.tile(np.arange(4), 2)
    mask = pc.feasibility_mask(soh, temperature_C=np.full(8, 25.0),
                               ah_cycle=np.zeros(8), cell_ids=cells,
                               cycles=cyc, which=("I1",))
    assert mask[:4].all()             # short run forgiven in cell a
    assert not mask[5:8].all()        # persistent run rejected in cell b


def test_projector_feasible_output():
    rng = np.random.default_rng(0)
    soh = 100 - np.cumsum(rng.uniform(0, 0.4, 50)) + rng.normal(0, 1.5, 50)
    cells = np.array(["x"] * 50); cyc = np.arange(50, dtype=float)
    T = np.full(50, 25.0)
    z = project_feasible(soh, cells, cyc, T, CHEM)
    assert (np.diff(z) <= 1e-9).all()                 # I1 monotone
    assert (z >= CHEM.soh_min - 1e-9).all() and (z <= CHEM.soh_max + 1e-9).all()
    k = CHEM.k_max(25.0)
    assert (np.abs(np.diff(z)) <= k + 1e-6).all()      # I2 rate cap
    pvr = compute_pvr(z, CHEM, T, np.zeros(50), cells, cyc)
    assert pvr == 0.0


def test_raw_vs_projected_pvr_separation():
    soh = np.array([100.0, 104.0, 99.0, 103.0, 98.0, 97.5])
    cells = np.array(["c"] * 6); cyc = np.arange(6, dtype=float)
    T = np.full(6, 25.0)
    raw = compute_pvr(soh, CHEM, T, np.zeros(6), cells, cyc,
                      persistence=False)
    z = project_feasible(soh, cells, cyc, T, CHEM)
    proj = compute_pvr(z, CHEM, T, np.zeros(6), cells, cyc)
    assert raw > 0.0 and proj == 0.0
