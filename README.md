# When Coulomb Counting Is Nearly Supervision

Reproduction pipeline for:

> H. Lee, *"When Coulomb Counting Is Nearly Supervision: A Leakage-Aware Audit of
> Label-Free Cross-Dataset Battery State-of-Health Transfer"*,
> submitted to **Energy and AI** (2026).

On public full-discharge ageing benchmarks (NASA PCoE, CALCE CS2), the
observable delivered charge nearly coincides with the capacity-derived SoH
label: plain coulomb counting `100*Q_t/Q_nom` reproduces the CALCE labels
identically under the shared evaluation clipping and attains 0.65 pp MAE on
NASA — outperforming every learned adaptation method evaluated in the paper,
including our own earlier HIPIF pipeline. This repository releases the
leakage-controlled protocol, the coulomb-tautology audit, and every experiment
behind the paper's tables and figures.

**Note.** This repository supersedes the earlier IoTJ-phase code and preprint
("Physics-Constrained Label-Free Adaptation ..."); all quantitative claims of
that version are retired (paper Sec. 2.3, Sec. 3).

## Repository layout
Every run writes `results/<run_id>/{*.csv, manifest.json}`; the manifest
records data checksums, configuration, git commit, and the runtime feature
audit (whitelist assertion).

## Setup

```bash
# Python >= 3.9
pip install -r requirements.txt
```

## Data placement

All datasets are public and are **not** redistributed here:
- NASA PCoE: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
- CALCE CS2: https://calce.umd.edu/battery-data
- eVTOL: Bills et al., *Scientific Data* 10:344 (2023)

Build the CALCE cycle cache once from the raw Arbin archives:

```bash
python scripts/prepare_calce.py --data-dir ./Data
```

## Reproducing the paper

```bash
# 1. Coulomb-tautology audit — Table 2, CC rows of Tables 4/S3/S4
python scripts/exp_coulomb_audit.py --data-dir ./Data

# 2. Main comparison (8 seeds, inductive LOCO) — Tables 3-5
python scripts/run_all.py --data-dir ./Data --stages all --seeds 8

# 3. Factorial repair decomposition — Table 6 / S6
python scripts/exp_factorial_repair.py --data-dir ./Data \
    --pred-csv results/<run_id>/predictions_per_sample.csv

# 4. Prefix-truncation stress test — Fig. 4 / Table S7
python scripts/exp_dod_curve.py --data-dir ./Data

# smoke test
python scripts/run_all.py --data-dir ./Data --stages e3 --quick
```

## The five-item audit checklist (paper Sec. 7) -> where it is enforced

| # | Checklist item | Implementation |
|---|---|---|
| 1 | Plain coulomb-counting baseline on every target, with each benchmark's label-generation equation | `scripts/exp_coulomb_audit.py` |
| 2 | Deployment observability model + feature audit, enforced in code | `hipif/features/schema.py` whitelist + runtime assertion in `run_all.py` |
| 3 | Inductive cell-level splits; held-out cell excluded from adaptation and selection | `hipif/splits.py` (`target_loco`, `assert_disjoint`) |
| 4 | Causal vs. retrospective evaluation, stated for trajectory-level post-processing | causal-prefix mode of the repair operator in `run_all.py` |
| 5 | Partial-depth-of-discharge performance (or explicit full-discharge scope) | `scripts/exp_dod_curve.py` |

## Tests

```bash
python -m pytest tests/ -v
```

Unit tests assert all four repair constraints on every repaired output,
including `100*Q_t/[Q_nom(1+tau)] <= 100` so the final clip cannot break the
coulomb floor (paper Sec. 5.1).

## License

MIT — see `LICENSE`.
