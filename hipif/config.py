"""Central configuration for HIPIF v2 (leakage-free protocol).

Changes vs v1 (plan v2.0, WP1/WP3):
  * NN input features contain NO target measured temperature and NO
    end-of-life-normalised cycle features (A1/A3). See hipif.features.schema.
  * Chemistry parameters are loaded from configs/physics/chemistry_registry.yaml
    with units and provenance (P3/P8). No dataset-name branches.
  * `input_dim` follows the schema; do not set it manually.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import torch
import yaml

R_GAS: float = 8.314            # J/(mol K)
KELVIN_OFFSET: float = 273.15
SOH_MIN: float = 50.0
SOH_MAX: float = 100.0

_REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = _REPO_ROOT / "configs" / "physics" / "chemistry_registry.yaml"


def load_registry(path: Path = REGISTRY_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ChemistryParams:
    """Constraint + thermal parameters for one chemistry, with provenance."""
    chemistry: str
    E_a: float                  # J/mol
    k0: float                   # pp per equivalent full cycle at T_ref
    Q_nom: float                # Ah
    ambient_C: float
    R_int: float                # ohm
    hA: float                   # W/K
    Cth: float                  # J/K
    T_ref_K: float = 298.15
    soh_min: float = SOH_MIN
    soh_max: float = SOH_MAX
    mono_eps: float = 0.5       # pp (I1 noise allowance)
    persistence_window: int = 3
    energy_tol: float = 0.05
    provenance: dict = field(default_factory=dict)

    @classmethod
    def from_registry(cls, chemistry: str, registry: Optional[dict] = None,
                      **overrides) -> "ChemistryParams":
        reg = registry or load_registry()
        d = reg["defaults"]
        try:
            c = reg["chemistries"][chemistry]
        except KeyError:
            raise KeyError(
                f"chemistry '{chemistry}' not in registry "
                f"{sorted(reg['chemistries'])}; add it to "
                f"configs/physics/chemistry_registry.yaml with provenance")
        p = cls(
            chemistry=chemistry,
            E_a=float(c["E_a_J_per_mol"]),
            k0=float(c["k0_pp_per_cycle"]),
            Q_nom=float(c["Q_nom_Ah"]),
            ambient_C=float(c["ambient_C"]),
            R_int=float(c["thermal"]["R_int_ohm"]),
            hA=float(c["thermal"]["hA_W_per_K"]),
            Cth=float(c["thermal"]["Cth_J_per_K"]),
            T_ref_K=float(d["T_ref_K"]),
            soh_min=float(d["soh_min_pct"]),
            soh_max=float(d["soh_max_pct"]),
            mono_eps=float(d["mono_eps_pp"]),
            persistence_window=int(d["persistence_window"]),
            energy_tol=float(d["energy_tol"]),
            provenance={k: c.get(k) for k in
                        ("E_a_source", "k0_source", "ambient_source",
                         "thermal_source")},
        )
        for k, v in overrides.items():
            if v is not None:
                setattr(p, k, v)
                p.provenance[f"{k}_override"] = "run-time override"
        return p

    def k_max(self, T_C):
        """Arrhenius kinetic bound k(T) [pp/cycle] at reconstructed temp T_C."""
        import numpy as np
        T_K = np.clip(np.asarray(T_C, dtype=float) + KELVIN_OFFSET, 220.0, 350.0)
        return self.k0 * np.exp(-self.E_a / R_GAS * (1.0 / T_K - 1.0 / self.T_ref_K))

    def as_dict(self) -> dict:
        return {
            "chemistry": self.chemistry, "E_a_J_per_mol": self.E_a,
            "k0_pp_per_cycle": self.k0, "Q_nom_Ah": self.Q_nom,
            "ambient_C": self.ambient_C, "R_int_ohm": self.R_int,
            "hA_W_per_K": self.hA, "Cth_J_per_K": self.Cth,
            "mono_eps_pp": self.mono_eps,
            "persistence_window": self.persistence_window,
            "energy_tol": self.energy_tol, "provenance": self.provenance,
        }


@dataclass
class HIPIFConfig:
    """Run configuration. NN features are defined by hipif.features.schema
    (MODEL_FEATURES); measured target temperature and EOL-normalised cycle
    features are structurally excluded (A1/A3)."""
    seed: int = 42
    input_dim: int = 7          # == len(schema.MODEL_FEATURES); asserted at runtime
    width: int = 118
    temp_residual_hidden: int = 32
    uncertainty_hidden: int = 32
    lr: float = 5e-4
    weight_decay: float = 1e-4
    epochs_pretrain: int = 300
    epochs_refine: int = 40
    batch_size: int = 64
    # loss weights (frozen from SOURCE validation only; P4)
    lambda_phys: float = 0.6
    lambda_temp: float = 0.5
    lambda_residual_reg: float = 0.1
    lambda_target: float = 1.0      # adaptation: projected pseudo-label loss
    lambda_source: float = 0.5      # adaptation: source anchor loss
    ema_momentum: float = 0.9       # teacher EMA per refine iteration
    n_refine_iters: int = 14
    drift_eps: float = 2e-4
    # physics
    chemistry: str = "LFP"
    active_constraints: Tuple[str, ...] = ("I1", "I2", "I3", "I4")
    Ea_override: Optional[float] = None
    k0_override: Optional[float] = None
    qnom_override: Optional[float] = None
    device: torch.device = field(default_factory=_default_device)
    base_dir: Path = field(default_factory=lambda: Path("."))
    data_dir: Path = field(default_factory=lambda: Path("./Data"))
    results_dir: Path = field(default_factory=lambda: Path("./results"))

    _chem_cache: Optional[ChemistryParams] = field(
        default=None, repr=False, compare=False)

    @property
    def chem(self) -> ChemistryParams:
        if (self._chem_cache is None
                or self._chem_cache.chemistry != self.chemistry
                or self._chem_cache.provenance.get("_ovr") != (
                    self.Ea_override, self.k0_override, self.qnom_override)):
            self._chem_cache = ChemistryParams.from_registry(
                self.chemistry,
                E_a=self.Ea_override, k0=self.k0_override,
                Q_nom=self.qnom_override)
            self._chem_cache.provenance["_ovr"] = (
                self.Ea_override, self.k0_override, self.qnom_override)
        return self._chem_cache

    # Backwards-compatible aliases used by models/losses
    @property
    def E_a(self) -> float: return self.chem.E_a
    @property
    def Q_nom(self) -> float: return self.chem.Q_nom
    @property
    def mono_thresh(self) -> float: return self.chem.mono_eps
    @property
    def persistence_window(self) -> int: return self.chem.persistence_window
    @property
    def energy_tol(self) -> float: return self.chem.energy_tol
    @property
    def arrhenius_kmax_lab(self) -> float: return self.chem.k0

    def as_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__
             if not k.startswith("_")}
        d["device"] = str(d["device"])
        for k in ("base_dir", "data_dir", "results_dir"):
            d[k] = str(d[k])
        d["chem"] = self.chem.as_dict()
        return d
