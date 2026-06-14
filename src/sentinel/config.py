"""Typed YAML config loading (plain YAML, not Hydra).

We deliberately avoid Hydra: its working-directory rewriting and multirun magic
fight Windows path handling and add reproducibility surface area we don't need.
Plain YAML + small frozen dataclasses give explicit, testable config.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from .paths import PATHS


def read_yaml(name_or_path: str | Path) -> dict[str, Any]:
    p = Path(name_or_path)
    if not p.is_absolute() and p.suffix == "":
        p = PATHS.config_root / f"{name_or_path}.yaml"
    elif not p.is_absolute():
        p = PATHS.config_root / p
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _filter_known(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    known = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in known}


@dataclass(frozen=True)
class CohortConfig:
    """ICU cohort inclusion + site-partition settings (config/cohort.yaml)."""

    mode: str = "dev"  # "dev" | "full"
    seed: int = 7

    # inclusion criteria
    min_age: int = 18
    max_age: int = 89  # MIMIC caps ages >89 at 91; exclude the artifact band
    min_los_hours: float = 6.0
    first_stay_only: bool = True  # first ICU stay per hospital admission

    # dev-mode subsampling for fast iteration
    dev_careunits: list[str] = field(default_factory=lambda: [
        "Medical Intensive Care Unit (MICU)",
        "Surgical Intensive Care Unit (SICU)",
    ])
    dev_max_stays: int = 4000

    # federated site partition (Phase 5) — by admitting ICU
    site_partition_key: str = "first_careunit"
    external_site: str = "Cardiac Vascular Intensive Care Unit (CVICU)"

    @classmethod
    def load(cls, name: str = "cohort") -> "CohortConfig":
        try:
            data = read_yaml(name)
        except FileNotFoundError:
            return cls()
        return cls(**_filter_known(cls, data))


@dataclass(frozen=True)
class LabelConfig:
    """Sepsis-3 / SOFA derivation settings (config/labels.yaml)."""

    # early-warning prediction horizon
    horizon_hours: int = 6

    # SOFA: rolling worst-value window for each component (hours)
    sofa_window_hours: int = 24

    # Sepsis-3 acute SOFA rise required for organ dysfunction
    sofa_increase_threshold: int = 2
    # baseline SOFA assumption (Sepsis-3 assumes 0 unless chronic dysfunction known)
    sofa_baseline: int = 0

    # Suspicion of infection time window relative to antibiotic/culture
    # (Seymour 2016 / mimic-code): if antibiotic first, culture within +72h;
    # if culture first, antibiotic within +24h.
    si_abx_then_culture_hours: int = 72
    si_culture_then_abx_hours: int = 24

    # Sepsis onset window: SOFA rise must occur within [-48h, +24h] of the
    # suspicion-of-infection time (mimic-code sepsis3 convention).
    si_sofa_lookback_hours: int = 48
    si_sofa_lookahead_hours: int = 24

    @classmethod
    def load(cls, name: str = "labels") -> "LabelConfig":
        try:
            data = read_yaml(name)
        except FileNotFoundError:
            return cls()
        return cls(**_filter_known(cls, data))
