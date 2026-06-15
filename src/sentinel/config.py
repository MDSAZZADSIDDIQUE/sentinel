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
class FeatureConfig:
    """Hourly feature tensor + episode/observation settings (config/features.yaml)."""

    seed: int = 7
    # episode = the last `max_obs_hours` before the episode end (onset for
    # positives; a length-matched pseudo-onset for controls). Bounds sequence
    # length for the GTX 1650 and focuses on the pre-onset window.
    max_obs_hours: int = 72
    min_obs_hours: int = 6
    # per-hour positive label: y(t)=1 iff onset is within the next `alert_window`
    # hours, i.e. t in [onset-alert_window, onset-1].
    alert_window_hours: int = 6

    # respiratory surrogate: room-air FiO2 default + S/F fallback (Rice et al.).
    room_air_fio2: float = 0.21
    spo2_sf_cap: float = 97.0  # S/F invalid where SpO2 saturates the O2-Hb curve

    # measurement-frequency / missingness features (double-edged — clinician
    # behavior signal). Toggle for the with/without ablation (spec E).
    include_measurement_features: bool = True

    # drop suspected-not-septic controls from the modeling set (optional; they
    # are the ambiguous negatives flagged in the cohort report).
    drop_suspected_controls: bool = False

    @classmethod
    def load(cls, name: str = "features") -> "FeatureConfig":
        try:
            data = read_yaml(name)
        except FileNotFoundError:
            return cls()
        return cls(**_filter_known(cls, data))


@dataclass(frozen=True)
class MARLConfig:
    """SENTINEL MARL (MAPPO/IPPO/QMIX) settings (config/marl.yaml)."""

    algo: str = "mappo"            # mappo | ippo | qmix
    # MAPPO uses a centralized critic over the joint state; IPPO uses per-organ
    # decentralized critics (the coordination control — the ONLY difference).
    hidden_actor: int = 64
    hidden_critic: int = 128
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    ppo_epochs: int = 4
    updates: int = 60             # PPO updates (dev-light); scale for final
    batch_episodes: int = 128
    minibatch_episodes: int = 64
    grad_clip: float = 1.0
    eval_every: int = 10
    seed: int = 0
    ablation: str = "full"        # full | no_measurement | physiology

    @classmethod
    def load(cls, name: str = "marl") -> "MARLConfig":
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

    # Per-variable forward-fill windows live in labels/sofa.py:FFILL_HOURS (a
    # creatinine and an HR cannot share a validity window).
    # Intubated patients can't be assessed verbally; don't penalize GCS verbal=1T
    # as neurologic dysfunction (a major SOFA-CNS inflator otherwise).
    gcs_vent_adjust: bool = True

    # Sepsis-3 acute SOFA rise required for organ dysfunction
    sofa_increase_threshold: int = 2
    # Baseline SOFA. dynamic_baseline=True computes the *acute change* in SOFA per
    # the Sepsis-3 wording: baseline = min SOFA over the pre-suspicion admission
    # window [0, min(si_hour, baseline_window_hours)] (falls back to sofa_baseline
    # if no pre-suspicion data). This excludes chronically high-SOFA patients who
    # don't acutely deteriorate, matching published prevalence (~35-38%) and the
    # early-warning framing. Set False to assume baseline 0 (strict mimic-code).
    dynamic_baseline: bool = True
    baseline_window_hours: int = 24
    sofa_baseline: int = 0  # fallback baseline when no pre-suspicion data

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
