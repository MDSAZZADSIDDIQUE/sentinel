"""Load the hourly feature tensor for modeling, with the measurement-channel
ablation as a load-time toggle (no rebuild).

`physiology_only=True` drops the non-physiology signals — the measurement
channels (`__measured`, `__mask`, which encode clinician *ordering* behavior) and
the timing index (`hour_idx`, a residual length/timing proxy). The gap between
the full and physiology-only models is reported as a headline result: how much
performance is physiology vs clinician-behavior/timing (the Epic-sepsis failure
mode).
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import CohortConfig
from ..paths import PATHS

MEASUREMENT_SUFFIXES = ("__measured", "__mask")   # clinician-ordering behavior
TIMING_COLS = ("hour_idx",)                        # temporal prior

# Three ablation conditions decompose the non-physiology signal so we can tell
# the dangerous behavior leak apart from the more-benign timing prior:
#   full           : physiology + measurement-behavior + timing
#   no_measurement : drop measurement channels, KEEP timing  (isolates the leak)
#   physiology     : drop measurement AND timing             (pure physiology)
ABLATIONS = ("full", "no_measurement", "physiology")


def _manifest(cfg: CohortConfig) -> dict:
    with (PATHS.features_root / f"feature_manifest_{cfg.mode}.json").open() as fh:
        return json.load(fh)


def col_dropped(c: str, ablation: str) -> bool:
    is_measurement = c.endswith(MEASUREMENT_SUFFIXES)
    is_timing = c in TIMING_COLS
    if ablation == "full":
        return False
    if ablation == "no_measurement":
        return is_measurement
    if ablation == "physiology":
        return is_measurement or is_timing
    raise ValueError(f"unknown ablation {ablation!r}; expected one of {ABLATIONS}")


def feature_columns(manifest: dict, ablation: str = "full") -> tuple[list[str], dict]:
    """Flat ordered feature list + per-organ manifest, honoring the ablation."""
    fman: dict[str, list[str]] = {}
    for organ, cols in manifest.items():
        kept = [c for c in cols if not col_dropped(c, ablation)]
        fman[organ] = kept
    flat: list[str] = []
    for cols in fman.values():
        for c in cols:
            if c not in flat:
                flat.append(c)
    return flat, fman


@dataclass
class SplitData:
    X: np.ndarray            # [rows, n_features] hour-level
    y: np.ndarray            # [rows] per-hour label
    stay_ids: np.ndarray     # [rows]
    hr: np.ndarray           # [rows] hour since admission
    t_to_onset: np.ndarray   # [rows] onset - hr (nan for controls)
    ep_label: np.ndarray     # [rows] stay-level label (0/1)
    feature_names: list[str]
    manifest: dict


def load_hourly(cfg: CohortConfig) -> pd.DataFrame:
    p = PATHS.features_root / f"hourly_{cfg.mode}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing — run `sentinel build-features --mode {cfg.mode}`.")
    return pd.read_parquet(p)


def load_split(cfg: CohortConfig, split: str, *, ablation: str = "full",
               df: pd.DataFrame | None = None) -> SplitData:
    df = df if df is not None else load_hourly(cfg)
    sub = df[df["split"] == split]
    if sub.empty:
        raise ValueError(f"no rows for split={split!r} (mode={cfg.mode})")
    cols, fman = feature_columns(_manifest(cfg), ablation)
    cols = [c for c in cols if c in sub.columns]
    return SplitData(
        X=sub[cols].to_numpy(dtype=np.float32),
        y=sub["y"].to_numpy(dtype=np.int64),
        stay_ids=sub["stay_id"].to_numpy(),
        hr=sub["hr"].to_numpy(),
        t_to_onset=sub["t_to_onset"].to_numpy(dtype=np.float32),
        ep_label=sub["label"].to_numpy(dtype=np.int64),
        feature_names=cols,
        manifest=fman,
    )
