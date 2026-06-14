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

BEHAVIOR_SUFFIXES = ("__measured", "__mask")
BEHAVIOR_EXTRA = ("hour_idx",)


def _manifest(cfg: CohortConfig) -> dict:
    with (PATHS.features_root / f"feature_manifest_{cfg.mode}.json").open() as fh:
        return json.load(fh)


def is_behavior_col(c: str) -> bool:
    return c.endswith(BEHAVIOR_SUFFIXES) or c in BEHAVIOR_EXTRA


def feature_columns(manifest: dict, physiology_only: bool = False) -> tuple[list[str], dict]:
    """Flat ordered feature list + per-organ manifest, honoring the ablation."""
    fman: dict[str, list[str]] = {}
    for organ, cols in manifest.items():
        kept = [c for c in cols if not (physiology_only and is_behavior_col(c))]
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


def load_split(cfg: CohortConfig, split: str, *, physiology_only: bool = False,
               df: pd.DataFrame | None = None) -> SplitData:
    df = df if df is not None else load_hourly(cfg)
    sub = df[df["split"] == split]
    if sub.empty:
        raise ValueError(f"no rows for split={split!r} (mode={cfg.mode})")
    cols, fman = feature_columns(_manifest(cfg), physiology_only)
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
