"""NEWS2 early-warning score (rule-based baseline).

Computes the National Early Warning Score 2 from the (de-normalized) hourly
vitals. NEWS is inherently physiology-only — no training, no measurement
channels — so it doubles as the clinical reference point the learned models must
beat. Higher score = sicker; used directly as the per-hour risk score.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import CohortConfig
from ..paths import PATHS


def load_norm_stats(cfg: CohortConfig) -> dict:
    with (PATHS.features_root / f"norm_stats_{cfg.mode}.json").open() as fh:
        return json.load(fh)


def _denorm(z: np.ndarray, stat: dict) -> np.ndarray:
    return z * stat["std"] + stat["mean"]


def news_score(df: pd.DataFrame, norm_stats: dict) -> np.ndarray:
    """Per-row NEWS2 (0-20ish). df holds z-scored features; we invert to raw."""
    def raw(col):
        return _denorm(df[col].to_numpy(dtype=float), norm_stats[col])

    rr, spo2, temp = raw("resp_rate"), raw("spo2"), raw("temp_c")
    sbp, hr, gcs = raw("sbp"), raw("heart_rate"), raw("gcs_total")
    fio2 = raw("fio2")
    vent = df["vent"].to_numpy(dtype=float) if "vent" in df else np.zeros(len(df))
    suppl_o2 = ((vent >= 0.5) | (fio2 > 0.22)).astype(int)

    s_rr = np.select([rr <= 8, rr <= 11, rr <= 20, rr <= 24], [3, 1, 0, 2], default=3)
    s_spo2 = np.select([spo2 <= 91, spo2 <= 93, spo2 <= 95], [3, 2, 1], default=0)
    s_temp = np.select([temp <= 35.0, temp <= 36.0, temp <= 38.0, temp <= 39.0],
                       [3, 1, 0, 1], default=2)
    s_sbp = np.select([sbp <= 90, sbp <= 100, sbp <= 110, sbp <= 219], [3, 2, 1, 0], default=3)
    s_hr = np.select([hr <= 40, hr <= 50, hr <= 90, hr <= 110, hr <= 130],
                     [3, 1, 0, 1, 2], default=3)
    s_cns = np.where(gcs < 15, 3, 0)
    s_o2 = suppl_o2 * 2
    total = s_rr + s_spo2 + s_temp + s_sbp + s_hr + s_cns + s_o2
    return total.astype(np.float32)


def news_scores_for_split(cfg: CohortConfig, split_df: pd.DataFrame) -> np.ndarray:
    return news_score(split_df, load_norm_stats(cfg))
