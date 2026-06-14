"""Assemble episodes from the hourly feature matrix: windows, labels, norm:

  * observation window per stay = [max(0, end-max_obs), end-1], strictly < end
    (no leakage). end = onset (positive) or a length-matched pseudo-onset drawn
    from the positives' onset distribution (control), so episode LENGTH cannot
    proxy the label (spec D).
  * per-hour label y(t)=1 iff onset within the next alert_window hours.
  * z-score continuous features on TRAIN rows only; persist stats.
  * write a per-organ feature manifest for the Dec-POMDP agents.

Outputs: data/features/hourly_<mode>.parquet, norm_stats_<mode>.json,
         feature_manifest_<mode>.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import CohortConfig, FeatureConfig, LabelConfig
from ..constants import SHARED_GROUP
from ..logging_utils import get_logger
from ..paths import PATHS
from ..labels.sepsis3 import sepsis3_path
from .build import MEASURED_VARS, ORGAN_VALUE_FEATURES, hourly_path
from .partition import partition_path

log = get_logger("features.episodes")

VALUE_COLS = sorted({v for vs in ORGAN_VALUE_FEATURES.values() for v in vs} - {"age"})
CONT_COLS = [c for c in VALUE_COLS if c != "vent"]  # vent is binary


def _episode_ends(model: pd.DataFrame, fcfg: FeatureConfig) -> np.ndarray:
    """End hour per stay: onset (positive) or length-matched pseudo-onset (control)."""
    rng = np.random.default_rng(fcfg.seed)
    is_pos = (model["label"] == 1).to_numpy()
    onset = model["onset_hour"].to_numpy()
    los_floor = np.floor(model["los_hours"].to_numpy()).astype(int)
    pos_onsets = model.loc[model["label"] == 1, "onset_hour"].dropna().astype(int).to_numpy()
    if len(pos_onsets) == 0:
        pos_onsets = np.array([fcfg.max_obs_hours])
    pseudo = rng.choice(pos_onsets, size=len(model))
    # controls: pseudo-onset capped at LOS; positives: actual onset
    end = np.where(is_pos, np.nan_to_num(onset, nan=fcfg.min_obs_hours),
                   np.minimum(pseudo, los_floor)).astype(int)
    lo = fcfg.min_obs_hours
    end = np.clip(end, lo, np.maximum(los_floor, lo))
    return end


def assemble_and_write(cfg: CohortConfig, lcfg: LabelConfig, fcfg: FeatureConfig,
                       feat: pd.DataFrame, cohort: pd.DataFrame) -> "object":
    part = pd.read_parquet(partition_path(cfg))
    model = part[part["label"].notna()].copy()
    model["label"] = model["label"].astype(int)
    if fcfg.drop_suspected_controls:
        # optionally drop suspected-not-septic controls (ambiguous negatives)
        susp = pd.read_parquet(sepsis3_path(cfg))[["stay_id", "has_suspicion"]]
        model = model.merge(susp, on="stay_id", how="left")
        drop = (model["label"] == 0) & (model["has_suspicion"] == 1)
        log.info("dropping %d suspected-not-septic controls", int(drop.sum()))
        model = model[~drop].drop(columns=["has_suspicion"])

    sep = pd.read_parquet(sepsis3_path(cfg))[["stay_id", "onset_hour"]]
    model = model.merge(sep, on="stay_id", how="left")
    # `model` (from the partition) already carries los_hours.

    model["end_hour"] = _episode_ends(model, fcfg)
    model["obs_start"] = np.maximum(0, model["end_hour"] - fcfg.max_obs_hours)

    meta = model[["stay_id", "split", "label", "onset_hour", "end_hour", "obs_start",
                  "age", "gender", "site"]]
    df = feat.merge(meta, on="stay_id", how="inner")
    # observation window: [obs_start, end_hour) — strictly before onset (leakage-safe)
    df = df[(df["hr"] >= df["obs_start"]) & (df["hr"] < df["end_hour"])].copy()

    lead = df["onset_hour"] - df["hr"]
    df["y"] = ((df["label"] == 1) & (lead >= 1) & (lead <= fcfg.alert_window_hours)).astype("int8")
    df["t_to_onset"] = np.where(df["label"] == 1, lead, np.nan).astype("float32")

    # --- normalization on TRAIN rows only ---
    train = df["split"] == "train"
    stats: dict[str, dict] = {}
    for c in CONT_COLS:
        mu = float(df.loc[train, c].mean())
        sd = float(df.loc[train, c].std())
        sd = sd if sd and sd > 1e-6 else 1.0
        stats[c] = {"mean": mu, "std": sd}
        df[c] = ((df[c] - mu) / sd).astype("float32")
    df[CONT_COLS] = df[CONT_COLS].fillna(0.0)  # post-norm NaN -> train mean (0)
    amu, asd = float(df.loc[train, "age"].mean()), float(df.loc[train, "age"].std() or 1.0)
    stats["age"] = {"mean": amu, "std": asd}
    df["age_z"] = ((df["age"] - amu) / asd).astype("float32")
    df["sex_female"] = (df["gender"] == "F").astype("int8")

    # --- per-organ feature manifest (for the Dec-POMDP agents) ---
    manifest: dict[str, list[str]] = {}
    for organ, vs in ORGAN_VALUE_FEATURES.items():
        cols: list[str] = []
        for v in vs:
            if v == "age":
                continue
            cols.append(v)
            if fcfg.include_measurement_features and v in MEASURED_VARS:
                cols += [f"{v}__measured", f"{v}__mask"]
        manifest[organ] = cols
    manifest[SHARED_GROUP] = manifest.get(SHARED_GROUP, []) + ["age_z", "sex_female"]

    keep = (["stay_id", "hr", "split", "label", "y", "t_to_onset", "end_hour"]
            + [c for cols in manifest.values() for c in cols])
    keep = [c for c in dict.fromkeys(keep) if c in df.columns]
    out = df[keep].sort_values(["stay_id", "hr"]).reset_index(drop=True)

    PATHS.features_root.mkdir(parents=True, exist_ok=True)
    out.to_parquet(hourly_path(cfg), index=False)
    with (PATHS.features_root / f"norm_stats_{cfg.mode}.json").open("w") as fh:
        json.dump(stats, fh, indent=2)
    with (PATHS.features_root / f"feature_manifest_{cfg.mode}.json").open("w") as fh:
        json.dump(manifest, fh, indent=2)

    # --- summary ---
    n_ep = out["stay_id"].nunique()
    log.info("episodes: %d stays | %s hourly rows | %d feature cols",
             n_ep, f"{len(out):,}", sum(len(c) for c in manifest.values()))
    for s in ("train", "val", "test", "external"):
        sub = out[out["split"] == s]
        if len(sub):
            ep = sub.groupby("stay_id")["label"].first()
            log.info("  %-9s %7s rows | %5d episodes (%4d pos) | %.1f%% positive hours",
                     s, f"{len(sub):,}", len(ep), int((ep == 1).sum()), 100 * sub["y"].mean())
    # LOS-proxy guard: episode length by class should overlap
    elen = out.groupby("stay_id").agg(label=("label", "first"), n=("hr", "size"))
    log.info("  episode length (hours): pos median=%.0f, ctrl median=%.0f (aligned)",
             elen.loc[elen.label == 1, "n"].median(), elen.loc[elen.label == 0, "n"].median())
    log.info("  wrote %s", hourly_path(cfg))
    return hourly_path(cfg)
