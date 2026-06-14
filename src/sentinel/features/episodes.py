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
    """End hour per stay.

    Positives: end = onset (episode = last min(onset, max_obs) hours before it).
    Controls: end = a pseudo length **sampled from the positives' episode-LENGTH
    distribution, restricted to lengths feasible within the control's own LOS** —
    this matches the full length *distribution* (not just the median), so episode
    length cannot proxy the label even in the tails. Residual mismatch is bounded
    by the (real, unchangeable) LOS difference between sick and well patients.
    """
    rng = np.random.default_rng(fcfg.seed)
    is_pos = (model["label"] == 1).to_numpy()
    onset = model["onset_hour"].to_numpy()
    los_floor = np.floor(model["los_hours"].to_numpy()).astype(int)

    pos_onset = model.loc[model["label"] == 1, "onset_hour"].dropna().astype(int).to_numpy()
    pos_len = np.minimum(pos_onset, fcfg.max_obs_hours)        # positive episode lengths
    pos_len = pos_len[pos_len >= fcfg.min_obs_hours]
    if len(pos_len) == 0:
        pos_len = np.array([fcfg.min_obs_hours])
    pos_sorted = np.sort(pos_len)

    caps = np.clip(los_floor, fcfg.min_obs_hours, None)        # max feasible length / stay
    k = np.searchsorted(pos_sorted, caps, side="right")        # feasible samples / stay
    ridx = np.minimum((rng.random(len(model)) * np.maximum(k, 1)).astype(int),
                      np.maximum(k - 1, 0))
    sampled = np.where(k > 0, pos_sorted[ridx], caps)
    sampled = np.minimum(sampled, caps)

    end = np.where(is_pos, np.nan_to_num(onset, nan=fcfg.min_obs_hours).astype(int), sampled)
    end = np.clip(end.astype(int), fcfg.min_obs_hours, np.maximum(los_floor, fcfg.min_obs_hours))
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
    # explicit, minimal shared time index (hours since admission, train-normalized)
    hmu, hsd = float(df.loc[train, "hr"].mean()), float(df.loc[train, "hr"].std() or 1.0)
    stats["hour_idx"] = {"mean": hmu, "std": hsd}
    df["hour_idx"] = ((df["hr"] - hmu) / hsd).astype("float32")

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
    # shared context is deliberately MINIMAL & explicit: demographics + time +
    # temperature (a global vital owned by no single organ). No organ's detailed
    # signal is shared, preserving agent specialization / interpretability.
    manifest[SHARED_GROUP] = manifest.get(SHARED_GROUP, []) + ["age_z", "sex_female", "hour_idx"]

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
    # LOS-proxy guard: episode-length DISTRIBUTION (not just median) by class
    elen = out.groupby("stay_id").agg(label=("label", "first"), n=("hr", "size"))
    qs = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    pq = elen.loc[elen.label == 1, "n"].quantile(qs).round(0).tolist()
    cq = elen.loc[elen.label == 0, "n"].quantile(qs).round(0).tolist()
    log.info("  episode length q%s: pos=%s ctrl=%s", [int(q * 100) for q in qs], pq, cq)
    # explicit per-agent observation composition (interpretability depends on this)
    log.info("  shared context (every agent): %s", manifest[SHARED_GROUP])
    for organ in [o for o in manifest if o != SHARED_GROUP]:
        log.info("    agent[%-14s] obs dim=%d", organ,
                 len(manifest[organ]) + len(manifest[SHARED_GROUP]))
    log.info("  wrote %s", hourly_path(cfg))
    return hourly_path(cfg)
