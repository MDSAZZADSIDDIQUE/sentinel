"""Data partition for modeling: train / val / test (temporal) + external site.

Two independent generalization axes, both leakage-safe:
  * TEMPORAL — internal stays split by `anchor_year_group` (the only honest time
    axis in MIMIC-IV, whose calendar dates are per-patient shifted): train on the
    earlier bands, validate on 2017-2019, test on the latest 2020-2022.
  * SITE — the external care unit (CVICU) is held out ENTIRELY (all years) and
    never seen in training; it is the external-validation set.

Normalization statistics (Phase 2 features) are computed on TRAIN only. The
modeling label is binary over the three-way split: POSITIVE=1, CONTROL=0,
EXCLUDED -> NA (dropped from modeling).

Output: data/features/partition_<mode>.parquet
"""
from __future__ import annotations

import time

import pandas as pd

from ..config import CohortConfig, LabelConfig
from ..data.cohort import load_cohort
from ..logging_utils import get_logger
from ..paths import PATHS
from ..labels.splits import CONTROL, POSITIVE, analysis_group_path

log = get_logger("features.partition")

# anchor_year_group -> temporal split (internal sites only)
TEMPORAL_SPLIT = {
    "2008 - 2010": "train",
    "2011 - 2013": "train",
    "2014 - 2016": "train",
    "2017 - 2019": "val",
    "2020 - 2022": "test",
}
SPLITS = ("train", "val", "test", "external")


def partition_path(cfg: CohortConfig):
    return PATHS.features_root / f"partition_{cfg.mode}.parquet"


def build_partition(cfg: CohortConfig, lcfg: LabelConfig) -> pd.DataFrame:
    cohort = load_cohort(cfg)
    groups = pd.read_parquet(analysis_group_path(cfg))[["stay_id", "analysis_group"]]
    df = cohort.merge(groups, on="stay_id", how="left")

    ext = cfg.external_site

    def split_of(row) -> str:
        if row["site"] == ext:
            return "external"
        return TEMPORAL_SPLIT.get(row["anchor_year_group"], "train")

    df["split"] = df.apply(split_of, axis=1)
    # binary modeling label; EXCLUDED -> NA (dropped from train/eval)
    label_map = {POSITIVE: 1, CONTROL: 0}
    df["label"] = df["analysis_group"].map(label_map).astype("Int64")

    out_cols = ["stay_id", "subject_id", "hadm_id", "site", "anchor_year_group",
                "split", "analysis_group", "label", "los_hours", "age", "age_band",
                "gender", "race"]
    out = df[out_cols].copy()
    PATHS.features_root.mkdir(parents=True, exist_ok=True)
    p = partition_path(cfg)
    out.to_parquet(p, index=False)

    # report split x class
    model = out[out["label"].notna()]
    log.info("partition (mode=%s): external site = %s", cfg.mode, ext)
    log.info("  %-9s %8s %8s %8s   (positives / controls)", "split", "n", "pos", "ctrl")
    for s in SPLITS:
        sub = model[model["split"] == s]
        npos = int((sub["label"] == 1).sum())
        nctrl = int((sub["label"] == 0).sum())
        log.info("  %-9s %8d %8d %8d", s, len(sub), npos, nctrl)
    n_excl = int(out["label"].isna().sum())
    log.info("  excluded (held out of modeling): %d", n_excl)
    if (model["split"] == "external").sum() == 0:
        log.warning("  no external-site stays in this cohort (expected for dev mode)")
    log.info("  wrote %s", p)
    return out


def run(cfg: CohortConfig | None = None, lcfg: LabelConfig | None = None) -> None:
    cfg = cfg or CohortConfig.load()
    lcfg = lcfg or LabelConfig.load()
    t0 = time.perf_counter()
    build_partition(cfg, lcfg)
    log.info("partition done in %.1fs", time.perf_counter() - t0)
