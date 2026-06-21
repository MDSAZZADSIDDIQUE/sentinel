"""eICU partition (Phase 9): every stay is the EXTERNAL split.

Unlike features/partition.py (temporal train/val/test + held-out care unit),
ALL eICU stays are external held-out test data — the MIMIC-trained models are
frozen and only evaluated here. Output schema matches partition_<mode>.parquet
so features/dataset.py:load_split and the episode assembler consume it unchanged.

Output: data/features/partition_eicu.parquet
"""
from __future__ import annotations

import time

import pandas as pd

from ..config import CohortConfig, LabelConfig
from ..data.cohort import load_cohort
from ..logging_utils import get_logger
from ..paths import PATHS
from ..labels.splits import CONTROL, POSITIVE, analysis_group_path
from .partition import partition_path

log = get_logger("features.eicu_partition")


def build_partition_eicu(cfg: CohortConfig, lcfg: LabelConfig) -> pd.DataFrame:
    cohort = load_cohort(cfg)
    groups = pd.read_parquet(analysis_group_path(cfg))[["stay_id", "analysis_group"]]
    df = cohort.merge(groups, on="stay_id", how="left")

    df["split"] = "external"                       # all eICU = external test
    label_map = {POSITIVE: 1, CONTROL: 0}          # EXCLUDED -> NA (dropped)
    df["label"] = df["analysis_group"].map(label_map).astype("Int64")

    out_cols = ["stay_id", "subject_id", "hadm_id", "site", "anchor_year_group",
                "split", "analysis_group", "label", "los_hours", "age", "age_band",
                "gender", "race"]
    out = df[out_cols].copy()
    PATHS.features_root.mkdir(parents=True, exist_ok=True)
    p = partition_path(cfg)
    out.to_parquet(p, index=False)

    model = out[out["label"].notna()]
    npos = int((model["label"] == 1).sum())
    nctrl = int((model["label"] == 0).sum())
    log.info("eICU partition: all external | %d modeling stays (%d pos / %d ctrl) | %d excluded",
             len(model), npos, nctrl, int(out["label"].isna().sum()))
    log.info("  wrote %s", p)
    return out


def run(cfg: CohortConfig | None = None, lcfg: LabelConfig | None = None) -> None:
    cfg = cfg or CohortConfig(mode="eicu")
    lcfg = lcfg or LabelConfig.load()
    t0 = time.perf_counter()
    build_partition_eicu(cfg, lcfg)
    log.info("eICU partition done in %.1fs", time.perf_counter() - t0)
