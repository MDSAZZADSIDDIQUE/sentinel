"""Three-way analysis split — prevents control contamination (Gating A).

An early-warning model may only be trained/evaluated on:
  * POSITIVE  : Sepsis-3 with onset >= H (a genuine pre-onset window exists), and
  * CONTROL   : never meets Sepsis-3 at any hour (sepsis3 == 0).
Everything else — Sepsis-3 with onset < H (too early to anticipate) and
present-on-admission — is a THIRD, EXCLUDED group. These ambiguous-positive
stays must NOT sit in the controls, or every downstream metric is meaningless.

Output: data/labels/analysis_groups_<mode>.parquet (stay_id, analysis_group,
onset_hour, has_suspicion) — consumed by the Phase 2 feature/label builder.
"""
from __future__ import annotations

import time

import pandas as pd

from ..config import CohortConfig, LabelConfig
from ..logging_utils import get_logger
from ..paths import PATHS
from .sepsis3 import sepsis3_path

log = get_logger("labels.splits")

POSITIVE = "positive"
EXCLUDED = "excluded_prevalent_or_early"
CONTROL = "control"


def analysis_group_path(cfg: CohortConfig):
    return PATHS.labels_root / f"analysis_groups_{cfg.mode}.parquet"


def assign_groups(sep: pd.DataFrame, horizon_hours: int) -> pd.Series:
    """Vectorized three-way assignment from a sepsis3 dataframe."""
    septic = sep["sepsis3"] == 1
    onset = sep["onset_hour"].fillna(-1).astype("int64")
    is_positive = septic & (onset >= horizon_hours)
    is_excluded = septic & (onset < horizon_hours)   # incl. present-on-admission
    group = pd.Series(CONTROL, index=sep.index, dtype="object")
    group[is_excluded] = EXCLUDED
    group[is_positive] = POSITIVE
    return group


def build_splits(cfg: CohortConfig, lcfg: LabelConfig) -> pd.DataFrame:
    sep = pd.read_parquet(sepsis3_path(cfg))
    sep = sep.copy()
    sep["analysis_group"] = assign_groups(sep, lcfg.horizon_hours)

    out_df = sep[["stay_id", "analysis_group", "onset_hour", "has_suspicion", "sepsis3"]].copy()
    PATHS.labels_root.mkdir(parents=True, exist_ok=True)
    out = analysis_group_path(cfg)
    out_df.to_parquet(out, index=False)

    counts = out_df["analysis_group"].value_counts()
    n = len(out_df)
    log.info("three-way split (H=%dh):", lcfg.horizon_hours)
    for grp in (POSITIVE, EXCLUDED, CONTROL):
        c = int(counts.get(grp, 0))
        log.info("  %-28s %7d  (%.1f%%)", grp, c, 100 * c / n)
    # transparency: controls that nonetheless had suspicion of infection
    ctrl = out_df[out_df["analysis_group"] == CONTROL]
    susp_ctrl = int(ctrl["has_suspicion"].sum())
    log.info("  (of controls, %d had suspicion-of-infection but no qualifying SOFA rise)",
             susp_ctrl)
    # integrity check: counts must sum and groups are disjoint
    assert counts.sum() == n, "analysis groups must partition the cohort"
    log.info("  wrote %s", out)
    return out_df


def run(cfg: CohortConfig | None = None, lcfg: LabelConfig | None = None) -> None:
    cfg = cfg or CohortConfig.load()
    lcfg = lcfg or LabelConfig.load()
    t0 = time.perf_counter()
    build_splits(cfg, lcfg)
    log.info("splits done in %.1fs", time.perf_counter() - t0)
