"""Build the ICU cohort from icustays + patients + admissions.

These three tables are small, so we join them in DuckDB and materialize the
(few-tens-of-thousands-row) cohort to pandas for inclusion filtering and the
deterministic dev subsample. The big event tables are never touched here.

Output: data/cohort/cohort_<mode>.parquet  (one row per included ICU stay).
"""
from __future__ import annotations

import time

import pandas as pd

from ..config import CohortConfig
from ..duck import connect, mimic
from ..logging_utils import get_logger
from ..paths import PATHS

log = get_logger("data.cohort")

COHORT_COLUMNS = [
    "subject_id", "hadm_id", "stay_id", "site", "first_careunit", "last_careunit",
    "intime", "outtime", "los_hours", "age", "age_band", "gender", "race",
    "admittime", "dischtime", "deathtime", "dod", "hospital_expire_flag",
]


def cohort_path(cfg: CohortConfig) -> "object":
    return PATHS.cohort_root / f"cohort_{cfg.mode}.parquet"


def _eligible_sql(cfg: CohortConfig) -> str:
    """Eligible-stay SQL (inclusion criteria + first-stay dedup), pre dev filter."""
    return f"""
    WITH base AS (
        SELECT
            i.subject_id, i.hadm_id, i.stay_id,
            i.first_careunit, i.last_careunit, i.intime, i.outtime,
            i.los * 24.0 AS los_hours,
            p.gender, p.anchor_age, p.anchor_year, p.dod,
            a.admittime, a.dischtime, a.deathtime, a.race, a.hospital_expire_flag,
            (p.anchor_age + (EXTRACT(YEAR FROM a.admittime) - p.anchor_year)) AS age,
            ROW_NUMBER() OVER (PARTITION BY i.hadm_id ORDER BY i.intime) AS stay_rank
        FROM {mimic('icustays')} i
        JOIN {mimic('patients')} p   ON p.subject_id = i.subject_id
        JOIN {mimic('admissions')} a ON a.hadm_id = i.hadm_id
    )
    SELECT
        subject_id, hadm_id, stay_id,
        first_careunit AS site, first_careunit, last_careunit,
        intime, outtime, los_hours, age,
        CASE
            WHEN age BETWEEN 18 AND 44 THEN '18-44'
            WHEN age BETWEEN 45 AND 64 THEN '45-64'
            WHEN age BETWEEN 65 AND 74 THEN '65-74'
            ELSE '75-89'
        END AS age_band,
        gender, race, admittime, dischtime, deathtime, dod, hospital_expire_flag
    FROM base
    WHERE age BETWEEN {cfg.min_age} AND {cfg.max_age}
      AND los_hours >= {cfg.min_los_hours}
      {"AND stay_rank = 1" if cfg.first_stay_only else ""}
    """


def build(cfg: CohortConfig) -> tuple[object, pd.DataFrame]:
    con = connect()
    try:
        df = con.execute(_eligible_sql(cfg)).df()
    finally:
        con.close()

    log.info("eligible stays after inclusion criteria: %d", len(df))
    careunits = df["first_careunit"].value_counts()
    log.info("care-unit distribution (eligible):")
    for unit, n in careunits.items():
        log.info("  %6d  %s", n, unit)

    if cfg.mode == "dev":
        before = len(df)
        df = df[df["first_careunit"].isin(cfg.dev_careunits)].copy()
        log.info("dev: restricted to %s -> %d stays (from %d)",
                 cfg.dev_careunits, len(df), before)
        if len(df) > cfg.dev_max_stays:
            df = df.sample(n=cfg.dev_max_stays, random_state=cfg.seed).copy()
            log.info("dev: subsampled to %d stays (seed=%d)", len(df), cfg.seed)
    elif cfg.mode != "full":
        raise ValueError(f"Unknown cohort mode: {cfg.mode!r} (expected dev|full)")

    df = df.sort_values("stay_id").reset_index(drop=True)
    df = df[COHORT_COLUMNS]

    PATHS.cohort_root.mkdir(parents=True, exist_ok=True)
    out = cohort_path(cfg)
    df.to_parquet(out, index=False)
    log.info("wrote cohort: %s (%d stays, %d cols)", out, len(df), df.shape[1])
    return out, df


def load_cohort(cfg: CohortConfig) -> pd.DataFrame:
    p = cohort_path(cfg)
    if not p.exists():
        raise FileNotFoundError(f"Cohort not built for mode={cfg.mode}: {p}. Run build-cohort.")
    return pd.read_parquet(p)


def run(cfg: CohortConfig | None = None) -> None:
    cfg = cfg or CohortConfig.load()
    t0 = time.perf_counter()
    log.info("building cohort (mode=%s)", cfg.mode)
    _, df = build(cfg)

    # quick descriptive summary (logged; full report comes in cohort-report)
    n = len(df)
    mort = df["hospital_expire_flag"].mean() if n else 0.0
    log.info("cohort summary: n=%d | in-hosp mortality=%.1f%% | median LOS=%.1fh | median age=%.0f",
             n, 100 * mort, df["los_hours"].median(), df["age"].median())
    log.info("  sites: %s", dict(df["site"].value_counts()))
    log.info("done in %.1fs", time.perf_counter() - t0)
