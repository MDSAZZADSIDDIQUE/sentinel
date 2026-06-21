"""Build the eICU external-validation cohort from the `patient` table (Phase 9).

eICU analog of data/cohort.py. The `patient` table is small (~200k rows), so we
filter + dedup in DuckDB and materialize a small cohort parquet. Time is in
OFFSETS (minutes from unit admit); there are no timestamps. Keys: patientunitstayid
(stay) / patienthealthsystemstayid (hospital stay) / uniquepid (patient) /
hospitalid (site).

Inclusion mirrors MIMIC (config/cohort.yaml) as closely as the schema allows:
  * adults: age >= min_age. eICU censors ages >89 as the string '> 89'; we parse
    that to 90 and RETAIN those stays (MIMIC instead excludes its own >89 de-id
    band at anchor_age=91). This is a documented, minor cohort difference — the
    frozen MIMIC normalizer handles age=90 within range. No upper age cut.
  * LOS >= min_los_hours (unitdischargeoffset/60).
  * first ELIGIBLE unit stay per uniquepid (one stay per patient; absolute admit
    order is unavailable with offsets, so we order by hospitaldischargeyear, then
    unitvisitnumber, then stay_id — deterministic, prefers the first ICU visit).

Output: data/cohort/cohort_eicu.parquet (consumed by eICU SOFA/suspicion/report,
and by the Phase-9 partition/feature builders with all split="external").
"""
from __future__ import annotations

import time

import pandas as pd

from ..config import CohortConfig
from ..duck import connect, eicu
from ..logging_utils import get_logger
from ..paths import PATHS
from .cohort import cohort_path  # path is mode-keyed -> cohort_eicu.parquet

log = get_logger("data.eicu_cohort")

EICU_MODE = "eicu"

# Superset schema: report needs (stay_id, subject_id, age, gender, los_hours,
# hospital_expire_flag, race, site); partition/episodes need (subject_id, hadm_id,
# site, age_band, anchor_year_group, race); federation needs hospitalid; SOFA grid
# needs (stay_id, los_hours); vaso ug/kg/min fallback needs weight_kg.
COHORT_COLUMNS = [
    "subject_id", "hadm_id", "stay_id", "uniquepid", "hospitalid", "site",
    "unittype", "los_hours", "unitdischargeoffset", "hospitaladmitoffset",
    "age", "age_band", "gender", "race", "anchor_year_group",
    "hospitaldischargeyear", "hospital_expire_flag", "weight_kg",
]


def _eligible_sql(cfg: CohortConfig) -> str:
    """Eligible-stay SQL with age parsing, LOS filter, first-per-patient rank."""
    return f"""
    WITH base AS (
        SELECT
            patientunitstayid                      AS stay_id,
            patienthealthsystemstayid              AS hadm_id,
            uniquepid                              AS subject_id,
            uniquepid,
            hospitalid,
            CAST(hospitalid AS VARCHAR)            AS site,
            unittype,
            admissionweight                        AS weight_kg,
            unitvisitnumber,
            hospitaldischargeyear,
            unitdischargeoffset,
            hospitaladmitoffset,
            unitdischargeoffset / 60.0             AS los_hours,
            CASE
                WHEN replace(trim(age), ' ', '') = '>89' THEN 90
                WHEN TRY_CAST(age AS INTEGER) IS NOT NULL THEN TRY_CAST(age AS INTEGER)
                ELSE NULL
            END                                    AS age,
            CASE gender WHEN 'Male' THEN 'M' WHEN 'Female' THEN 'F' ELSE 'U' END AS gender,
            ethnicity                              AS race,
            CASE WHEN hospitaldischargestatus = 'Expired' THEN 1 ELSE 0 END AS hospital_expire_flag
        FROM {eicu('patient')}
    ),
    eligible AS (
        SELECT *,
            CASE
                WHEN age BETWEEN 18 AND 44 THEN '18-44'
                WHEN age BETWEEN 45 AND 64 THEN '45-64'
                WHEN age BETWEEN 65 AND 74 THEN '65-74'
                ELSE '75-89'
            END                                    AS age_band,
            'eICU-2014-2015'                       AS anchor_year_group
        FROM base
        WHERE age >= {cfg.min_age}              -- adults; >89 retained as 90 (no upper cut)
          AND los_hours >= {cfg.min_los_hours}
    )
    SELECT *,
        ROW_NUMBER() OVER (
            PARTITION BY uniquepid
            ORDER BY hospitaldischargeyear NULLS LAST, unitvisitnumber, stay_id
        ) AS patient_stay_rank
    FROM eligible
    """


def build(cfg: CohortConfig) -> tuple[object, pd.DataFrame]:
    con = connect()
    try:
        df = con.execute(_eligible_sql(cfg)).df()
    finally:
        con.close()

    n_elig = len(df)
    n_over89 = int((df["age"] == 90).sum())
    if cfg.first_stay_only:
        df = df[df["patient_stay_rank"] == 1].copy()
        log.info("first-stay-per-patient dedup: %d -> %d stays", n_elig, len(df))
    df = df.drop(columns=["patient_stay_rank", "unitvisitnumber"])

    df = df.sort_values("stay_id").reset_index(drop=True)
    df = df[COHORT_COLUMNS]

    PATHS.cohort_root.mkdir(parents=True, exist_ok=True)
    out = cohort_path(cfg)
    df.to_parquet(out, index=False)
    log.info("wrote eICU cohort: %s (%d stays, %d cols)", out, len(df), df.shape[1])
    log.info("  eligible-stay rows pre-dedup: %d | censored elderly (>89->90): %d (%.1f%%)",
             n_elig, n_over89, 100 * n_over89 / max(n_elig, 1))
    return out, df


def run(cfg: CohortConfig | None = None) -> None:
    cfg = cfg or CohortConfig(mode=EICU_MODE)
    t0 = time.perf_counter()
    log.info("building eICU cohort (mode=%s)", cfg.mode)
    _, df = build(cfg)

    n = len(df)
    mort = df["hospital_expire_flag"].mean() if n else 0.0
    log.info("eICU cohort: n=%d | patients=%d | hospitals=%d | in-hosp mortality=%.1f%%",
             n, df["subject_id"].nunique(), df["hospitalid"].nunique(), 100 * mort)
    log.info("  median age=%.0f | female=%.1f%% | median LOS=%.1fh",
             df["age"].median(), 100 * (df["gender"] == "F").mean(), df["los_hours"].median())
    log.info("  unit types: %s", dict(df["unittype"].value_counts()))
    log.info("  hospitals with >=200 stays: %d",
             int((df["hospitalid"].value_counts() >= 200).sum()))
    log.info("done in %.1fs", time.perf_counter() - t0)
