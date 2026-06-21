"""Suspicion of infection on eICU (Phase 9) — offset-time analog of suspicion.py.

THE eICU LABEL-FIDELITY CRUX. The Seymour-2016 / mimic-code rule pairs a
qualifying antibiotic with a culture (abx->culture +72 h or culture->abx +24 h).
But eICU's microLab is extremely sparse — only ~1.4 % of stays have ANY culture
recorded (vs MIMIC's dense microbiologyevents). Requiring a culture pair would
collapse suspicion to ~1-2 % and make Sepsis-3 prevalence meaningless.

So the PRIMARY eICU suspicion = a qualifying systemic antibiotic start (cultures
used only to refine si_time earlier when present). This is a documented deviation
forced by eICU's under-recorded microbiology. Crucially the downstream Sepsis-3
gate is UNCHANGED — onset still requires an *acute SOFA rise >= 2* within
[-48 h, +24 h] of suspicion (labels/sepsis3.py), which filters prophylactic-only
antibiotic stays that never deteriorate. The strict-pair coverage is computed
alongside for the honesty report (eicu_report.py).

Times are OFFSETS (minutes from unit admit). Output schema matches suspicion.py:
data/labels/suspicion_eicu.parquet (stay_id, hadm_id, si_time, si_hour,
has_suspicion) + extra coverage columns for the report.
"""
from __future__ import annotations

import time

import pandas as pd

from ..config import CohortConfig, LabelConfig, read_yaml
from ..duck import connect, eicu
from ..logging_utils import get_logger
from ..paths import PATHS
from ..data.cohort import load_cohort
from .suspicion import suspicion_path

log = get_logger("labels.eicu_suspicion")

# Primary suspicion = qualifying antibiotic start (eICU cultures too sparse to
# require a pair). Set True to require the strict abx<->culture pair instead.
REQUIRE_CULTURE = False


def _antibiotic_names() -> list[str]:
    abx = read_yaml("antibiotics").get("names", [])
    extra = (read_yaml("eicu_harmonization").get("suspicion", {})
             .get("antibiotics", {}).get("eicu_extra", []))
    # dedup, lowercase, drop empties
    return sorted({str(n).lower() for n in [*abx, *extra] if str(n).strip()})


def build_suspicion_eicu(cfg: CohortConfig, lcfg: LabelConfig) -> "object":
    names = _antibiotic_names()
    like = " OR ".join(f"lower(drugname) LIKE '%{n}%'" for n in names)
    abx_then_cx = lcfg.si_abx_then_culture_hours * 60   # -> minutes
    cx_then_abx = lcfg.si_culture_then_abx_hours * 60

    cohort = load_cohort(cfg)[["stay_id", "hadm_id"]].copy()
    con = connect()
    try:
        con.register("cohort_df", cohort)
        con.execute("CREATE OR REPLACE TEMP TABLE cohort AS SELECT * FROM cohort_df")
        q = f"""
        WITH abx AS (
            SELECT DISTINCT m.patientunitstayid AS stay_id, m.drugstartoffset AS t
            FROM {eicu('medication')} m JOIN cohort c ON c.stay_id = m.patientunitstayid
            WHERE m.drugstartoffset IS NOT NULL AND ({like})
        ),
        cx AS (
            SELECT DISTINCT ml.patientunitstayid AS stay_id, ml.culturetakenoffset AS t
            FROM {eicu('microLab')} ml JOIN cohort c ON c.stay_id = ml.patientunitstayid
            WHERE ml.culturetakenoffset IS NOT NULL
        ),
        abx_first AS (SELECT stay_id, min(t) AS abx_time FROM abx GROUP BY stay_id),
        cx_first  AS (SELECT stay_id, min(t) AS cx_time  FROM cx  GROUP BY stay_id),
        pairs AS (
            SELECT a.stay_id, least(a.t, c.t) AS si_time
            FROM abx a JOIN cx c ON a.stay_id = c.stay_id
            WHERE (c.t BETWEEN a.t AND a.t + {abx_then_cx})
               OR (a.t BETWEEN c.t AND c.t + {cx_then_abx})
        ),
        pair_si AS (SELECT stay_id, min(si_time) AS si_pair FROM pairs GROUP BY stay_id)
        SELECT c.stay_id, c.hadm_id,
               af.abx_time, xf.cx_time, ps.si_pair
        FROM cohort c
        LEFT JOIN abx_first af ON af.stay_id = c.stay_id
        LEFT JOIN cx_first  xf ON xf.stay_id = c.stay_id
        LEFT JOIN pair_si   ps ON ps.stay_id = c.stay_id
        """
        df = con.execute(q).df()
    finally:
        con.close()

    has_abx = df["abx_time"].notna()
    has_cx = df["cx_time"].notna()
    has_pair = df["si_pair"].notna()

    if REQUIRE_CULTURE:
        df["si_time"] = df["si_pair"]
    else:
        # antibiotic start; refined earlier by a paired culture when present
        df["si_time"] = df["abx_time"]
        refine = has_pair & (df["si_pair"] < df["si_time"].fillna(df["si_pair"]))
        df.loc[refine, "si_time"] = df.loc[refine, "si_pair"]

    df["has_suspicion"] = df["si_time"].notna().astype(int)
    df["si_hour"] = (df["si_time"] / 60.0).apply(
        lambda x: int(x // 1) if pd.notna(x) else pd.NA).astype("Int64")
    # transparency columns (consumed by the eICU report)
    df["has_antibiotic"] = has_abx.astype(int)
    df["has_culture"] = has_cx.astype(int)
    df["has_strict_pair"] = has_pair.astype(int)

    out_cols = ["stay_id", "hadm_id", "si_time", "si_hour", "has_suspicion",
                "has_antibiotic", "has_culture", "has_strict_pair"]
    df = df[out_cols]
    PATHS.labels_root.mkdir(parents=True, exist_ok=True)
    out = suspicion_path(cfg)
    df.to_parquet(out, index=False)

    n = len(df)
    log.info("  eICU suspicion (primary=%s):",
             "abx+culture pair" if REQUIRE_CULTURE else "antibiotic-based")
    log.info("    has antibiotic : %6d (%.1f%%)", int(has_abx.sum()), 100 * has_abx.mean())
    log.info("    has culture    : %6d (%.1f%%)  <-- SPARSE (eICU microLab)", int(has_cx.sum()), 100 * has_cx.mean())
    log.info("    strict abx+cx pair: %6d (%.1f%%)", int(has_pair.sum()), 100 * has_pair.mean())
    log.info("    => has_suspicion : %6d (%.1f%%)", int(df["has_suspicion"].sum()),
             100 * df["has_suspicion"].mean())
    susp = df[df["has_suspicion"] == 1]
    if len(susp):
        log.info("    median si_hour=%.0f", susp["si_hour"].dropna().median())
    log.info("  wrote %s", out)
    return out


def run(cfg: CohortConfig | None = None, lcfg: LabelConfig | None = None) -> None:
    cfg = cfg or CohortConfig(mode="eicu")
    lcfg = lcfg or LabelConfig.load()
    t0 = time.perf_counter()
    build_suspicion_eicu(cfg, lcfg)
    log.info("eICU suspicion done in %.1fs", time.perf_counter() - t0)
