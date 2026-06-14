"""Suspicion of infection (Sepsis-3 component).

Seymour 2016 / mimic-code rule: a qualifying antibiotic + culture pair where
  * antibiotic first  -> culture within +72h, or
  * culture first     -> antibiotic within +24h.
The suspicion time is the earlier event of the qualifying pair; we take the
earliest such time per stay. Times are expressed as hours since ICU intime
(can be negative if suspicion began before ICU admission).

Output: data/labels/suspicion_<mode>.parquet — one row per cohort stay with
si_time, si_hour, has_suspicion.
"""
from __future__ import annotations

import time

import pandas as pd

from ..config import CohortConfig, LabelConfig, read_yaml
from ..duck import connect
from ..logging_utils import get_logger
from ..paths import PATHS

log = get_logger("labels.suspicion")


def suspicion_path(cfg: CohortConfig):
    return PATHS.labels_root / f"suspicion_{cfg.mode}.parquet"


def _pq(table: str, cfg: CohortConfig) -> str:
    p = PATHS.parquet_root / f"{table}_{cfg.mode}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run to-parquet first.")
    return p.as_posix()


def build_suspicion(cfg: CohortConfig, lcfg: LabelConfig) -> "object":
    from ..data.cohort import cohort_path

    abx_cfg = read_yaml("antibiotics")
    excl = abx_cfg.get("exclude_routes", [])
    excl_clause = ""
    if excl:
        vals = ",".join(f"'{r}'" for r in excl)
        excl_clause = f"AND (route IS NULL OR upper(route) NOT IN ({vals}))"

    con = connect()
    try:
        cp = cohort_path(cfg).as_posix()
        con.execute(
            f"CREATE OR REPLACE TEMP TABLE cohort AS "
            f"SELECT stay_id, hadm_id, intime FROM read_parquet('{cp}')"
        )
        q = f"""
        WITH abx AS (
            SELECT DISTINCT hadm_id, starttime AS t
            FROM read_parquet('{_pq('antibiotics', cfg)}')
            WHERE starttime IS NOT NULL {excl_clause}
        ),
        cx AS (
            SELECT DISTINCT hadm_id, COALESCE(charttime, chartdate) AS t
            FROM read_parquet('{_pq('microbiology', cfg)}')
            WHERE COALESCE(charttime, chartdate) IS NOT NULL
        ),
        pairs AS (
            SELECT a.hadm_id, least(a.t, c.t) AS si_time
            FROM abx a JOIN cx c ON a.hadm_id = c.hadm_id
            WHERE (c.t BETWEEN a.t AND a.t + INTERVAL {lcfg.si_abx_then_culture_hours} HOUR)
               OR (a.t BETWEEN c.t AND c.t + INTERVAL {lcfg.si_culture_then_abx_hours} HOUR)
        ),
        si AS (
            SELECT hadm_id, min(si_time) AS si_time FROM pairs GROUP BY hadm_id
        )
        SELECT c.stay_id, c.hadm_id, si.si_time,
               CAST(floor(epoch(si.si_time - c.intime)/3600.0) AS INTEGER) AS si_hour
        FROM cohort c LEFT JOIN si ON si.hadm_id = c.hadm_id
        """
        df = con.execute(q).df()
    finally:
        con.close()

    df["has_suspicion"] = df["si_time"].notna().astype(int)
    PATHS.labels_root.mkdir(parents=True, exist_ok=True)
    out = suspicion_path(cfg)
    df.to_parquet(out, index=False)
    log.info("  suspicion of infection: %d/%d stays (%.1f%%); median si_hour=%s",
             int(df["has_suspicion"].sum()), len(df),
             100 * df["has_suspicion"].mean(),
             f'{df.loc[df.has_suspicion == 1, "si_hour"].median():.0f}' if df["has_suspicion"].any() else "n/a")
    log.info("  wrote %s", out)
    return out


def run(cfg: CohortConfig | None = None, lcfg: LabelConfig | None = None) -> None:
    cfg = cfg or CohortConfig.load()
    lcfg = lcfg or LabelConfig.load()
    t0 = time.perf_counter()
    build_suspicion(cfg, lcfg)
    log.info("suspicion done in %.1fs", time.perf_counter() - t0)
