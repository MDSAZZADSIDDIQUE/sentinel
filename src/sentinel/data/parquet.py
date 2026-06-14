"""Phase 1 event extraction: filter the big MIMIC tables to the cohort + itemids
and write compact Parquet. Everything stays in DuckDB (streamed COPY); the raw
csv.gz are scanned in place and never decompressed or loaded into pandas.

Outputs (data/parquet/, suffixed by cohort mode):
  chartevents_<mode>.parquet   stay_id, charttime, itemid, valuenum, value, valueuom
  labevents_<mode>.parquet     hadm_id, charttime, itemid, valuenum, valueuom
  inputevents_<mode>.parquet   stay_id, starttime, endtime, itemid, rate, ...
  outputevents_<mode>.parquet  stay_id, charttime, itemid, value
  microbiology_<mode>.parquet  hadm_id, chartdate, charttime, spec_type_desc
  antibiotics_<mode>.parquet   hadm_id, starttime, stoptime, drug, route
"""
from __future__ import annotations

import time

import duckdb

from ..config import CohortConfig
from ..duck import connect, mimic
from ..logging_utils import get_logger
from ..paths import PATHS
from .cohort import cohort_path
from .itemids import itemids_by_source, load_itemids

log = get_logger("data.parquet")


def _antibiotic_like_clause(col: str = "drug") -> str:
    from ..config import read_yaml

    names = read_yaml("antibiotics")["names"]
    ors = " OR ".join(f"lower({col}) LIKE '%{n.lower()}%'" for n in names)
    return f"({ors})"


def _register_cohort(con: duckdb.DuckDBPyConnection, cfg: CohortConfig) -> None:
    cp = cohort_path(cfg)
    if not cp.exists():
        raise FileNotFoundError(f"Cohort missing: {cp}. Run build-cohort --mode {cfg.mode} first.")
    con.execute(
        f"CREATE OR REPLACE TEMP TABLE cohort AS "
        f"SELECT stay_id, hadm_id FROM read_parquet('{cp.as_posix()}')"
    )
    n = con.execute("SELECT COUNT(*) FROM cohort").fetchone()[0]
    log.info("cohort registered: %d stays (mode=%s)", n, cfg.mode)


def _copy(con: duckdb.DuckDBPyConnection, select_sql: str, out) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"COPY ({select_sql}) TO '{out.as_posix()}' "
        f"(FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    return con.execute(f"SELECT COUNT(*) FROM read_parquet('{out.as_posix()}')").fetchone()[0]


def run(cfg: CohortConfig | None = None) -> None:
    cfg = cfg or CohortConfig.load()
    data = load_itemids()
    by_src = itemids_by_source(data)
    pr = PATHS.parquet_root

    con = connect()
    try:
        _register_cohort(con, cfg)

        def ids(src: str) -> str:
            return ",".join(str(i) for i in by_src.get(src, []))

        tables: list[tuple[str, str, object]] = [
            # chartevents: keep numeric (vitals/labs/GCS) and categorical (vent/device)
            ("chartevents", f"""
                SELECT ce.stay_id, ce.charttime, ce.itemid, ce.valuenum, ce.value, ce.valueuom
                FROM {mimic('chartevents')} ce
                WHERE ce.itemid IN ({ids('chartevents')})
                  AND ce.stay_id IN (SELECT stay_id FROM cohort)
                  AND (ce.valuenum IS NOT NULL OR ce.value IS NOT NULL)
            """, pr / f"chartevents_{cfg.mode}.parquet"),
            # labevents: hadm-keyed (no stay_id); attribute to stay in labels phase
            ("labevents", f"""
                SELECT le.hadm_id, le.charttime, le.itemid, le.valuenum, le.valueuom
                FROM {mimic('labevents')} le
                WHERE le.itemid IN ({ids('labevents')})
                  AND le.hadm_id IN (SELECT hadm_id FROM cohort)
                  AND le.valuenum IS NOT NULL
            """, pr / f"labevents_{cfg.mode}.parquet"),
            # inputevents: vasopressor infusions
            ("inputevents", f"""
                SELECT ie.stay_id, ie.starttime, ie.endtime, ie.itemid,
                       ie.rate, ie.rateuom, ie.amount, ie.amountuom,
                       ie.patientweight, ie.ordercategoryname, ie.statusdescription
                FROM {mimic('inputevents')} ie
                WHERE ie.itemid IN ({ids('inputevents')})
                  AND ie.stay_id IN (SELECT stay_id FROM cohort)
            """, pr / f"inputevents_{cfg.mode}.parquet"),
            # outputevents: urine
            ("outputevents", f"""
                SELECT oe.stay_id, oe.charttime, oe.itemid, oe.value
                FROM {mimic('outputevents')} oe
                WHERE oe.itemid IN ({ids('outputevents')})
                  AND oe.stay_id IN (SELECT stay_id FROM cohort)
                  AND oe.value IS NOT NULL
            """, pr / f"outputevents_{cfg.mode}.parquet"),
            # microbiology: culture sampling (suspicion of infection)
            ("microbiology", f"""
                SELECT me.hadm_id, me.chartdate, me.charttime, me.spec_type_desc
                FROM {mimic('microbiologyevents')} me
                WHERE me.hadm_id IN (SELECT hadm_id FROM cohort)
            """, pr / f"microbiology_{cfg.mode}.parquet"),
            # antibiotics: prescriptions filtered by drug name (suspicion of infection)
            ("antibiotics", f"""
                SELECT rx.hadm_id, rx.starttime, rx.stoptime, rx.drug, rx.route
                FROM {mimic('prescriptions')} rx
                WHERE rx.hadm_id IN (SELECT hadm_id FROM cohort)
                  AND {_antibiotic_like_clause('rx.drug')}
            """, pr / f"antibiotics_{cfg.mode}.parquet"),
        ]

        for name, sql, out in tables:
            t0 = time.perf_counter()
            n = _copy(con, sql, out)
            dt = time.perf_counter() - t0
            size_mb = out.stat().st_size / 1e6
            log.info("  %-13s -> %-28s %12s rows  %7.1f MB  (%.1fs)",
                     name, out.name, f"{n:,}", size_mb, dt)
    finally:
        con.close()
