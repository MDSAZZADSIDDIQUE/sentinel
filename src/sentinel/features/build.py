"""Phase 2 hourly feature tensors (organ-grouped, leakage-safe).

Per modeling stay (POSITIVE + CONTROL) we build an hourly matrix over a bounded
observation window and write it to parquet for fast, recompute-free training.

Design (specs D-H):
  * D  episode = last `max_obs_hours` before the episode end; end = onset for
       positives, a length-matched **pseudo-onset** for controls (so episode
       LENGTH can't proxy the label). Observation is strictly < end (no leakage).
       Per-hour label y(t)=1 iff onset within the next `alert_window_hours`.
  * E  each variable carries value + missingness mask + measured-this-hour; the
       measured/frequency channels are toggle-able (FeatureConfig) for the
       with/without ablation.
  * F  FiO2 defaults to room-air 0.21; respiration also gets an S/F surrogate
       (SpO2/FiO2) with SpO2 capped at `spo2_sf_cap` (Rice et al.).
  * normalization: z-score continuous values on TRAIN rows only.

Outputs: data/features/hourly_<mode>.parquet, norm_stats_<mode>.json
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from ..config import CohortConfig, FeatureConfig, LabelConfig
from ..duck import connect
from ..logging_utils import get_logger
from ..paths import PATHS
from ..data.itemids import variable_itemids
from ..labels.sepsis3 import sepsis3_path
from ..labels.sofa import FFILL_HOURS, _build_grid, _register_cohort, _vaso_doses
from .partition import partition_path

log = get_logger("features.build")

# continuous value features by organ system (the per-agent observation groups)
ORGAN_VALUE_FEATURES = {
    "respiratory": ["spo2", "resp_rate", "fio2", "pao2", "pf_ratio", "sf_ratio", "vent"],
    "cardiovascular": ["heart_rate", "sbp", "map", "lactate", "norepinephrine",
                       "epinephrine", "dopamine", "dobutamine", "vasopressin", "phenylephrine"],
    "coagulation": ["platelets"],
    "hepatic": ["bilirubin_total"],
    "renal": ["creatinine", "urine_rate"],
    "neurologic": ["gcs_total", "gcs_eye", "gcs_verbal", "gcs_motor"],
    "shared": ["temp_c", "age"],
}
# variables measured directly from chart/lab (get value+mask+measured channels)
MEASURED_VARS = ["spo2", "resp_rate", "fio2", "pao2", "heart_rate", "sbp", "map",
                 "lactate", "platelets", "bilirubin_total", "creatinine",
                 "gcs_eye", "gcs_verbal", "gcs_motor", "temp_c", "urine_rate",
                 "norepinephrine", "epinephrine", "dopamine", "dobutamine",
                 "vasopressin", "phenylephrine"]
# ffill windows for feature variables not in FFILL_HOURS
EXTRA_FFILL = {"heart_rate": 3, "resp_rate": 3, "spo2": 3, "sbp": 3, "lactate": 12,
               "temp_c": 6, "urine_rate": 6}


def hourly_path(cfg: CohortConfig):
    return PATHS.features_root / f"hourly_{cfg.mode}.parquet"


def _pq(table: str, cfg: CohortConfig) -> str:
    return (PATHS.parquet_root / f"{table}_{cfg.mode}.parquet").as_posix()


def _case_var(items: dict, varlist: list[str]) -> str:
    parts = []
    for v in varlist:
        ids = ",".join(str(i) for i in items[v])
        if ids:
            parts.append(f"WHEN itemid IN ({ids}) THEN '{v}'")
    return "CASE " + " ".join(parts) + " ELSE NULL END"


def _chart_features(con, cfg, items) -> pd.DataFrame:
    chart_vars = ["heart_rate", "resp_rate", "spo2", "sbp", "map", "fio2",
                  "gcs_eye", "gcs_verbal", "gcs_motor", "temp_c", "temp_f"]
    case = _case_var(items, chart_vars)
    allids = ",".join(str(i) for v in chart_vars for i in items[v])
    q = f"""
        SELECT ce.stay_id,
               CAST(floor(epoch(ce.charttime - c.intime)/3600.0) AS INTEGER) AS hr,
               {case} AS var, avg(ce.valuenum) AS val, count(*) AS n
        FROM read_parquet('{_pq('chartevents', cfg)}') ce
        JOIN cohort c ON c.stay_id = ce.stay_id
        WHERE ce.itemid IN ({allids}) AND ce.valuenum IS NOT NULL
          AND ce.charttime >= c.intime
          AND epoch(ce.charttime - c.intime)/3600.0 <= c.los_hours + 1
        GROUP BY 1, 2, 3
    """
    return con.execute(q).df()


def _lab_features(con, cfg, items) -> pd.DataFrame:
    lab_vars = ["pao2", "platelets", "bilirubin_total", "creatinine", "lactate"]
    case = _case_var(items, lab_vars)
    allids = ",".join(str(i) for v in lab_vars for i in items[v])
    q = f"""
        SELECT c.stay_id,
               CAST(floor(epoch(le.charttime - c.intime)/3600.0) AS INTEGER) AS hr,
               {case} AS var, avg(le.valuenum) AS val, count(*) AS n
        FROM read_parquet('{_pq('labevents', cfg)}') le
        JOIN cohort c ON c.hadm_id = le.hadm_id
        WHERE le.itemid IN ({allids}) AND le.valuenum IS NOT NULL
          AND epoch(le.charttime - c.intime)/3600.0 BETWEEN -6 AND c.los_hours + 1
        GROUP BY 1, 2, 3
    """
    return con.execute(q).df()


def _urine_features(con, cfg) -> pd.DataFrame:
    q = f"""
        SELECT oe.stay_id,
               CAST(floor(epoch(oe.charttime - c.intime)/3600.0) AS INTEGER) AS hr,
               sum(CASE WHEN oe.value >= 0 THEN oe.value ELSE 0 END) AS val, count(*) AS n
        FROM read_parquet('{_pq('outputevents', cfg)}') oe
        JOIN cohort c ON c.stay_id = oe.stay_id
        WHERE oe.charttime >= c.intime
          AND epoch(oe.charttime - c.intime)/3600.0 <= c.los_hours + 1
        GROUP BY 1, 2
    """
    return con.execute(q).df()


def _vent_hours(con, cfg, items) -> pd.DataFrame:
    from ..labels.sofa import _vent_hourly
    return _vent_hourly(con, cfg, items)


def build_hourly(cfg: CohortConfig, lcfg: LabelConfig, fcfg: FeatureConfig) -> "object":
    items = variable_itemids()
    con = connect()
    try:
        _register_cohort(con, cfg)
        # restrict to the modeling cohort (POSITIVE + CONTROL) to bound memory
        con.execute(
            f"CREATE OR REPLACE TEMP TABLE cohort AS SELECT c.* FROM cohort c "
            f"JOIN read_parquet('{partition_path(cfg).as_posix()}') p "
            f"ON p.stay_id = c.stay_id WHERE p.label IS NOT NULL"
        )
        cohort = con.execute("SELECT stay_id, hadm_id, intime, los_hours FROM cohort").df()
        log.info("building features for %d modeling stays (mode=%s)", len(cohort), cfg.mode)
        chart = _chart_features(con, cfg, items)
        lab = _lab_features(con, cfg, items)
        urine = _urine_features(con, cfg)
        vent = _vent_hours(con, cfg, items)
        vaso = _vaso_doses(con, cfg, items, cohort)
    finally:
        con.close()

    grid = _build_grid(cohort)

    def pivot_long(df: pd.DataFrame, value="val"):
        val = df.pivot_table(index=["stay_id", "hr"], columns="var", values=value,
                             aggfunc="mean")
        cnt = df.pivot_table(index=["stay_id", "hr"], columns="var", values="n",
                             aggfunc="sum")
        return val, cnt

    cval, ccnt = pivot_long(chart)
    lval, lcnt = pivot_long(lab)
    wide = grid.merge(cval, on=["stay_id", "hr"], how="left")
    wide = wide.merge(lval, on=["stay_id", "hr"], how="left")
    # measured counts
    counts = grid.merge(ccnt, on=["stay_id", "hr"], how="left")
    counts = counts.merge(lcnt, on=["stay_id", "hr"], how="left")
    # urine (rate per hour)
    wide = wide.merge(urine.rename(columns={"val": "urine_rate"})[["stay_id", "hr", "urine_rate"]],
                      on=["stay_id", "hr"], how="left")
    counts = counts.merge(urine.rename(columns={"n": "urine_rate"})[["stay_id", "hr", "urine_rate"]],
                          on=["stay_id", "hr"], how="left")
    # vent flag
    wide = wide.merge(vent[["stay_id", "hr", "vent"]], on=["stay_id", "hr"], how="left")
    # vasopressors -> wide value + measured
    if not vaso.empty:
        vw = vaso.pivot_table(index=["stay_id", "hr"], columns="drug", values="dose", aggfunc="max")
        wide = wide.merge(vw, on=["stay_id", "hr"], how="left")
        counts = counts.merge((vw.notna()).astype(int), on=["stay_id", "hr"], how="left")
    for d in ["norepinephrine", "epinephrine", "dopamine", "dobutamine", "vasopressin", "phenylephrine"]:
        if d not in wide.columns:
            wide[d] = np.nan

    wide = wide.sort_values(["stay_id", "hr"]).reset_index(drop=True)
    counts = counts.sort_values(["stay_id", "hr"]).reset_index(drop=True)
    g = wide.groupby("stay_id", sort=False)

    # measured-this-hour channels (E) from the count tables
    measured = {}
    for v in MEASURED_VARS:
        col = counts[v] if v in counts.columns else pd.Series(np.nan, index=counts.index)
        measured[v] = (col.fillna(0) > 0).astype("int8").to_numpy()

    # merge temperature F->C before ffill
    if "temp_f" in wide.columns:
        tf_c = (wide["temp_f"] - 32.0) * 5.0 / 9.0
        wide["temp_c"] = wide["temp_c"].where(wide["temp_c"].notna(), tf_c)
        measured["temp_c"] = ((counts.get("temp_c", 0).fillna(0)
                               + counts.get("temp_f", 0).fillna(0)) > 0).astype("int8").to_numpy()

    # per-variable forward fill + missingness mask
    ffill_win = {**FFILL_HOURS, **EXTRA_FFILL}
    mask = {}
    for v in MEASURED_VARS:
        if v not in wide.columns:
            wide[v] = np.nan
        wide[v] = g[v].ffill(limit=ffill_win.get(v, 6))
        mask[v] = wide[v].notna().astype("int8").to_numpy()
    # vasopressors: missing = not running -> 0
    for d in ["norepinephrine", "epinephrine", "dopamine", "dobutamine", "vasopressin", "phenylephrine"]:
        wide[d] = wide[d].fillna(0.0)
    wide["vent"] = g["vent"].ffill(limit=FFILL_HOURS["vent"]).fillna(0).astype("int8")
    wide["urine_rate"] = wide["urine_rate"].fillna(0.0)

    # --- derived respiratory features (F) ---
    fio2 = wide["fio2"].to_numpy()
    fio2 = np.where(fio2 > 1.0, fio2 / 100.0, fio2)
    fio2 = np.where((fio2 < 0.21) | (fio2 > 1.0), np.nan, fio2)
    fio2 = np.where(np.isnan(fio2), fcfg.room_air_fio2, fio2)  # room-air default
    wide["pf_ratio"] = wide["pao2"].to_numpy() / fio2
    spo2_capped = np.minimum(wide["spo2"].to_numpy(), fcfg.spo2_sf_cap)
    wide["sf_ratio"] = spo2_capped / fio2
    wide["gcs_total"] = (wide["gcs_eye"].fillna(4) + wide["gcs_verbal"].fillna(5)
                         + wide["gcs_motor"].fillna(6))

    feat = wide[["stay_id", "hr"]].copy()
    value_cols = sorted({v for vs in ORGAN_VALUE_FEATURES.values() for v in vs} - {"age"})
    for v in value_cols:
        feat[v] = wide[v] if v in wide.columns else np.nan
    if fcfg.include_measurement_features:
        for v in MEASURED_VARS:
            feat[f"{v}__measured"] = measured.get(v, 0)
            feat[f"{v}__mask"] = mask.get(v, 0)

    PATHS.features_root.mkdir(parents=True, exist_ok=True)
    return feat, cohort


def run(cfg: CohortConfig | None = None, lcfg: LabelConfig | None = None,
        fcfg: FeatureConfig | None = None) -> None:
    cfg = cfg or CohortConfig.load()
    lcfg = lcfg or LabelConfig.load()
    fcfg = fcfg or FeatureConfig.load()
    t0 = time.perf_counter()
    from .episodes import assemble_and_write

    feat, cohort = build_hourly(cfg, lcfg, fcfg)
    assemble_and_write(cfg, lcfg, fcfg, feat, cohort)
    log.info("features done in %.1fs", time.perf_counter() - t0)
