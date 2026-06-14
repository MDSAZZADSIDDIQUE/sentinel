"""Hourly SOFA builder.

Strategy: DuckDB does the heavy hour-binned aggregation of the (already
cohort/itemid-filtered) event parquet; pandas assembles the per-stay hourly grid,
forward-fills within clinically sensible limits, derives PaO2/FiO2, GCS total,
ventilation status and 24h urine, then applies the vectorized SOFA scoring.

Output: data/labels/sofa_<mode>.parquet — one row per (stay_id, hr) with the six
component scores, total SOFA, and the key driving values (for the dashboard).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..config import CohortConfig, LabelConfig
from ..duck import connect
from ..logging_utils import get_logger
from ..paths import PATHS
from ..data.itemids import variable_itemids
from . import scoring

log = get_logger("labels.sofa")

# forward-fill limits (hours) by variable class
FFILL_LABS = None        # unbounded within stay (labs persist)
FFILL_VITALS = 24        # MAP, GCS, FiO2 shouldn't persist beyond a day

# o2 devices implying invasive ventilation (approx ventilation status)
INVASIVE_DEVICES = ("endotracheal", "tracheostomy")


def sofa_path(cfg: CohortConfig):
    return PATHS.labels_root / f"sofa_{cfg.mode}.parquet"


def _pq(table: str, cfg: CohortConfig) -> str:
    p = PATHS.parquet_root / f"{table}_{cfg.mode}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run `sentinel to-parquet --mode {cfg.mode}` first.")
    return p.as_posix()


def _register_cohort(con, cfg: CohortConfig) -> pd.DataFrame:
    from ..data.cohort import cohort_path

    cp = cohort_path(cfg).as_posix()
    con.execute(
        f"CREATE OR REPLACE TEMP TABLE cohort AS "
        f"SELECT stay_id, hadm_id, intime, los_hours FROM read_parquet('{cp}')"
    )
    return con.execute("SELECT stay_id, hadm_id, intime, los_hours FROM cohort").df()


def _chart_hourly(con, cfg, items: dict[str, list[int]]) -> pd.DataFrame:
    """Per (stay, hr) min & max for chart SOFA vars + vitals (for features later)."""
    chart_vars = ["map", "fio2", "gcs_eye", "gcs_verbal", "gcs_motor"]
    ids = sorted({i for v in chart_vars for i in items[v]})
    id_list = ",".join(str(i) for i in ids)
    q = f"""
        SELECT ce.stay_id,
               CAST(floor(epoch(ce.charttime - c.intime) / 3600.0) AS INTEGER) AS hr,
               ce.itemid,
               min(ce.valuenum) AS vmin,
               max(ce.valuenum) AS vmax
        FROM read_parquet('{_pq('chartevents', cfg)}') ce
        JOIN cohort c ON c.stay_id = ce.stay_id
        WHERE ce.itemid IN ({id_list}) AND ce.valuenum IS NOT NULL
          AND ce.charttime >= c.intime
          AND epoch(ce.charttime - c.intime)/3600.0 <= c.los_hours + 1
        GROUP BY 1, 2, 3
    """
    return con.execute(q).df()


def _vent_hourly(con, cfg, items: dict[str, list[int]]) -> pd.DataFrame:
    """Hours with a ventilator-mode entry or invasive O2 device (vent status)."""
    vent_ids = ",".join(str(i) for i in items["vent_mode"])
    dev_ids = ",".join(str(i) for i in items["o2_device"])
    cond = " OR ".join(f"lower(value) LIKE '%{d}%'" for d in INVASIVE_DEVICES)
    q = f"""
        SELECT ce.stay_id,
               CAST(floor(epoch(ce.charttime - c.intime)/3600.0) AS INTEGER) AS hr
        FROM read_parquet('{_pq('chartevents', cfg)}') ce
        JOIN cohort c ON c.stay_id = ce.stay_id
        WHERE ce.charttime >= c.intime
          AND ( ce.itemid IN ({vent_ids})
                OR (ce.itemid IN ({dev_ids}) AND ({cond})) )
        GROUP BY 1, 2
    """
    df = con.execute(q).df()
    df["vent"] = 1
    return df


def _lab_hourly(con, cfg, items: dict[str, list[int]]) -> pd.DataFrame:
    lab_vars = ["pao2", "platelets", "bilirubin_total", "creatinine"]
    ids = sorted({i for v in lab_vars for i in items[v]})
    id_list = ",".join(str(i) for i in ids)
    q = f"""
        SELECT c.stay_id,
               CAST(floor(epoch(le.charttime - c.intime)/3600.0) AS INTEGER) AS hr,
               le.itemid,
               min(le.valuenum) AS vmin,
               max(le.valuenum) AS vmax
        FROM read_parquet('{_pq('labevents', cfg)}') le
        JOIN cohort c ON c.hadm_id = le.hadm_id
        WHERE le.valuenum IS NOT NULL AND le.itemid IN ({id_list})
          AND epoch(le.charttime - c.intime)/3600.0 <= c.los_hours + 1
          AND epoch(le.charttime - c.intime)/3600.0 >= -6
        GROUP BY 1, 2, 3
    """
    return con.execute(q).df()


def _urine_hourly(con, cfg) -> pd.DataFrame:
    q = f"""
        SELECT oe.stay_id,
               CAST(floor(epoch(oe.charttime - c.intime)/3600.0) AS INTEGER) AS hr,
               sum(CASE WHEN oe.value >= 0 THEN oe.value ELSE 0 END) AS urine_hr
        FROM read_parquet('{_pq('outputevents', cfg)}') oe
        JOIN cohort c ON c.stay_id = oe.stay_id
        WHERE oe.charttime >= c.intime
          AND epoch(oe.charttime - c.intime)/3600.0 <= c.los_hours + 1
        GROUP BY 1, 2
    """
    return con.execute(q).df()


def _vaso_doses(con, cfg, items: dict[str, list[int]], cohort: pd.DataFrame) -> pd.DataFrame:
    """Per (stay, hr, drug) active norepi-equiv dose, via pandas interval expansion."""
    drugs = {"norepinephrine": items["norepinephrine"], "epinephrine": items["epinephrine"],
             "dopamine": items["dopamine"], "dobutamine": items["dobutamine"]}
    all_ids = sorted({i for v in drugs.values() for i in v})
    id_list = ",".join(str(i) for i in all_ids)
    q = f"""
        SELECT ie.stay_id, ie.itemid, ie.starttime, ie.endtime,
               ie.rate, lower(ie.rateuom) AS rateuom, ie.patientweight
        FROM read_parquet('{_pq('inputevents', cfg)}') ie
        WHERE ie.itemid IN ({id_list}) AND ie.rate IS NOT NULL AND ie.rate > 0
    """
    raw = con.execute(q).df()
    if raw.empty:
        return pd.DataFrame(columns=["stay_id", "hr", "drug", "dose"])

    log.info("  vasopressor rateuom distribution: %s",
             dict(raw["rateuom"].value_counts().head(8)))

    id2drug = {i: d for d, ids in drugs.items() for i in ids}
    raw["drug"] = raw["itemid"].map(id2drug)

    # normalize to µg/kg/min
    def norm(row):
        uom, rate, wt = row["rateuom"], row["rate"], row["patientweight"]
        if not isinstance(uom, str):
            return np.nan
        if "kg" in uom:                       # already per-kg
            return rate
        if "mcg/min" in uom and wt and wt > 0:  # per-minute -> per-kg
            return rate / wt
        if "units/min" in uom:                # vasopressin (not scored here)
            return rate
        return np.nan
    raw["dose"] = raw.apply(norm, axis=1)

    raw = raw.merge(cohort[["stay_id", "intime", "los_hours"]], on="stay_id", how="inner")
    raw["start_hr"] = np.floor((raw["starttime"] - raw["intime"]).dt.total_seconds() / 3600.0)
    raw["end_hr"] = np.ceil((raw["endtime"] - raw["intime"]).dt.total_seconds() / 3600.0)
    raw["start_hr"] = raw["start_hr"].clip(lower=0).astype(int)
    raw["end_hr"] = np.minimum(raw["end_hr"], np.ceil(raw["los_hours"]) + 1).astype(int)
    raw = raw[raw["end_hr"] > raw["start_hr"]]

    # expand each interval to hourly rows
    rows = []
    for r in raw.itertuples(index=False):
        if r.end_hr <= r.start_hr:
            continue
        hrs = np.arange(r.start_hr, r.end_hr)
        rows.append(pd.DataFrame({"stay_id": r.stay_id, "hr": hrs, "drug": r.drug, "dose": r.dose}))
    if not rows:
        return pd.DataFrame(columns=["stay_id", "hr", "drug", "dose"])
    expanded = pd.concat(rows, ignore_index=True)
    # max active dose per (stay, hr, drug)
    return expanded.groupby(["stay_id", "hr", "drug"], as_index=False)["dose"].max()


def _build_grid(cohort: pd.DataFrame) -> pd.DataFrame:
    """Full hourly grid (stay_id, hr) for hr in [0, ceil(los)]."""
    los = np.ceil(cohort["los_hours"].to_numpy()).astype(int) + 1
    los = np.clip(los, 1, None)
    stay_ids = np.repeat(cohort["stay_id"].to_numpy(), los)
    hrs = np.concatenate([np.arange(n) for n in los])
    return pd.DataFrame({"stay_id": stay_ids, "hr": hrs})


def build_hourly_sofa(cfg: CohortConfig, lcfg: LabelConfig) -> "object":
    items = variable_itemids()
    con = connect()
    try:
        cohort = _register_cohort(con, cfg)
        log.info("building hourly SOFA for %d stays (mode=%s)", len(cohort), cfg.mode)
        chart = _chart_hourly(con, cfg, items)
        vent = _vent_hourly(con, cfg, items)
        lab = _lab_hourly(con, cfg, items)
        urine = _urine_hourly(con, cfg)
        vaso = _vaso_doses(con, cfg, items, cohort)
    finally:
        con.close()

    grid = _build_grid(cohort)
    log.info("  hourly grid: %s rows", f"{len(grid):,}")

    # --- map itemids -> variable, choose worst agg, pivot to wide ---
    def pivot(df, var_agg: dict[str, str]):
        """var_agg: {var: 'min'|'max'} -> wide columns on grid."""
        id2var = {}
        for var in var_agg:
            for i in items[var]:
                id2var[i] = var
        df = df[df["itemid"].isin(id2var)].copy()
        df["var"] = df["itemid"].map(id2var)
        out = grid.copy()
        for var, agg in var_agg.items():
            sub = df[df["var"] == var]
            col = sub.groupby(["stay_id", "hr"])["vmin" if agg == "min" else "vmax"].agg(agg)
            out = out.merge(col.rename(var), on=["stay_id", "hr"], how="left")
        return out

    wide = pivot(chart, {"map": "min", "fio2": "max",
                         "gcs_eye": "min", "gcs_verbal": "min", "gcs_motor": "min"})
    labw = pivot(lab, {"pao2": "min", "platelets": "min",
                       "bilirubin_total": "max", "creatinine": "max"})
    for c in ["pao2", "platelets", "bilirubin_total", "creatinine"]:
        wide[c] = labw[c]

    # ventilation flag (1 at observed hours)
    wide = wide.merge(vent[["stay_id", "hr", "vent"]], on=["stay_id", "hr"], how="left")
    # urine hourly
    wide = wide.merge(urine, on=["stay_id", "hr"], how="left")
    # vasopressors -> wide
    if not vaso.empty:
        vw = vaso.pivot_table(index=["stay_id", "hr"], columns="drug", values="dose",
                              aggfunc="max").reset_index()
        wide = wide.merge(vw, on=["stay_id", "hr"], how="left")
    for d in ["norepinephrine", "epinephrine", "dopamine", "dobutamine"]:
        if d not in wide.columns:
            wide[d] = np.nan

    wide = wide.sort_values(["stay_id", "hr"]).reset_index(drop=True)
    g = wide.groupby("stay_id", sort=False)

    # --- forward fill within limits ---
    for c in ["pao2", "platelets", "bilirubin_total", "creatinine"]:  # labs: unbounded
        wide[c] = g[c].ffill()
    for c in ["map", "fio2", "gcs_eye", "gcs_verbal", "gcs_motor"]:    # vitals: 24h
        wide[c] = g[c].ffill(limit=FFILL_VITALS)
    # ventilation: active for 24h after a vent entry
    wide["vent"] = g["vent"].ffill(limit=FFILL_VITALS).fillna(0).astype(int)
    # vasopressors already explicit per active hour -> missing = not running
    for d in ["norepinephrine", "epinephrine", "dopamine", "dobutamine"]:
        wide[d] = wide[d].fillna(0.0)

    # --- derived quantities ---
    # FiO2 may be fraction (0–1) or percent (21–100); normalize to fraction.
    fio2 = wide["fio2"].to_numpy()
    fio2 = np.where(fio2 > 1.0, fio2 / 100.0, fio2)
    fio2 = np.where((fio2 < 0.21) | (fio2 > 1.0), np.nan, fio2)
    pf_ratio = wide["pao2"].to_numpy() / np.where(fio2 > 0, fio2, np.nan)
    gcs_total = wide["gcs_eye"] + wide["gcs_verbal"] + wide["gcs_motor"]
    # 24h urine: rolling sum; require a full day observed (hr >= 24) to apply
    wide["urine_hr"] = wide["urine_hr"].fillna(0.0)
    urine_24h = g["urine_hr"].rolling(24, min_periods=24).sum().reset_index(level=0, drop=True)
    urine_24h = np.where(wide["hr"].to_numpy() >= 24, urine_24h.to_numpy(), np.nan)

    # --- component scores ---
    out = wide[["stay_id", "hr"]].copy()
    out["sofa_respiration"] = scoring.score_respiration(pf_ratio, wide["vent"].to_numpy() == 1)
    out["sofa_coagulation"] = scoring.score_coagulation(wide["platelets"].to_numpy())
    out["sofa_liver"] = scoring.score_liver(wide["bilirubin_total"].to_numpy())
    out["sofa_cardiovascular"] = scoring.score_cardiovascular(
        wide["map"].to_numpy(), wide["dopamine"].to_numpy(), wide["dobutamine"].to_numpy(),
        wide["epinephrine"].to_numpy(), wide["norepinephrine"].to_numpy())
    out["sofa_cns"] = scoring.score_cns(gcs_total.to_numpy())
    out["sofa_renal"] = scoring.score_renal(wide["creatinine"].to_numpy(), urine_24h)
    out["sofa_total"] = out[list(scoring.SOFA_COMPONENTS)].sum(axis=1)

    # carry a few driving values for the dashboard / debugging
    out["map"] = wide["map"]; out["gcs_total"] = gcs_total
    out["pf_ratio"] = pf_ratio; out["creatinine"] = wide["creatinine"]
    out["platelets"] = wide["platelets"]; out["bilirubin_total"] = wide["bilirubin_total"]

    PATHS.labels_root.mkdir(parents=True, exist_ok=True)
    out_path = sofa_path(cfg)
    out.to_parquet(out_path, index=False)
    log.info("  wrote %s (%s rows)", out_path, f"{len(out):,}")
    log.info("  SOFA total: mean=%.2f max=%d | hours with SOFA>=2: %.1f%%",
             out["sofa_total"].mean(), int(out["sofa_total"].max()),
             100 * (out["sofa_total"] >= 2).mean())
    return out_path


def run(cfg: CohortConfig | None = None, lcfg: LabelConfig | None = None) -> None:
    cfg = cfg or CohortConfig.load()
    lcfg = lcfg or LabelConfig.load()
    t0 = time.perf_counter()
    build_hourly_sofa(cfg, lcfg)
    log.info("hourly SOFA done in %.1fs", time.perf_counter() - t0)
