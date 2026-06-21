"""Hourly SOFA on eICU (Phase 9) — offset-time analog of labels/sofa.py.

Reproduces the MIMIC SOFA derivation EXACTLY (same worst-in-hour aggregation,
same per-variable forward-fill windows, same physiologic range filtering,
ventilation-aware GCS, 24 h urine, and the pure-numpy `scoring.py`) but reads
eICU sources with offset time (minutes from unit admit; hr = floor(offset/60)):

  * MAP      vitalPeriodic.systemicmean  COALESCE  vitalAperiodic.noninvasivemean (min/hr)
  * GCS      nurseCharting 'Glasgow coma score' {Eyes,Verbal,Motor,GCS Total} (min/hr)
  * PaO2/FiO2/platelets/bilirubin/creatinine  lab (labname match; min or max/hr)
  * urine    intakeOutput (volume celllabels only, sum/hr)
  * vaso     infusionDrug (unit parsed from drugname; ug/kg/min; carried forward)
  * vent     respiratoryCare ventstart..ventend intervals

Output: data/labels/sofa_eicu.parquet — IDENTICAL schema to sofa_<mode>.parquet
so labels/sepsis3.py:build_sepsis3 consumes it unchanged.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..config import CohortConfig, LabelConfig
from ..duck import connect, eicu
from ..logging_utils import get_logger
from ..paths import PATHS
from ..data.cohort import load_cohort
from . import scoring
from .sofa import FFILL_HOURS, RANGES, _build_grid, sofa_path

log = get_logger("labels.eicu_sofa")

# vasopressor forward-fill: an eICU infusionDrug row is a point-in-time rate
# (no end), so a recorded dose is carried forward this many hours (titration
# records are frequent while running; bridges gaps without wild over-counting).
VASO_FFILL_HOURS = 4

# drugname -> canonical drug, in PRECEDENCE order (norepinephrine BEFORE
# epinephrine: 'norepinephrine' contains the substring 'epinephrine').
VASO_MATCH = [
    ("norepinephrine", ("norepinephrine", "levophed")),
    ("epinephrine", ("epinephrine", "adrenalin")),
    ("dopamine", ("dopamine",)),
    ("dobutamine", ("dobutamine",)),
]
# effective ug/kg/min for presence-only records (running, dose not derivable);
# combined via max() with any known dose -> conservative SOFA-cardiovascular floor.
VASO_PRESENCE_FLOOR = {"norepinephrine": 0.05, "epinephrine": 0.05,
                       "dopamine": 3.0, "dobutamine": 1.0}


def _register_cohort(con, cfg: CohortConfig) -> pd.DataFrame:
    cohort = load_cohort(cfg)[["stay_id", "los_hours", "weight_kg"]].copy()
    con.register("cohort_df", cohort)
    con.execute("CREATE OR REPLACE TEMP TABLE cohort AS SELECT * FROM cohort_df")
    return cohort


def _map_hourly(con) -> pd.DataFrame:
    """min MAP per (stay, hr): arterial (vitalPeriodic) + NIBP (vitalAperiodic)."""
    q = f"""
        WITH art AS (
            SELECT v.patientunitstayid AS stay_id,
                   CAST(floor(v.observationoffset/60.0) AS INTEGER) AS hr,
                   min(v.systemicmean) AS map_art
            FROM {eicu('vitalPeriodic')} v JOIN cohort c ON c.stay_id = v.patientunitstayid
            WHERE v.systemicmean IS NOT NULL AND v.observationoffset >= 0
              AND v.observationoffset/60.0 <= c.los_hours + 1
            GROUP BY 1, 2
        ), nibp AS (
            SELECT a.patientunitstayid AS stay_id,
                   CAST(floor(a.observationoffset/60.0) AS INTEGER) AS hr,
                   min(a.noninvasivemean) AS map_nibp
            FROM {eicu('vitalAperiodic')} a JOIN cohort c ON c.stay_id = a.patientunitstayid
            WHERE a.noninvasivemean IS NOT NULL AND a.observationoffset >= 0
              AND a.observationoffset/60.0 <= c.los_hours + 1
            GROUP BY 1, 2
        )
        SELECT COALESCE(art.stay_id, nibp.stay_id) AS stay_id,
               COALESCE(art.hr, nibp.hr)           AS hr,
               least(COALESCE(map_art, map_nibp), COALESCE(map_nibp, map_art)) AS map
        FROM art FULL OUTER JOIN nibp ON art.stay_id = nibp.stay_id AND art.hr = nibp.hr
    """
    return con.execute(q).df()


def _gcs_hourly(con) -> pd.DataFrame:
    """min GCS components + charted total per (stay, hr) from nurseCharting."""
    q = f"""
        SELECT n.patientunitstayid AS stay_id,
               CAST(floor(n.nursingchartoffset/60.0) AS INTEGER) AS hr,
               min(CASE WHEN n.nursingchartcelltypevalname='Eyes'
                        THEN TRY_CAST(n.nursingchartvalue AS DOUBLE) END) AS gcs_eye,
               min(CASE WHEN n.nursingchartcelltypevalname='Verbal'
                        THEN TRY_CAST(n.nursingchartvalue AS DOUBLE) END) AS gcs_verbal,
               min(CASE WHEN n.nursingchartcelltypevalname='Motor'
                        THEN TRY_CAST(n.nursingchartvalue AS DOUBLE) END) AS gcs_motor,
               min(CASE WHEN n.nursingchartcelltypevalname='GCS Total'
                        THEN TRY_CAST(n.nursingchartvalue AS DOUBLE) END) AS gcs_total_charted
        FROM {eicu('nurseCharting')} n JOIN cohort c ON c.stay_id = n.patientunitstayid
        WHERE n.nursingchartcelltypevallabel='Glasgow coma score'
          AND n.nursingchartoffset >= 0 AND n.nursingchartoffset/60.0 <= c.los_hours + 1
        GROUP BY 1, 2
    """
    return con.execute(q).df()


def _lab_hourly(con) -> pd.DataFrame:
    """SOFA lab analytes per (stay, hr): worst-in-hour (min or max)."""
    q = f"""
        SELECT l.patientunitstayid AS stay_id,
               CAST(floor(l.labresultoffset/60.0) AS INTEGER) AS hr,
               min(CASE WHEN lower(l.labname)='pao2'             THEN l.labresult END) AS pao2,
               max(CASE WHEN lower(l.labname)='fio2'             THEN l.labresult END) AS fio2,
               min(CASE WHEN lower(l.labname)='platelets x 1000' THEN l.labresult END) AS platelets,
               max(CASE WHEN lower(l.labname)='total bilirubin'  THEN l.labresult END) AS bilirubin_total,
               max(CASE WHEN lower(l.labname)='creatinine'       THEN l.labresult END) AS creatinine
        FROM {eicu('lab')} l JOIN cohort c ON c.stay_id = l.patientunitstayid
        WHERE lower(l.labname) IN ('pao2','fio2','platelets x 1000','total bilirubin','creatinine')
          AND l.labresult IS NOT NULL
          AND l.labresultoffset/60.0 BETWEEN -6 AND c.los_hours + 1
        GROUP BY 1, 2
    """
    return con.execute(q).df()


def _urine_hourly(con) -> pd.DataFrame:
    """Sum urine volume per (stay, hr) from intakeOutput (volume labels only)."""
    q = f"""
        SELECT io.patientunitstayid AS stay_id,
               CAST(floor(io.intakeoutputoffset/60.0) AS INTEGER) AS hr,
               sum(CASE WHEN io.cellvaluenumeric >= 0 THEN io.cellvaluenumeric ELSE 0 END) AS urine_hr
        FROM {eicu('intakeOutput')} io JOIN cohort c ON c.stay_id = io.patientunitstayid
        WHERE io.intakeoutputoffset >= 0 AND io.intakeoutputoffset/60.0 <= c.los_hours + 1
          AND (lower(io.celllabel) LIKE '%urine%' OR lower(io.celllabel) LIKE '%foley%'
               OR lower(io.celllabel) LIKE '%void%')
          AND lower(io.celllabel) NOT LIKE '%count%'
          AND lower(io.celllabel) NOT LIKE '%occurrence%'
          AND lower(io.celllabel) NOT LIKE '%incontinen%'
          AND lower(io.celllabel) NOT LIKE '%number of%'
          AND lower(io.celllabel) NOT LIKE '%unmeasured%'
        GROUP BY 1, 2
    """
    return con.execute(q).df()


def _vent_hourly(con) -> pd.DataFrame:
    """Ventilated hours from respiratoryCare ventstart..ventend intervals.

    eICU records a vent episode's ventstartoffset and ventendoffset (0/ongoing
    -> use respcarestatusoffset, the observed-active time, as a conservative end).
    Expand each interval to hourly rows; ffill (12 h) happens in assembly.
    """
    q = f"""
        SELECT r.patientunitstayid AS stay_id, r.ventstartoffset AS start_min,
               greatest(r.ventendoffset, r.respcarestatusoffset) AS end_min,
               c.los_hours
        FROM {eicu('respiratoryCare')} r JOIN cohort c ON c.stay_id = r.patientunitstayid
        WHERE r.ventstartoffset IS NOT NULL
    """
    raw = con.execute(q).df()
    if raw.empty:
        return pd.DataFrame(columns=["stay_id", "hr", "vent"])
    raw = raw[raw["end_min"] > raw["start_min"]]
    rows = []
    for r in raw.itertuples(index=False):
        start_hr = max(int(np.floor(r.start_min / 60.0)), 0)
        end_hr = min(int(np.ceil(r.end_min / 60.0)), int(np.ceil(r.los_hours)) + 1)
        if end_hr > start_hr:
            rows.append(pd.DataFrame({"stay_id": r.stay_id, "hr": np.arange(start_hr, end_hr)}))
    if not rows:
        return pd.DataFrame(columns=["stay_id", "hr", "vent"])
    out = pd.concat(rows, ignore_index=True).drop_duplicates(["stay_id", "hr"])
    out["vent"] = 1
    return out


def _vaso_doses(con, cohort: pd.DataFrame) -> pd.DataFrame:
    """Per (stay, hr, drug) effective ug/kg/min for the 4 SOFA-scored pressors."""
    like = " OR ".join(
        f"lower(drugname) LIKE '%{p}%'" for _, pats in VASO_MATCH for p in pats)
    q = f"""
        SELECT i.patientunitstayid AS stay_id, i.infusionoffset AS off_min,
               i.drugname,
               COALESCE(TRY_CAST(i.drugrate AS DOUBLE), i.infusionrate) AS rate,
               i.patientweight
        FROM {eicu('infusionDrug')} i JOIN cohort c ON c.stay_id = i.patientunitstayid
        WHERE i.drugname IS NOT NULL AND ({like})
    """
    raw = con.execute(q).df()
    if raw.empty:
        return pd.DataFrame(columns=["stay_id", "hr", "drug", "dose"])

    name = raw["drugname"].str.lower()

    def assign(n: str) -> str | None:
        for drug, pats in VASO_MATCH:
            if any(p in n for p in pats):
                return drug
        return None
    raw["drug"] = name.map(assign)
    raw = raw[raw["drug"].notna()]
    # rate must be > 0 to count as infusing (drops charted-but-off lines)
    raw = raw[raw["rate"].notna() & (raw["rate"] > 0)]
    if raw.empty:
        return pd.DataFrame(columns=["stay_id", "hr", "drug", "dose"])

    wt = cohort.set_index("stay_id")["weight_kg"]
    raw = raw.merge(wt.rename("cohort_wt"), left_on="stay_id", right_index=True, how="left")
    weight = raw["patientweight"].where(raw["patientweight"] > 0, raw["cohort_wt"])

    is_per_kg = name.loc[raw.index].str.contains("mcg/kg/min", regex=False)
    is_per_min = name.loc[raw.index].str.contains("mcg/min", regex=False)
    dose = np.full(len(raw), np.nan)
    dose = np.where(is_per_kg, raw["rate"], dose)
    dose = np.where(is_per_min & (weight > 0), raw["rate"] / weight, dose)
    # presence-only (volumetric/unitless): conservative per-drug floor
    floor = raw["drug"].map(VASO_PRESENCE_FLOOR).to_numpy()
    raw["dose"] = np.where(np.isnan(dose), floor, dose)
    raw["hr"] = np.floor(raw["off_min"] / 60.0).clip(lower=0).astype(int)

    # max effective dose per (stay, hr, drug)
    return raw.groupby(["stay_id", "hr", "drug"], as_index=False)["dose"].max()


def build_hourly_sofa_eicu(cfg: CohortConfig, lcfg: LabelConfig) -> "object":
    cohort = load_cohort(cfg)
    con = connect()
    try:
        _register_cohort(con, cfg)
        log.info("building eICU hourly SOFA for %d stays", len(cohort))
        mp = _map_hourly(con)
        gcs = _gcs_hourly(con)
        lab = _lab_hourly(con)
        urine = _urine_hourly(con)
        vent = _vent_hourly(con)
        vaso = _vaso_doses(con, cohort[["stay_id", "weight_kg"]])
    finally:
        con.close()

    grid = _build_grid(cohort)
    log.info("  hourly grid: %s rows", f"{len(grid):,}")

    wide = grid.merge(mp, on=["stay_id", "hr"], how="left")
    wide = wide.merge(gcs, on=["stay_id", "hr"], how="left")
    wide = wide.merge(lab, on=["stay_id", "hr"], how="left")
    wide = wide.merge(vent, on=["stay_id", "hr"], how="left")
    wide = wide.merge(urine, on=["stay_id", "hr"], how="left")
    if not vaso.empty:
        vw = vaso.pivot_table(index=["stay_id", "hr"], columns="drug", values="dose",
                              aggfunc="max").reset_index()
        wide = wide.merge(vw, on=["stay_id", "hr"], how="left")
    for d in ["norepinephrine", "epinephrine", "dopamine", "dobutamine"]:
        if d not in wide.columns:
            wide[d] = np.nan

    wide = wide.sort_values(["stay_id", "hr"]).reset_index(drop=True)

    # drop physiologically implausible values before carrying forward (artifacts).
    # NB: use numpy on an independent copied array per column. The pandas
    # `wide[c] = wide[c].where(mask)` pattern (as in MIMIC sofa.py) corrupts
    # ADJACENT float columns on this wide frame under pandas 2.3.3 — a
    # consolidated-block aliasing bug that non-deterministically zeroed/NaN'd
    # platelets across the whole frame. Operating on a fresh array sidesteps it.
    for c, (lo, hi) in RANGES.items():
        if c in wide.columns:
            v = wide[c].to_numpy(dtype="float64", copy=True)
            v[(v < lo) | (v > hi)] = np.nan
            wide[c] = v

    g = wide.groupby("stay_id", sort=False)
    for c in ["pao2", "platelets", "bilirubin_total", "creatinine",
              "map", "fio2", "gcs_eye", "gcs_verbal", "gcs_motor"]:
        wide[c] = g[c].ffill(limit=FFILL_HOURS[c])
    wide["vent"] = g["vent"].ffill(limit=FFILL_HOURS["vent"]).fillna(0).astype(int)
    for d in ["norepinephrine", "epinephrine", "dopamine", "dobutamine"]:
        wide[d] = g[d].ffill(limit=VASO_FFILL_HOURS).fillna(0.0)

    # --- derived quantities (identical to MIMIC) ---
    fio2 = wide["fio2"].to_numpy()
    fio2 = np.where(fio2 > 1.0, fio2 / 100.0, fio2)
    fio2 = np.where((fio2 < 0.21) | (fio2 > 1.0), np.nan, fio2)
    pf_ratio = wide["pao2"].to_numpy() / np.where(fio2 > 0, fio2, np.nan)
    # ventilation-aware GCS: intubated verbal=1T restored to normal
    verbal = wide["gcs_verbal"].to_numpy(dtype="float64")
    if lcfg.gcs_vent_adjust:
        vent_arr = wide["vent"].to_numpy() == 1
        verbal = np.where(vent_arr & (verbal <= 1), 5.0, verbal)
    gcs_total = (wide["gcs_eye"].to_numpy(dtype="float64") + verbal
                 + wide["gcs_motor"].to_numpy(dtype="float64"))
    # fall back to charted 'GCS Total' where components are incomplete
    charted = wide["gcs_total_charted"].to_numpy(dtype="float64")
    gcs_total = np.where(np.isnan(gcs_total) & ~np.isnan(charted), charted, gcs_total)
    # 24h urine: rolling sum; require a full observed day (hr >= 24)
    wide["urine_hr"] = wide["urine_hr"].fillna(0.0)
    urine_24h = g["urine_hr"].rolling(24, min_periods=24).sum().reset_index(level=0, drop=True)
    urine_24h = np.where(wide["hr"].to_numpy() >= 24, urine_24h.to_numpy(), np.nan)

    # --- component scores (same pure-numpy scoring as MIMIC) ---
    out = wide[["stay_id", "hr"]].copy()
    out["sofa_respiration"] = scoring.score_respiration(pf_ratio, wide["vent"].to_numpy() == 1)
    out["sofa_coagulation"] = scoring.score_coagulation(wide["platelets"].to_numpy())
    out["sofa_liver"] = scoring.score_liver(wide["bilirubin_total"].to_numpy())
    out["sofa_cardiovascular"] = scoring.score_cardiovascular(
        wide["map"].to_numpy(), wide["dopamine"].to_numpy(), wide["dobutamine"].to_numpy(),
        wide["epinephrine"].to_numpy(), wide["norepinephrine"].to_numpy())
    out["sofa_cns"] = scoring.score_cns(gcs_total)
    out["sofa_renal"] = scoring.score_renal(wide["creatinine"].to_numpy(), urine_24h)
    out["sofa_total"] = out[list(scoring.SOFA_COMPONENTS)].sum(axis=1)

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
    for c in scoring.SOFA_COMPONENTS:
        log.info("    %-22s mean=%.2f  %%hrs>0=%.1f%%",
                 c, out[c].mean(), 100 * (out[c] > 0).mean())
    return out_path


def run(cfg: CohortConfig | None = None, lcfg: LabelConfig | None = None) -> None:
    cfg = cfg or CohortConfig(mode="eicu")
    lcfg = lcfg or LabelConfig.load()
    t0 = time.perf_counter()
    build_hourly_sofa_eicu(cfg, lcfg)
    log.info("eICU hourly SOFA done in %.1fs", time.perf_counter() - t0)
