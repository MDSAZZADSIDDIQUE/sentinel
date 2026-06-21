"""Harmonized eICU hourly features (Phase 9, Step 4) — offset-time analog of
features/build.py + an episode assembler that applies the FROZEN MIMIC normalizer.

Produces data/features/hourly_eicu.parquet with the EXACT MIMIC feature schema
(identical organ grouping, column names + order = feature_manifest_full.json) so
the frozen MIMIC-trained models score it through features/dataset.py:load_split
with ZERO model-code changes. Normalization uses MIMIC's saved train-split stats
(norm_stats_full.json) — NEVER refit on eICU. The MIMIC manifest + normalizer are
copied verbatim to feature_manifest_eicu.json / norm_stats_eicu.json (asserted
equal by tests/test_eicu_harmonization.py).

Aggregation mirrors build.py: per (stay, hr) mean value + a measured-this-hour
count; bounded forward-fill per variable; missingness mask; room-air FiO2 +
S/F surrogate; vasopressor doses (presence floor where the eICU unit is
non-derivable). All stays are the external split.
"""
from __future__ import annotations

import json
import shutil
import time

import numpy as np
import pandas as pd

from ..config import CohortConfig, FeatureConfig, LabelConfig
from ..constants import SHARED_GROUP
from ..duck import DuckSettings, connect, eicu
from ..logging_utils import get_logger
from ..paths import PATHS
from ..data.cohort import load_cohort
from ..labels.sepsis3 import sepsis3_path
from ..labels.sofa import FFILL_HOURS, _build_grid
from ..labels.eicu_sofa import _vent_hourly
from .build import EXTRA_FFILL, MEASURED_VARS, ORGAN_VALUE_FEATURES, hourly_path
from .episodes import CONT_COLS, VALUE_COLS, _episode_ends
from .partition import partition_path

log = get_logger("features.eicu_build")

VASO_DRUGS = ["norepinephrine", "epinephrine", "dopamine", "dobutamine",
              "vasopressin", "phenylephrine"]
# drugname substring -> canonical (precedence: norepinephrine BEFORE epinephrine)
VASO_MATCH = [
    ("norepinephrine", ("norepinephrine", "levophed")),
    ("epinephrine", ("epinephrine", "adrenalin")),
    ("dopamine", ("dopamine",)),
    ("dobutamine", ("dobutamine",)),
    ("phenylephrine", ("phenylephrine", "neosynephrine")),
    ("vasopressin", ("vasopressin",)),
]
VASO_PRESENCE_FLOOR = {"norepinephrine": 0.05, "epinephrine": 0.05, "dopamine": 3.0,
                       "dobutamine": 1.0, "phenylephrine": 0.5, "vasopressin": 0.03}


def _register_modeling_cohort(con, cfg: CohortConfig) -> pd.DataFrame:
    """Modeling cohort (POSITIVE + CONTROL) registered as temp table `cohort`."""
    cohort = load_cohort(cfg)[["stay_id", "los_hours", "weight_kg"]]
    part = pd.read_parquet(partition_path(cfg))
    keep = part.loc[part["label"].notna(), "stay_id"]
    cohort = cohort[cohort["stay_id"].isin(keep)].copy()
    con.register("cohort_df", cohort)
    con.execute("CREATE OR REPLACE TEMP TABLE cohort AS SELECT * FROM cohort_df")
    return cohort


def _vitals_hourly(con) -> pd.DataFrame:
    q = f"""
        SELECT v.patientunitstayid AS stay_id,
               CAST(floor(v.observationoffset/60.0) AS INTEGER) AS hr,
               avg(v.heartrate) AS heart_rate, count(v.heartrate) AS heart_rate__n,
               avg(v.respiration) AS resp_rate, count(v.respiration) AS resp_rate__n,
               avg(v.sao2) AS spo2, count(v.sao2) AS spo2__n,
               avg(CASE WHEN v.temperature BETWEEN 25 AND 46 THEN v.temperature END) AS temp_c,
               count(CASE WHEN v.temperature BETWEEN 25 AND 46 THEN 1 END) AS temp_c__n,
               avg(v.systemicmean) AS map_art, count(v.systemicmean) AS map_art__n,
               avg(v.systemicsystolic) AS sbp_art, count(v.systemicsystolic) AS sbp_art__n
        FROM {eicu('vitalPeriodic')} v JOIN cohort c ON c.stay_id = v.patientunitstayid
        WHERE v.observationoffset >= 0 AND v.observationoffset/60.0 <= c.los_hours + 1
        GROUP BY 1, 2
    """
    return con.execute(q).df()


def _nibp_hourly(con) -> pd.DataFrame:
    q = f"""
        SELECT a.patientunitstayid AS stay_id,
               CAST(floor(a.observationoffset/60.0) AS INTEGER) AS hr,
               avg(a.noninvasivemean) AS map_nibp, count(a.noninvasivemean) AS map_nibp__n,
               avg(a.noninvasivesystolic) AS sbp_nibp, count(a.noninvasivesystolic) AS sbp_nibp__n
        FROM {eicu('vitalAperiodic')} a JOIN cohort c ON c.stay_id = a.patientunitstayid
        WHERE a.observationoffset >= 0 AND a.observationoffset/60.0 <= c.los_hours + 1
        GROUP BY 1, 2
    """
    return con.execute(q).df()


def _gcs_feat(con) -> pd.DataFrame:
    q = f"""
        SELECT n.patientunitstayid AS stay_id,
               CAST(floor(n.nursingchartoffset/60.0) AS INTEGER) AS hr,
               avg(CASE WHEN n.nursingchartcelltypevalname='Eyes'   THEN TRY_CAST(n.nursingchartvalue AS DOUBLE) END) AS gcs_eye,
               count(CASE WHEN n.nursingchartcelltypevalname='Eyes' THEN 1 END) AS gcs_eye__n,
               avg(CASE WHEN n.nursingchartcelltypevalname='Verbal'   THEN TRY_CAST(n.nursingchartvalue AS DOUBLE) END) AS gcs_verbal,
               count(CASE WHEN n.nursingchartcelltypevalname='Verbal' THEN 1 END) AS gcs_verbal__n,
               avg(CASE WHEN n.nursingchartcelltypevalname='Motor'   THEN TRY_CAST(n.nursingchartvalue AS DOUBLE) END) AS gcs_motor,
               count(CASE WHEN n.nursingchartcelltypevalname='Motor' THEN 1 END) AS gcs_motor__n
        FROM {eicu('nurseCharting')} n JOIN cohort c ON c.stay_id = n.patientunitstayid
        WHERE n.nursingchartcelltypevallabel='Glasgow coma score'
          AND n.nursingchartoffset >= 0 AND n.nursingchartoffset/60.0 <= c.los_hours + 1
        GROUP BY 1, 2
    """
    return con.execute(q).df()


def _lab_feat(con) -> pd.DataFrame:
    q = f"""
        SELECT l.patientunitstayid AS stay_id,
               CAST(floor(l.labresultoffset/60.0) AS INTEGER) AS hr,
               avg(CASE WHEN lower(l.labname)='pao2' THEN l.labresult END) AS pao2,
               count(CASE WHEN lower(l.labname)='pao2' THEN 1 END) AS pao2__n,
               avg(CASE WHEN lower(l.labname)='fio2' THEN l.labresult END) AS fio2,
               count(CASE WHEN lower(l.labname)='fio2' THEN 1 END) AS fio2__n,
               avg(CASE WHEN lower(l.labname)='platelets x 1000' THEN l.labresult END) AS platelets,
               count(CASE WHEN lower(l.labname)='platelets x 1000' THEN 1 END) AS platelets__n,
               avg(CASE WHEN lower(l.labname)='total bilirubin' THEN l.labresult END) AS bilirubin_total,
               count(CASE WHEN lower(l.labname)='total bilirubin' THEN 1 END) AS bilirubin_total__n,
               avg(CASE WHEN lower(l.labname)='creatinine' THEN l.labresult END) AS creatinine,
               count(CASE WHEN lower(l.labname)='creatinine' THEN 1 END) AS creatinine__n,
               avg(CASE WHEN lower(l.labname)='lactate' THEN l.labresult END) AS lactate,
               count(CASE WHEN lower(l.labname)='lactate' THEN 1 END) AS lactate__n
        FROM {eicu('lab')} l JOIN cohort c ON c.stay_id = l.patientunitstayid
        WHERE lower(l.labname) IN ('pao2','fio2','platelets x 1000','total bilirubin','creatinine','lactate')
          AND l.labresult IS NOT NULL
          AND l.labresultoffset/60.0 BETWEEN -6 AND c.los_hours + 1
        GROUP BY 1, 2
    """
    return con.execute(q).df()


def _urine_feat(con) -> pd.DataFrame:
    q = f"""
        SELECT io.patientunitstayid AS stay_id,
               CAST(floor(io.intakeoutputoffset/60.0) AS INTEGER) AS hr,
               sum(CASE WHEN io.cellvaluenumeric >= 0 THEN io.cellvaluenumeric ELSE 0 END) AS urine_rate,
               count(*) AS urine_rate__n
        FROM {eicu('intakeOutput')} io JOIN cohort c ON c.stay_id = io.patientunitstayid
        WHERE io.intakeoutputoffset >= 0 AND io.intakeoutputoffset/60.0 <= c.los_hours + 1
          AND (lower(io.celllabel) LIKE '%urine%' OR lower(io.celllabel) LIKE '%foley%'
               OR lower(io.celllabel) LIKE '%void%')
          AND lower(io.celllabel) NOT LIKE '%count%' AND lower(io.celllabel) NOT LIKE '%occurrence%'
          AND lower(io.celllabel) NOT LIKE '%incontinen%' AND lower(io.celllabel) NOT LIKE '%number of%'
          AND lower(io.celllabel) NOT LIKE '%unmeasured%'
        GROUP BY 1, 2
    """
    return con.execute(q).df()


def _vaso_feat(con, cohort: pd.DataFrame) -> pd.DataFrame:
    """Per (stay, hr, drug) effective dose for all six pressors (feature channels)."""
    like = " OR ".join(f"lower(drugname) LIKE '%{p}%'" for _, pats in VASO_MATCH for p in pats)
    q = f"""
        SELECT i.patientunitstayid AS stay_id, i.infusionoffset AS off_min, i.drugname,
               COALESCE(TRY_CAST(i.drugrate AS DOUBLE), i.infusionrate) AS rate, i.patientweight
        FROM {eicu('infusionDrug')} i JOIN cohort c ON c.stay_id = i.patientunitstayid
        WHERE i.drugname IS NOT NULL AND ({like})
    """
    raw = con.execute(q).df()
    if raw.empty:
        return pd.DataFrame(columns=["stay_id", "hr", "drug", "dose"])
    name = raw["drugname"].str.lower()

    def assign(n: str):
        for drug, pats in VASO_MATCH:
            if any(p in n for p in pats):
                return drug
        return None
    raw["drug"] = name.map(assign)
    raw = raw[raw["drug"].notna() & raw["rate"].notna() & (raw["rate"] > 0)]
    if raw.empty:
        return pd.DataFrame(columns=["stay_id", "hr", "drug", "dose"])
    wt = cohort.set_index("stay_id")["weight_kg"]
    raw = raw.merge(wt.rename("cohort_wt"), left_on="stay_id", right_index=True, how="left")
    weight = raw["patientweight"].where(raw["patientweight"] > 0, raw["cohort_wt"])
    nm = name.loc[raw.index]
    is_per_kg = nm.str.contains("mcg/kg/min", regex=False)
    is_per_min = nm.str.contains("mcg/min", regex=False)
    is_units = nm.str.contains("units/", regex=False)
    dose = np.full(len(raw), np.nan)
    dose = np.where(is_per_kg, raw["rate"], dose)
    dose = np.where(is_per_min & (weight > 0), raw["rate"] / weight, dose)
    dose = np.where(is_units, raw["rate"], dose)           # vasopressin units/min
    floor = raw["drug"].map(VASO_PRESENCE_FLOOR).to_numpy()
    raw["dose"] = np.where(np.isnan(dose), floor, dose)
    raw["hr"] = np.floor(raw["off_min"] / 60.0).clip(lower=0).astype(int)
    return raw.groupby(["stay_id", "hr", "drug"], as_index=False)["dose"].max()


def build_hourly_eicu(cfg: CohortConfig, lcfg: LabelConfig, fcfg: FeatureConfig):
    import gc

    # 6 GB DuckDB cap (vs 8) leaves headroom for the big pandas assembly on this
    # 13.9 GB box — the 8 GB default OOM'd consolidating the wide frame.
    con = connect(DuckSettings(memory_limit="6GB"))
    try:
        cohort = _register_modeling_cohort(con, cfg)
        log.info("building eICU features for %d modeling stays", len(cohort))
        vitals = _vitals_hourly(con)
        nibp = _nibp_hourly(con)
        gcs = _gcs_feat(con)
        lab = _lab_feat(con)
        urine = _urine_feat(con)
        vent = _vent_hourly(con)
        vaso = _vaso_feat(con, cohort[["stay_id", "weight_kg"]])
    finally:
        con.close()
    del con
    gc.collect()

    grid = _build_grid(cohort)
    wide = grid

    def _merge_free(w, df, val_cols):
        keep = ["stay_id", "hr"] + list(val_cols) + [f"{c}__n" for c in val_cols if f"{c}__n" in df.columns]
        out = w.merge(df[keep], on=["stay_id", "hr"], how="left")
        del df
        gc.collect()
        return out

    # merge each loader then free it immediately (peak = wide + one loader, not all)
    wide = _merge_free(wide, vitals, ["heart_rate", "resp_rate", "spo2", "temp_c", "map_art", "sbp_art"]); del vitals
    wide = _merge_free(wide, nibp, ["map_nibp", "sbp_nibp"]); del nibp
    wide = _merge_free(wide, gcs, ["gcs_eye", "gcs_verbal", "gcs_motor"]); del gcs
    wide = _merge_free(wide, lab, ["pao2", "fio2", "platelets", "bilirubin_total", "creatinine", "lactate"]); del lab
    wide = wide.merge(urine[["stay_id", "hr", "urine_rate", "urine_rate__n"]], on=["stay_id", "hr"], how="left")
    del urine
    wide = wide.merge(vent[["stay_id", "hr", "vent"]], on=["stay_id", "hr"], how="left")
    del vent
    gc.collect()

    # coalesce arterial + NIBP -> map, sbp (prefer arterial; count summed). Use
    # numpy on independent arrays (avoid the pandas 2.3.3 Series.where block bug).
    for out_c, art, nibp_c in [("map", "map_art", "map_nibp"), ("sbp", "sbp_art", "sbp_nibp")]:
        a = wide[art].to_numpy(dtype="float64")
        b = wide[nibp_c].to_numpy(dtype="float64")
        wide[out_c] = np.where(np.isnan(a), b, a)
        wide[f"{out_c}__n"] = wide[[f"{art}__n", f"{nibp_c}__n"]].sum(axis=1, min_count=1)

    # vasopressors -> wide value + recorded-this-hour count
    if not vaso.empty:
        vw = vaso.pivot_table(index=["stay_id", "hr"], columns="drug", values="dose", aggfunc="max").reset_index()
        wide = wide.merge(vw, on=["stay_id", "hr"], how="left")
        vcount = vaso.assign(one=1).pivot_table(index=["stay_id", "hr"], columns="drug",
                                                 values="one", aggfunc="sum").reset_index()
        vcount = vcount.rename(columns={d: f"{d}__n" for d in VASO_DRUGS})
        wide = wide.merge(vcount, on=["stay_id", "hr"], how="left")
    for d in VASO_DRUGS:
        if d not in wide.columns:
            wide[d] = np.nan
    del vaso
    gc.collect()

    # measured-this-hour channel (count > 0), captured from the __n columns BEFORE
    # they are dropped (drop the bulky intermediates to fit the box in RAM).
    counts: dict[str, np.ndarray] = {}
    for v in MEASURED_VARS:
        col = wide[f"{v}__n"] if f"{v}__n" in wide.columns else pd.Series(np.nan, index=wide.index)
        counts[v] = (col.fillna(0) > 0).astype("int8").to_numpy()
    drop = [c for c in wide.columns if c.endswith("__n")] + ["map_art", "sbp_art", "map_nibp", "sbp_nibp"]
    wide = wide.drop(columns=[c for c in drop if c in wide.columns])
    for c in wide.columns:                       # float32 halves the frame
        if c not in ("stay_id", "hr"):
            wide[c] = wide[c].astype("float32")
    gc.collect()

    wide = wide.sort_values(["stay_id", "hr"]).reset_index(drop=True)
    g = wide.groupby("stay_id", sort=False)

    # per-variable bounded forward fill + missingness mask (mirrors build.py)
    ffill_win = {**FFILL_HOURS, **EXTRA_FFILL}
    mask: dict[str, np.ndarray] = {}
    for v in MEASURED_VARS:
        if v not in wide.columns:
            wide[v] = np.float32("nan")
        wide[v] = g[v].ffill(limit=ffill_win.get(v, 6))
        mask[v] = wide[v].notna().astype("int8").to_numpy()
    for d in VASO_DRUGS:
        wide[d] = wide[d].fillna(0.0)
    wide["vent"] = g["vent"].ffill(limit=FFILL_HOURS["vent"]).fillna(0).astype("int8")
    wide["urine_rate"] = wide["urine_rate"].fillna(0.0)

    # derived respiratory features (room-air FiO2 default; S/F surrogate)
    fio2 = wide["fio2"].to_numpy(dtype="float64")
    fio2 = np.where(fio2 > 1.0, fio2 / 100.0, fio2)
    fio2 = np.where((fio2 < 0.21) | (fio2 > 1.0), np.nan, fio2)
    fio2 = np.where(np.isnan(fio2), fcfg.room_air_fio2, fio2)
    wide["pf_ratio"] = wide["pao2"].to_numpy(dtype="float64") / fio2
    spo2_capped = np.minimum(wide["spo2"].to_numpy(dtype="float64"), fcfg.spo2_sf_cap)
    wide["sf_ratio"] = spo2_capped / fio2
    wide["gcs_total"] = (wide["gcs_eye"].fillna(4) + wide["gcs_verbal"].fillna(5)
                         + wide["gcs_motor"].fillna(6))

    feat = wide[["stay_id", "hr"]].copy()
    for v in VALUE_COLS:
        feat[v] = wide[v] if v in wide.columns else np.nan
    if fcfg.include_measurement_features:
        for v in MEASURED_VARS:
            feat[f"{v}__measured"] = counts.get(v, 0)
            feat[f"{v}__mask"] = mask.get(v, 0)
    return feat, cohort


def assemble_and_write_eicu(cfg, lcfg, fcfg, feat, cohort) -> "object":
    """Window episodes, label, normalize with the FROZEN MIMIC stats, write."""
    full = CohortConfig(mode="full")
    norm_path = PATHS.features_root / f"norm_stats_{full.mode}.json"
    man_path = PATHS.features_root / f"feature_manifest_{full.mode}.json"
    for p in (norm_path, man_path):
        if not p.exists():
            raise FileNotFoundError(f"MIMIC {p.name} missing — build MIMIC full features first.")
    with norm_path.open() as fh:
        stats = json.load(fh)              # MIMIC train-split normalizer (FROZEN)
    with man_path.open() as fh:
        manifest = json.load(fh)

    part = pd.read_parquet(partition_path(cfg))
    model = part[part["label"].notna()].copy()
    model["label"] = model["label"].astype(int)
    sep = pd.read_parquet(sepsis3_path(cfg))[["stay_id", "onset_hour"]]
    model = model.merge(sep, on="stay_id", how="left")

    model["end_hour"] = _episode_ends(model, fcfg)
    model["obs_start"] = np.maximum(0, model["end_hour"] - fcfg.max_obs_hours)
    meta = model[["stay_id", "split", "label", "onset_hour", "end_hour", "obs_start",
                  "age", "gender", "site"]]
    df = feat.merge(meta, on="stay_id", how="inner")
    df = df[(df["hr"] >= df["obs_start"]) & (df["hr"] < df["end_hour"])].copy()

    lead = df["onset_hour"] - df["hr"]
    df["y"] = ((df["label"] == 1) & (lead >= 1) & (lead <= fcfg.alert_window_hours)).astype("int8")
    df["t_to_onset"] = np.where(df["label"] == 1, lead, np.nan).astype("float32")

    # --- normalization with the FROZEN MIMIC train statistics (NEVER refit) ---
    for c in CONT_COLS:
        mu, sd = stats[c]["mean"], stats[c]["std"]
        df[c] = ((df[c] - mu) / sd).astype("float32")
    df[CONT_COLS] = df[CONT_COLS].fillna(0.0)
    df["age_z"] = ((df["age"] - stats["age"]["mean"]) / stats["age"]["std"]).astype("float32")
    df["sex_female"] = (df["gender"] == "F").astype("int8")
    df["hour_idx"] = ((df["hr"] - stats["hour_idx"]["mean"]) / stats["hour_idx"]["std"]).astype("float32")

    # column order EXACTLY = manifest (schema parity with MIMIC)
    flat = [c for cols in manifest.values() for c in cols]
    keep = ["stay_id", "hr", "split", "label", "y", "t_to_onset", "end_hour"] + flat
    keep = [c for c in dict.fromkeys(keep) if c in df.columns]
    missing = [c for c in flat if c not in df.columns]
    if missing:
        raise ValueError(f"eICU features missing manifest columns: {missing}")
    out = df[keep].sort_values(["stay_id", "hr"]).reset_index(drop=True)

    PATHS.features_root.mkdir(parents=True, exist_ok=True)
    out.to_parquet(hourly_path(cfg), index=False)
    # copy the FROZEN MIMIC manifest + normalizer verbatim (asserted by tests)
    shutil.copyfile(man_path, PATHS.features_root / f"feature_manifest_{cfg.mode}.json")
    shutil.copyfile(norm_path, PATHS.features_root / f"norm_stats_{cfg.mode}.json")

    n_ep = out["stay_id"].nunique()
    ep = out.groupby("stay_id")["label"].first()
    log.info("eICU episodes: %d stays (%d pos) | %s hourly rows | %.1f%% positive hours",
             n_ep, int((ep == 1).sum()), f"{len(out):,}", 100 * out["y"].mean())
    log.info("  applied FROZEN MIMIC normalizer (norm_stats_full.json); copied manifest+stats -> *_eicu")
    log.info("  wrote %s", hourly_path(cfg))
    return hourly_path(cfg)


def run(cfg: CohortConfig | None = None, lcfg: LabelConfig | None = None,
        fcfg: FeatureConfig | None = None) -> None:
    cfg = cfg or CohortConfig(mode="eicu")
    lcfg = lcfg or LabelConfig.load()
    fcfg = fcfg or FeatureConfig.load()
    t0 = time.perf_counter()
    from .eicu_partition import run as run_partition

    run_partition(cfg, lcfg)
    feat, cohort = build_hourly_eicu(cfg, lcfg, fcfg)
    assemble_and_write_eicu(cfg, lcfg, fcfg, feat, cohort)
    log.info("eICU features done in %.1fs", time.perf_counter() - t0)
