"""One-off eICU schema + distribution probe (Phase 9, Step 0).

Verifies the harmonization map against the REAL column names, units, and value
distributions BEFORE any derivation code is written (mirrors the Phase-1 itemid
discipline: no blind hardcoding). Nothing is written to disk; results print to
stdout for inspection.

Usage:
    python scripts/eicu_probe.py            # cheap: schemas + small-table value sets
    python scripts/eicu_probe.py heavy      # full scans of the big tables (slow)
    python scripts/eicu_probe.py lab nurse  # run named probes only

Governance: read-only, aggregate output only; no row-level PHI is persisted.
"""
from __future__ import annotations

import sys

import pandas as pd

from sentinel.duck import connect, eicu

pd.set_option("display.max_rows", 80)
pd.set_option("display.max_colwidth", 60)
pd.set_option("display.width", 200)

NEEDED = ["patient", "lab", "nurseCharting", "vitalPeriodic", "vitalAperiodic",
          "infusionDrug", "medication", "microLab", "intakeOutput",
          "respiratoryCare", "respiratoryCharting", "apacheApsVar"]


def show(con, label, sql, n=80):
    print("\n" + "=" * 78 + f"\n# {label}\n" + "-" * 78)
    try:
        df = con.execute(sql).df()
        print(df.head(n).to_string(index=False))
    except Exception as e:  # noqa: BLE001 — probe must keep going
        print(f"  !! ERROR: {type(e).__name__}: {e}")


def probe_schemas(con):
    for t in NEEDED:
        show(con, f"SCHEMA {t}", f"DESCRIBE SELECT * FROM {eicu(t)}")


def probe_patient(con):
    show(con, "patient: counts", f"""
        SELECT count(*) AS rows,
               count(DISTINCT patientunitstayid) AS stays,
               count(DISTINCT uniquepid)         AS patients,
               count(DISTINCT hospitalid)        AS hospitals
        FROM {eicu('patient')}""")
    show(con, "patient: age value set (string!)", f"""
        SELECT age, count(*) n FROM {eicu('patient')} GROUP BY age ORDER BY n DESC""")
    show(con, "patient: gender", f"""
        SELECT gender, count(*) n FROM {eicu('patient')} GROUP BY gender ORDER BY n DESC""")
    show(con, "patient: unittype", f"""
        SELECT unittype, count(*) n FROM {eicu('patient')} GROUP BY unittype ORDER BY n DESC""")
    show(con, "patient: LOS (unitdischargeoffset minutes) + hospitaladmitoffset quantiles", f"""
        SELECT
          quantile_cont(unitdischargeoffset, [0.1,0.25,0.5,0.75,0.9]) AS los_min_q,
          quantile_cont(hospitaladmitoffset, [0.1,0.5,0.9])           AS hadm_off_q,
          min(unitdischargeoffset) AS los_min, max(unitdischargeoffset) AS los_max
        FROM {eicu('patient')}""")
    show(con, "patient: hospitals with >=200 stays (federation sites preview)", f"""
        SELECT count(*) AS n_hospitals_ge200 FROM (
          SELECT hospitalid, count(*) n FROM {eicu('patient')}
          GROUP BY hospitalid HAVING count(*) >= 200)""")


def probe_microlab(con):
    show(con, "microLab: sample rows", f"SELECT * FROM {eicu('microLab')} LIMIT 8")
    show(con, "microLab: counts + coverage", f"""
        SELECT count(*) AS rows,
               count(DISTINCT patientunitstayid) AS stays_with_culture
        FROM {eicu('microLab')}""")
    show(con, "microLab: culturesite", f"""
        SELECT culturesite, count(*) n FROM {eicu('microLab')}
        GROUP BY culturesite ORDER BY n DESC LIMIT 30""")


def probe_resp(con):
    show(con, "respiratoryCare: sample", f"SELECT * FROM {eicu('respiratoryCare')} LIMIT 5")
    show(con, "apacheApsVar: sample (GCS eyes/motor/verbal)", f"SELECT * FROM {eicu('apacheApsVar')} LIMIT 5")


# ---- heavy (full-scan) probes -------------------------------------------
def probe_counts(con):
    for t in ["vitalPeriodic", "nurseCharting", "lab", "infusionDrug",
              "medication", "intakeOutput", "vitalAperiodic", "respiratoryCharting"]:
        show(con, f"COUNT {t}", f"SELECT count(*) AS rows FROM {eicu(t)}")


def probe_lab(con):
    show(con, "lab: top labname by count", f"""
        SELECT labname, count(*) n, any_value(labmeasurenamesystem) AS unit
        FROM {eicu('lab')} GROUP BY labname ORDER BY n DESC LIMIT 60""")
    show(con, "lab: SOFA analytes — unit + quantiles", f"""
        SELECT labname, any_value(labmeasurenamesystem) AS unit, count(*) n,
               quantile_cont(labresult, [0.05,0.5,0.95]) AS q
        FROM {eicu('lab')}
        WHERE lower(labname) IN ('pao2','fio2','paco2','platelets','platelet count',
              'total bilirubin','creatinine','lactate','bun','wbc x 1000','ph')
           OR lower(labname) LIKE '%bilirub%' OR lower(labname) LIKE '%platelet%'
           OR lower(labname) LIKE '%pao2%' OR lower(labname) LIKE '%fio2%'
        GROUP BY labname ORDER BY n DESC LIMIT 40""")


def probe_nurse(con):
    show(con, "nurseCharting: top celltypevallabel", f"""
        SELECT nursingchartcelltypevallabel AS label, count(*) n
        FROM {eicu('nurseCharting')} GROUP BY 1 ORDER BY n DESC LIMIT 50""")
    show(con, "nurseCharting: top valname for GCS/vitals", f"""
        SELECT nursingchartcelltypevallabel AS label,
               nursingchartcelltypevalname  AS name, count(*) n
        FROM {eicu('nurseCharting')}
        WHERE lower(nursingchartcelltypevallabel) LIKE '%glasgow%'
           OR lower(nursingchartcelltypevallabel) LIKE '%gcs%'
           OR lower(nursingchartcelltypevallabel) LIKE '%heart rate%'
           OR lower(nursingchartcelltypevallabel) LIKE '%temperature%'
           OR lower(nursingchartcelltypevallabel) LIKE '%respiratory%'
           OR lower(nursingchartcelltypevallabel) LIKE '%o2 sat%'
           OR lower(nursingchartcelltypevallabel) LIKE '%bp%'
        GROUP BY 1, 2 ORDER BY n DESC LIMIT 60""")


def probe_infusion(con):
    show(con, "infusionDrug: top drugname", f"""
        SELECT drugname, count(*) n FROM {eicu('infusionDrug')}
        GROUP BY drugname ORDER BY n DESC LIMIT 60""")
    show(con, "infusionDrug: vasopressor-like names", f"""
        SELECT drugname, count(*) n, any_value(drugrate) AS ex_rate
        FROM {eicu('infusionDrug')}
        WHERE lower(drugname) LIKE '%norepi%' OR lower(drugname) LIKE '%epineph%'
           OR lower(drugname) LIKE '%dopamine%' OR lower(drugname) LIKE '%dobutam%'
           OR lower(drugname) LIKE '%vasopressin%' OR lower(drugname) LIKE '%phenyleph%'
           OR lower(drugname) LIKE '%levophed%' OR lower(drugname) LIKE '%neosynephrine%'
        GROUP BY drugname ORDER BY n DESC LIMIT 40""")


def probe_medication(con):
    show(con, "medication: routeadmin", f"""
        SELECT routeadmin, count(*) n FROM {eicu('medication')}
        GROUP BY routeadmin ORDER BY n DESC LIMIT 25""")
    show(con, "medication: antibiotic-like drugname", f"""
        SELECT drugname, count(*) n FROM {eicu('medication')}
        WHERE lower(drugname) LIKE '%cillin%' OR lower(drugname) LIKE '%cef%'
           OR lower(drugname) LIKE '%vanco%' OR lower(drugname) LIKE '%zosyn%'
           OR lower(drugname) LIKE '%pipercillin%' OR lower(drugname) LIKE '%piperacillin%'
           OR lower(drugname) LIKE '%meropenem%' OR lower(drugname) LIKE '%cipro%'
           OR lower(drugname) LIKE '%levofloxacin%' OR lower(drugname) LIKE '%flagyl%'
           OR lower(drugname) LIKE '%metronidazole%' OR lower(drugname) LIKE '%azithro%'
        GROUP BY drugname ORDER BY n DESC LIMIT 40""")


def probe_intake(con):
    show(con, "intakeOutput: urine-like celllabel", f"""
        SELECT celllabel, count(*) n, quantile_cont(cellvaluenumeric,[0.5,0.95]) AS q
        FROM {eicu('intakeOutput')}
        WHERE lower(celllabel) LIKE '%urine%' OR lower(celllabel) LIKE '%foley%'
           OR lower(celllabel) LIKE '%void%' OR lower(celllabel) LIKE '%uop%'
        GROUP BY celllabel ORDER BY n DESC LIMIT 40""")


def probe_vitals(con):
    show(con, "vitalPeriodic: SOFA/vital quantiles", f"""
        SELECT
          quantile_cont(sao2,[0.05,0.5,0.95]) AS sao2_q,
          quantile_cont(temperature,[0.05,0.5,0.95]) AS temp_q,
          quantile_cont(heartrate,[0.05,0.5,0.95]) AS hr_q,
          quantile_cont(respiration,[0.05,0.5,0.95]) AS rr_q,
          quantile_cont(systemicmean,[0.05,0.5,0.95]) AS map_q
        FROM {eicu('vitalPeriodic')}""")
    show(con, "vitalAperiodic: noninvasive quantiles", f"""
        SELECT quantile_cont(noninvasivemean,[0.05,0.5,0.95]) AS nibp_mean_q,
               quantile_cont(noninvasivesystolic,[0.05,0.5,0.95]) AS nibp_sys_q
        FROM {eicu('vitalAperiodic')}""")


CHEAP = {"schemas": probe_schemas, "patient": probe_patient,
         "microlab": probe_microlab, "resp": probe_resp}
HEAVY = {"counts": probe_counts, "lab": probe_lab, "nurse": probe_nurse,
         "infusion": probe_infusion, "medication": probe_medication,
         "intake": probe_intake, "vitals": probe_vitals}
ALL = {**CHEAP, **HEAVY}


def main(argv):
    args = [a.lower() for a in argv[1:]] or ["cheap"]
    if args == ["cheap"]:
        todo = CHEAP
    elif args == ["heavy"]:
        todo = HEAVY
    elif args == ["all"]:
        todo = ALL
    else:
        todo = {k: ALL[k] for k in args if k in ALL}
        unknown = [a for a in args if a not in ALL]
        if unknown:
            print(f"unknown probes {unknown}; known: {sorted(ALL)}")
    con = connect()
    try:
        for name, fn in todo.items():
            fn(con)
    finally:
        con.close()
    print("\n[done]")


if __name__ == "__main__":
    main(sys.argv)
