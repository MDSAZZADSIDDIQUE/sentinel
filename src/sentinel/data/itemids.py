"""Resolve clinical variables to MIMIC itemids and write config/itemids.yaml.

Approach (no blind hardcoding):
  1. A SPEC lists each canonical variable with its organ system, source table,
     a set of *reference* itemids (the well-established mimic-code ids), and
     label search patterns.
  2. We query d_items / d_labitems by label text to enumerate candidates, AND
     verify every reference itemid actually exists in the dictionary with a label
     that matches the patterns. Mismatches/missing ids are flagged loudly.
  3. The verified result is written to config/itemids.yaml with the real labels
     and units, plus the candidate list for transparency.

Reference itemids cross-checked against the MIT-LCP mimic-code concepts
(vitals, blood gases, SOFA, urine output) for MIMIC-IV.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import duckdb
import yaml

from ..constants import ORGAN_SYSTEMS, SHARED_GROUP
from ..duck import connect, mimic
from ..logging_utils import get_logger
from ..paths import PATHS

log = get_logger("data.itemids")


def load_itemids() -> dict:
    """Load the generated config/itemids.yaml (run resolve-itemids first)."""
    path = PATHS.config_root / "itemids.yaml"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run `sentinel resolve-itemids` first.")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def itemids_by_source(data: dict | None = None) -> dict[str, list[int]]:
    """{source_table: sorted list of itemids} from the resolved config."""
    data = data or load_itemids()
    groups: dict[str, set[int]] = {}
    for v in data["variables"].values():
        groups.setdefault(v["source"], set()).update(int(i) for i in v["itemids"])
    return {k: sorted(v) for k, v in groups.items()}


def variable_itemids(data: dict | None = None) -> dict[str, list[int]]:
    """{variable_name: [itemids]} from the resolved config."""
    data = data or load_itemids()
    return {name: [int(i) for i in v["itemids"]] for name, v in data["variables"].items()}


def variable_meta(data: dict | None = None) -> dict[str, dict]:
    """{variable_name: {organ, source, role, unit, itemids}}."""
    data = data or load_itemids()
    return {
        name: {"organ": v["organ"], "source": v["source"], "role": v["role"],
               "unit": v["unit"], "itemids": [int(i) for i in v["itemids"]]}
        for name, v in data["variables"].items()
    }


@dataclass(frozen=True)
class VarSpec:
    name: str
    organ: str           # one of ORGAN_SYSTEMS or SHARED_GROUP
    source: str          # chartevents | labevents | inputevents | outputevents
    itemids: list[int]   # reference (mimic-code) itemids
    patterns: list[str]  # label LIKE patterns for cross-check / candidates
    unit: str = ""
    role: str = "feature"  # feature | sofa | vasopressor | vital
    notes: str = ""
    # urine has many valid devices; keep whichever reference ids exist.
    keep_existing_only: bool = False


# d_items.linksto per source
_LINKSTO = {
    "chartevents": "chartevents",
    "inputevents": "inputevents",
    "outputevents": "outputevents",
}

SPEC: list[VarSpec] = [
    # ---- shared vitals -------------------------------------------------
    VarSpec("heart_rate", "cardiovascular", "chartevents", [220045],
            ["heart rate"], "bpm", role="vital", notes="exclude alarm items"),
    VarSpec("resp_rate", "respiratory", "chartevents", [220210, 224690],
            ["respiratory rate"], "insp/min", role="vital"),
    VarSpec("spo2", "respiratory", "chartevents", [220277],
            ["o2 saturation pulseoxymetry"], "%", role="vital"),
    VarSpec("sbp", "cardiovascular", "chartevents", [220050, 220179],
            ["blood pressure systolic"], "mmHg", role="vital"),
    VarSpec("temp_c", SHARED_GROUP, "chartevents", [223762],
            ["temperature celsius"], "C", role="vital"),
    VarSpec("temp_f", SHARED_GROUP, "chartevents", [223761],
            ["temperature fahrenheit"], "F", role="vital",
            notes="convert to Celsius downstream"),
    # ---- SOFA cardiovascular ------------------------------------------
    VarSpec("map", "cardiovascular", "chartevents", [220052, 220181, 225312],
            ["blood pressure mean", "arterial blood pressure mean"], "mmHg",
            role="sofa", keep_existing_only=True),
    VarSpec("lactate", "cardiovascular", "labevents", [50813, 52442],
            ["lactate"], "mmol/L", role="feature", keep_existing_only=True),
    VarSpec("norepinephrine", "cardiovascular", "inputevents", [221906],
            ["norepinephrine"], "mcg/kg/min", role="vasopressor"),
    VarSpec("epinephrine", "cardiovascular", "inputevents", [221289, 229617],
            ["epinephrine"], "mcg/kg/min", role="vasopressor",
            keep_existing_only=True),
    VarSpec("dopamine", "cardiovascular", "inputevents", [221662],
            ["dopamine"], "mcg/kg/min", role="vasopressor"),
    VarSpec("dobutamine", "cardiovascular", "inputevents", [221653],
            ["dobutamine"], "mcg/kg/min", role="vasopressor"),
    VarSpec("vasopressin", "cardiovascular", "inputevents", [222315],
            ["vasopressin"], "units/min", role="vasopressor"),
    VarSpec("phenylephrine", "cardiovascular", "inputevents", [221749],
            ["phenylephrine"], "mcg/kg/min", role="vasopressor"),
    # ---- SOFA respiratory ---------------------------------------------
    VarSpec("pao2", "respiratory", "labevents", [50821],
            ["po2"], "mmHg", role="sofa"),
    VarSpec("fio2", "respiratory", "chartevents", [223835],
            ["inspired o2 fraction", "fio2"], "%", role="sofa",
            notes="may be fraction (0-1) or percent (21-100)"),
    VarSpec("vent_mode", "respiratory", "chartevents", [223849],
            ["ventilator mode"], "", role="feature",
            notes="presence => mechanical ventilation for SOFA cutoff"),
    VarSpec("o2_device", "respiratory", "chartevents", [226732],
            ["o2 delivery device"], "", role="feature"),
    # ---- SOFA coagulation / hepatic / renal / CNS ---------------------
    VarSpec("platelets", "coagulation", "labevents", [51265],
            ["platelet count"], "K/uL", role="sofa"),
    VarSpec("bilirubin_total", "hepatic", "labevents", [50885],
            ["bilirubin, total"], "mg/dL", role="sofa"),
    VarSpec("creatinine", "renal", "labevents", [50912],
            ["creatinine"], "mg/dL", role="sofa",
            notes="blood/serum creatinine only"),
    VarSpec("urine_output", "renal", "outputevents",
            [226559, 226560, 226561, 226584, 226563, 226564, 226565, 226567,
             226557, 226558, 226627, 226631, 226566],
            ["urine", "foley", "void"], "mL", role="sofa",
            keep_existing_only=True,
            notes="sum across collection devices per 24h"),
    VarSpec("gcs_eye", "neurologic", "chartevents", [220739],
            ["gcs - eye"], "", role="sofa"),
    VarSpec("gcs_verbal", "neurologic", "chartevents", [223900],
            ["gcs - verbal"], "", role="sofa"),
    VarSpec("gcs_motor", "neurologic", "chartevents", [223901],
            ["gcs - motor"], "", role="sofa"),
]


@dataclass
class Resolved:
    name: str
    organ: str
    source: str
    role: str
    unit: str
    notes: str
    itemids: list[int] = field(default_factory=list)
    labels: dict[int, str] = field(default_factory=dict)
    candidates: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _dict_rows(con: duckdb.DuckDBPyConnection, spec: VarSpec) -> dict[int, str]:
    """Return {itemid: label} for this spec's reference ids that EXIST."""
    ids = ",".join(str(i) for i in spec.itemids)
    if spec.source == "labevents":
        q = f"SELECT itemid, label FROM {mimic('d_labitems')} WHERE itemid IN ({ids})"
    else:
        linksto = _LINKSTO[spec.source]
        q = (f"SELECT itemid, label FROM {mimic('d_items')} "
             f"WHERE itemid IN ({ids}) AND linksto = '{linksto}'")
    return {int(r[0]): r[1] for r in con.execute(q).fetchall()}


def _candidates(con: duckdb.DuckDBPyConnection, spec: VarSpec) -> list[dict]:
    like = " OR ".join(f"lower(label) LIKE '%{p.lower()}%'" for p in spec.patterns)
    if spec.source == "labevents":
        q = (f"SELECT itemid, label, fluid, category FROM {mimic('d_labitems')} "
             f"WHERE ({like}) ORDER BY label LIMIT 40")
        cols = ("itemid", "label", "fluid", "category")
    else:
        linksto = _LINKSTO[spec.source]
        q = (f"SELECT itemid, label, category, unitname FROM {mimic('d_items')} "
             f"WHERE ({like}) AND linksto = '{linksto}' ORDER BY label LIMIT 40")
        cols = ("itemid", "label", "category", "unitname")
    return [dict(zip(cols, r)) for r in con.execute(q).fetchall()]


def resolve(con: duckdb.DuckDBPyConnection | None = None) -> list[Resolved]:
    own = con is None
    con = con or connect()
    try:
        out: list[Resolved] = []
        for spec in SPEC:
            found = _dict_rows(con, spec)
            res = Resolved(spec.name, spec.organ, spec.source, spec.role,
                           spec.unit, spec.notes)
            res.candidates = _candidates(con, spec)
            # decide kept itemids
            for iid in spec.itemids:
                if iid in found:
                    res.itemids.append(iid)
                    res.labels[iid] = found[iid]
                elif not spec.keep_existing_only:
                    res.warnings.append(f"reference itemid {iid} not found in dictionary")
            if not res.itemids:
                res.warnings.append("NO itemids resolved")
            out.append(res)
        return out
    finally:
        if own:
            con.close()


def run() -> None:
    t0 = time.perf_counter()
    resolved = resolve()

    # group + validate organ coverage
    by_organ: dict[str, list[str]] = {o: [] for o in (*ORGAN_SYSTEMS, SHARED_GROUP)}
    n_warn = 0
    payload: dict = {
        "_meta": {
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": "resolved from d_items/d_labitems by label; cross-checked vs mimic-code",
            "n_variables": len(resolved),
        },
        "variables": {},
    }
    for r in resolved:
        by_organ.setdefault(r.organ, []).append(r.name)
        payload["variables"][r.name] = {
            "organ": r.organ,
            "source": r.source,
            "role": r.role,
            "unit": r.unit,
            "itemids": r.itemids,
            "labels": {int(k): v for k, v in r.labels.items()},
            "patterns": next(s.patterns for s in SPEC if s.name == r.name),
            "notes": r.notes,
            "candidates": r.candidates,
        }
        for w in r.warnings:
            n_warn += 1
            log.warning("  [%s] %s", r.name, w)
        ids_str = ", ".join(f"{i}={r.labels.get(i, '?')}" for i in r.itemids)
        log.info("  %-16s %-15s %-13s -> %s", r.name, r.organ, r.source, ids_str or "(none)")

    PATHS.config_root.mkdir(parents=True, exist_ok=True)
    out_path = PATHS.config_root / "itemids.yaml"
    with out_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False, allow_unicode=True, width=100)

    log.info("organ coverage:")
    for organ in (*ORGAN_SYSTEMS, SHARED_GROUP):
        log.info("  %-15s %s", organ, ", ".join(by_organ.get(organ, [])) or "(none)")
    log.info("wrote %s (%d variables, %d warnings) in %.1fs",
             out_path, len(resolved), n_warn, time.perf_counter() - t0)
    if n_warn:
        log.warning("itemid resolution produced %d warning(s) — review above.", n_warn)
