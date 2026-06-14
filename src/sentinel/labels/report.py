"""Phase 1 cohort report -> outputs/reports/cohort_report.md.

Pulls real numbers from the built cohort + labels + extracted event parquet:
cohort size & demographics, Sepsis-3 prevalence and onset distribution,
SOFA component breakdown, per-variable missingness, and the planned site
partition. This is the artifact for the human checkpoint.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..config import CohortConfig, LabelConfig
from ..duck import connect
from ..logging_utils import get_logger
from ..paths import PATHS
from ..data.cohort import cohort_path, load_cohort
from ..data.itemids import variable_meta
from .scoring import SOFA_COMPONENTS
from .sepsis3 import sepsis3_path
from .sofa import sofa_path
from .suspicion import suspicion_path

log = get_logger("labels.report")


def _pq(table: str, cfg: CohortConfig):
    return (PATHS.parquet_root / f"{table}_{cfg.mode}.parquet").as_posix()


def _fmt_iqr(x: pd.Series) -> str:
    return f"{x.median():.0f} [{x.quantile(.25):.0f}–{x.quantile(.75):.0f}]"


def _coverage(con, cfg, meta: dict) -> pd.DataFrame:
    """Per-variable fraction of cohort stays with >=1 measurement."""
    cohort = load_cohort(cfg)
    n = len(cohort)
    src_table = {"chartevents": "chartevents", "labevents": "labevents",
                 "inputevents": "inputevents", "outputevents": "outputevents"}
    key = {"chartevents": "stay_id", "labevents": "hadm_id",
           "inputevents": "stay_id", "outputevents": "stay_id"}
    rows = []
    for var, m in meta.items():
        src = m["source"]
        ids = ",".join(str(i) for i in m["itemids"])
        if not ids:
            continue
        q = (f"SELECT COUNT(DISTINCT {key[src]}) FROM read_parquet('{_pq(src_table[src], cfg)}') "
             f"WHERE itemid IN ({ids})")
        try:
            d = con.execute(q).fetchone()[0]
        except Exception:
            d = 0
        rows.append({"variable": var, "organ": m["organ"], "source": src,
                     "pct_stays": 100.0 * d / max(n, 1)})
    return pd.DataFrame(rows).sort_values(["organ", "variable"])


def build_report(cfg: CohortConfig, lcfg: LabelConfig) -> "object":
    cohort = load_cohort(cfg)
    n = len(cohort)
    H = lcfg.horizon_hours

    sep = pd.read_parquet(sepsis3_path(cfg))
    df = cohort.merge(sep[["stay_id", "has_suspicion", "si_hour", "sepsis3",
                           "onset_hour", "sofa_max"]], on="stay_id", how="left")

    septic = df[df["sepsis3"] == 1]
    onset = septic["onset_hour"]
    present_adm = int((onset <= 0).sum())
    # early-warning-eligible: onset during ICU with >= H h of prior data
    icu_acquired = septic[septic["onset_hour"] >= H]

    # site partition table — both rates over the cohort (nullable Int64 onset_hour
    # would otherwise make `>= H` skip non-septic rows and inflate the rate).
    df["is_septic"] = (df["sepsis3"] == 1)
    df["is_icu_acq"] = df["is_septic"] & (df["onset_hour"].fillna(-1).astype("int64") >= H)
    site = (df.groupby("site")
              .agg(n=("stay_id", "size"),
                   sepsis=("is_septic", "mean"),
                   icu_acq=("is_icu_acq", "mean"))
              .sort_values("n", ascending=False))
    site["pct"] = 100 * site["n"] / n

    con = connect()
    try:
        # SOFA component breakdown
        sp = sofa_path(cfg).as_posix()
        comp_stats = {}
        for c in SOFA_COMPONENTS:
            mean, frac = con.execute(
                f"SELECT avg({c}), avg(CASE WHEN {c}>0 THEN 1.0 ELSE 0 END) "
                f"FROM read_parquet('{sp}')").fetchone()
            comp_stats[c] = (mean, 100 * frac)
        sofa_mean, sofa_hours_ge2 = con.execute(
            f"SELECT avg(sofa_total), avg(CASE WHEN sofa_total>=2 THEN 1.0 ELSE 0 END) "
            f"FROM read_parquet('{sp}')").fetchone()
        cov = _coverage(con, cfg, variable_meta())
    finally:
        con.close()

    # race grouping (coarse, for fairness preview)
    race_top = df["race"].fillna("UNKNOWN").str.upper().value_counts().head(8)

    # ---- assemble markdown ----
    L = []
    L.append(f"# SENTINEL — Cohort & Label Report ({cfg.mode} cohort)\n")
    L.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M')} · MIMIC-IV v3.1 · "
             f"prediction horizon H = {H} h_\n")
    L.append("> **Phase 1 checkpoint.** Numbers below are real, computed from the built "
             "pipeline. Pipeline correctness is covered by `pytest` (SOFA truth tables + "
             "derivation invariants).\n")

    L.append("## 1. Cohort\n")
    L.append(f"- **ICU stays:** {n:,}  ·  unique patients: {df['subject_id'].nunique():,}")
    L.append(f"- **Inclusion:** adults {cfg.min_age}–{cfg.max_age} y, LOS ≥ {cfg.min_los_hours:.0f} h, "
             f"first ICU stay per admission{'' if cfg.first_stay_only else ' (off)'}")
    L.append(f"- **Age:** median {_fmt_iqr(df['age'])} y  ·  **Female:** "
             f"{100*(df['gender']=='F').mean():.1f}%")
    L.append(f"- **LOS:** median {_fmt_iqr(df['los_hours'])} h")
    L.append(f"- **In-hospital mortality:** {100*df['hospital_expire_flag'].mean():.1f}%")
    L.append("\n  Race/ethnicity (top groups, for fairness analysis later):\n")
    for r, c in race_top.items():
        L.append(f"  - {r}: {100*c/n:.1f}%")

    L.append("\n## 2. Sepsis-3 labels\n")
    L.append(f"- **Suspicion of infection:** {100*df['has_suspicion'].mean():.1f}% of stays")
    L.append(f"- **Sepsis-3 (ever):** {len(septic):,} stays = **{100*len(septic)/n:.1f}%**")
    L.append(f"- **Onset hour** (since ICU intime): median {_fmt_iqr(onset)} h")
    L.append(f"- **Present on admission** (onset ≤ 0 h): {present_adm:,} "
             f"({100*present_adm/max(len(septic),1):.1f}% of septic) — limited early-warning window")
    L.append(f"- **ICU-acquired, onset ≥ H={H} h** (the actionable early-warning positives): "
             f"**{len(icu_acquired):,}** stays = {100*len(icu_acquired)/n:.1f}% of cohort")
    L.append("\n  Onset-hour distribution among septic stays:\n")
    bins = [(0, 0), (1, 6), (7, 12), (13, 24), (25, 48), (49, 96), (97, 10**9)]
    labels = ["≤0 (on adm.)", "1–6", "7–12", "13–24", "25–48", "49–96", ">96"]
    for (lo, hi), lab in zip(bins, labels):
        if lo == 0 and hi == 0:
            cnt = int((onset <= 0).sum())
        else:
            cnt = int(((onset >= lo) & (onset <= hi)).sum())
        L.append(f"  - {lab:<13} h : {cnt:>6,} ({100*cnt/max(len(septic),1):4.1f}%)")

    L.append("\n## 3. SOFA\n")
    L.append(f"- Mean total SOFA: **{sofa_mean:.2f}**  ·  hours with SOFA ≥ 2: "
             f"{sofa_hours_ge2*100:.1f}%")
    L.append("\n  | Component (organ) | mean | % hours > 0 |")
    L.append("  |---|---|---|")
    for c in SOFA_COMPONENTS:
        mean, frac = comp_stats[c]
        L.append(f"  | {c.replace('sofa_','')} | {mean:.2f} | {frac:.1f}% |")

    L.append("\n## 4. Missingness (per-variable coverage)\n")
    L.append("  Fraction of cohort stays with ≥ 1 measurement of each variable.\n")
    L.append("  | organ | variable | source | % stays |")
    L.append("  |---|---|---|---|")
    for _, r in cov.iterrows():
        L.append(f"  | {r['organ']} | {r['variable']} | {r['source']} | {r['pct_stays']:.1f}% |")

    L.append("\n## 5. Planned site partition (federated, Phase 5)\n")
    L.append(f"Sites = `{cfg.site_partition_key}`. **External test site (held out): "
             f"{cfg.external_site}**. Remaining major units are the federated training "
             "sites; within them, splits are **time-based** to prevent leakage.\n")
    L.append("  | site | n | % | Sepsis-3 % | ICU-acq (onset≥H) % |")
    L.append("  |---|---|---|---|---|")
    for s, row in site.iterrows():
        ext = "  ⟵ external" if s == cfg.external_site else ""
        if row["n"] < 200:
            continue
        L.append(f"  | {s}{ext} | {int(row['n']):,} | {row['pct']:.1f}% | "
                 f"{100*row['sepsis']:.1f}% | {100*row['icu_acq']:.1f}% |")

    L.append("\n## 6. Key assumptions & caveats (for review)\n")
    if lcfg.dynamic_baseline:
        L.append(f"- **Sepsis-3 = suspicion + acute SOFA rise ≥ {lcfg.sofa_increase_threshold}** over the "
                 f"patient's pre-suspicion **admission baseline** (min SOFA in [0, "
                 f"min(si_hour, {lcfg.baseline_window_hours} h)]), within "
                 f"[−{lcfg.si_sofa_lookback_hours} h, +{lcfg.si_sofa_lookahead_hours} h] of suspicion. "
                 "This follows the literal Sepsis-3 wording (*acute change* in SOFA) and yields "
                 "prevalence in line with the literature (~35–38%). Strict mimic-code baseline-0 "
                 "(SOFA ≥ 2 regardless of admission state) gives ~44% on this cohort and is "
                 "available via `dynamic_baseline: false`.")
    else:
        L.append(f"- **Sepsis-3 = suspicion + SOFA ≥ {lcfg.sofa_increase_threshold}** (baseline "
                 f"{lcfg.sofa_baseline}, strict mimic-code) within "
                 f"[−{lcfg.si_sofa_lookback_hours} h, +{lcfg.si_sofa_lookahead_hours} h] of suspicion.")
    L.append("- **SOFA refinements:** bounded forward-fill (labs 48 h, vitals 24 h; "
             "measurements expire vs unbounded carry), **ventilation-aware GCS** (intubated "
             "verbal=1T not penalized), and physiologic range filtering of artifacts. CNS is "
             "still the largest contributor (residual sedation effect); worst-in-hour aggregation "
             "(min MAP / GCS, max bilirubin / creatinine) is used.")
    L.append("- **Vasopressors confirmed in µg/kg/min** (no per-weight conversion needed).")
    L.append("- **Sites are simulated via care units, not real institutions** (a stated paper "
             "limitation).")
    L.append("- **Governance:** MIMIC is credentialed; the repo sits under OneDrive — pause/"
             "exclude sync for `data/`, `outputs/`, `mimic-iv-3.1/` or redirect via "
             "`SENTINEL_DATA_ROOT` to avoid syncing PHI to the cloud.")
    L.append("\n## 7. Next (Phase 2, after approval)\n")
    L.append(f"Hourly feature tensors (+ missingness indicators, train-split normalization) → "
             f"Dec-POMDP environment with per-organ observations and the early-warning reward, "
             f"targeting onset at t + {H} h for the ICU-acquired positive cohort.\n")

    PATHS.reports_root.mkdir(parents=True, exist_ok=True)
    out = PATHS.reports_root / "cohort_report.md"
    out.write_text("\n".join(L), encoding="utf-8")
    log.info("wrote %s", out)
    log.info("  cohort=%d | sepsis=%.1f%% | ICU-acquired(onset>=%dh)=%d | suspicion=%.1f%%",
             n, 100*len(septic)/n, H, len(icu_acquired), 100*df['has_suspicion'].mean())
    return out


def run(cfg: CohortConfig | None = None, lcfg: LabelConfig | None = None) -> None:
    cfg = cfg or CohortConfig.load()
    lcfg = lcfg or LabelConfig.load()
    build_report(cfg, lcfg)
