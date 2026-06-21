"""Phase 9 eICU cohort/label report -> outputs/reports/eicu_report.md.

The external-validation checkpoint artifact. Reports REAL numbers from the built
eICU cohort + labels, side by side with MIMIC (full mode, if present), and is
deliberately honest about the eICU label-fidelity gap (sparse cultures ->
antibiotic-based suspicion). Includes hand-checkable Sepsis-3 onset cases (SOFA
trajectory around onset) for manual validation, per the Phase-9 checkpoint.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..config import CohortConfig, LabelConfig
from ..logging_utils import get_logger
from ..paths import PATHS
from ..data.cohort import cohort_path, load_cohort
from .scoring import SOFA_COMPONENTS
from .sepsis3 import sepsis3_path
from .sofa import sofa_path
from .splits import CONTROL, EXCLUDED, POSITIVE, assign_groups
from .suspicion import suspicion_path

log = get_logger("labels.eicu_report")


def _fmt_iqr(x: pd.Series) -> str:
    return f"{x.median():.0f} [{x.quantile(.25):.0f}-{x.quantile(.75):.0f}]"


def _mimic_compare(H: int) -> dict | None:
    """Pull MIMIC (full) prevalence/suspicion for side-by-side, if available."""
    try:
        mcfg = CohortConfig(mode="full")
        if not (cohort_path(mcfg).exists() and sepsis3_path(mcfg).exists()):
            return None
        mc = load_cohort(mcfg)
        ms = pd.read_parquet(sepsis3_path(mcfg))
        n = len(mc)
        septic = ms["sepsis3"] == 1
        onset = ms["onset_hour"].fillna(-1).astype("int64")
        return {
            "n": n,
            "suspicion": 100 * ms["has_suspicion"].mean(),
            "sepsis": 100 * septic.mean(),
            "icu_acq": 100 * (septic & (onset >= H)).mean(),
            "median_age": mc["age"].median(),
            "female": 100 * (mc["gender"] == "F").mean(),
            "mortality": 100 * mc["hospital_expire_flag"].mean(),
            "median_los": mc["los_hours"].median(),
        }
    except Exception as exc:  # pragma: no cover
        log.warning("MIMIC comparison unavailable: %s", exc)
        return None


def build_report(cfg: CohortConfig, lcfg: LabelConfig) -> "object":
    H = lcfg.horizon_hours
    cohort = load_cohort(cfg)
    n = len(cohort)
    sep = pd.read_parquet(sepsis3_path(cfg))
    susp = pd.read_parquet(suspicion_path(cfg))
    sofa = pd.read_parquet(sofa_path(cfg))

    df = cohort.merge(sep[["stay_id", "has_suspicion", "si_hour", "sepsis3",
                           "onset_hour", "sofa_at_onset", "baseline_sofa", "sofa_max"]],
                      on="stay_id", how="left")
    septic = df[df["sepsis3"] == 1]
    onset = septic["onset_hour"]
    present_adm = int((onset <= 0).sum())
    icu_acq = septic[septic["onset_hour"] >= H]

    groups = assign_groups(df, H)
    n_pos = int((groups == POSITIVE).sum())
    n_exc = int((groups == EXCLUDED).sum())
    n_ctrl = int((groups == CONTROL).sum())

    # suspicion coverage breakdown (the eICU fidelity story)
    cov = {k: 100 * susp[k].mean() for k in
           ["has_antibiotic", "has_culture", "has_strict_pair", "has_suspicion"]}

    # SOFA component breakdown
    comp = {c: (sofa[c].mean(), 100 * (sofa[c] > 0).mean()) for c in SOFA_COMPONENTS}
    sofa_mean = sofa["sofa_total"].mean()

    # per-hospital federation preview (sites with >= 200 stays)
    df["is_septic"] = df["sepsis3"] == 1
    df["is_icu_acq"] = df["is_septic"] & (df["onset_hour"].fillna(-1).astype("int64") >= H)
    site = (df.groupby("hospitalid")
              .agg(n=("stay_id", "size"), sepsis=("is_septic", "mean"),
                   icu_acq=("is_icu_acq", "mean")).sort_values("n", ascending=False))
    n_sites_200 = int((site["n"] >= 200).sum())

    mim = _mimic_compare(H)

    # ---- assemble markdown ----
    L = []
    L.append("# SENTINEL — eICU External Validation: Cohort & Label Report\n")
    L.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M')} · eICU-CRDB v2.0 · "
             f"prediction horizon H = {H} h · all stays = EXTERNAL split_\n")
    L.append("> **Phase 9 checkpoint (Steps 0-3).** Real numbers from the eICU pipeline. "
             "MIMIC-trained models are NOT retrained here; eICU is held-out external test "
             "data. Label derivation reuses the MIMIC Sepsis-3 logic (dynamic-baseline acute "
             "SOFA rise, three-way split) verbatim; only the data SOURCES are harmonized.\n")

    L.append("## 1. Cohort\n")
    L.append(f"- **ICU (unit) stays:** {n:,} · unique patients: {df['subject_id'].nunique():,} "
             f"· hospitals: {df['hospitalid'].nunique()}")
    L.append(f"- **Inclusion:** adults {cfg.min_age}+ (eICU '> 89' parsed to 90 and retained), "
             f"LOS >= {cfg.min_los_hours:.0f} h, first eligible unit stay per patient")
    L.append(f"- **Age:** median {_fmt_iqr(df['age'])} y · **Female:** {100*(df['gender']=='F').mean():.1f}%")
    L.append(f"- **LOS:** median {_fmt_iqr(df['los_hours'])} h · **In-hospital mortality:** "
             f"{100*df['hospital_expire_flag'].mean():.1f}%")
    L.append(f"- **Unit types:** " + ", ".join(f"{k} {v:,}" for k, v in
             df["unittype"].value_counts().items()))

    L.append("\n## 2. Suspicion of infection — the eICU fidelity gap (READ)\n")
    L.append("eICU's microbiology (`microLab`) is **far sparser** than MIMIC's. Requiring the "
             "strict Seymour abx+culture pair collapses suspicion to a fraction of a percent, so "
             "the **primary eICU suspicion = a qualifying systemic antibiotic start** (cultures "
             "refine timing when present). This is a documented deviation; the Sepsis-3 gate "
             "(acute SOFA rise >= 2 within the suspicion window) is UNCHANGED and remains the "
             "strong filter.\n")
    L.append("  | suspicion signal | eICU coverage |")
    L.append("  |---|---|")
    L.append(f"  | has qualifying antibiotic | {cov['has_antibiotic']:.1f}% |")
    L.append(f"  | has ANY culture (microLab) | {cov['has_culture']:.1f}%  ← sparse |")
    L.append(f"  | strict abx+culture pair | {cov['has_strict_pair']:.2f}%  ← unusable as primary |")
    L.append(f"  | **has_suspicion (primary, abx-based)** | **{cov['has_suspicion']:.1f}%** |")

    L.append("\n## 3. Sepsis-3 labels\n")
    L.append(f"- **Sepsis-3 (ever):** {len(septic):,} = **{100*len(septic)/n:.1f}%**")
    L.append(f"- **Onset hour** (since unit admit): median {_fmt_iqr(onset)} h")
    L.append(f"- **Present on admission** (onset <= 0): {present_adm:,} "
             f"({100*present_adm/max(len(septic),1):.1f}% of septic)")
    L.append(f"- **ICU-acquired (onset >= H={H})** — actionable early-warning positives: "
             f"**{len(icu_acq):,}** = {100*len(icu_acq)/n:.1f}% of cohort")

    L.append("\n## 4. Three-way analysis split (no control contamination)\n")
    L.append(f"- **POSITIVE** (onset >= {H} h): **{n_pos:,}** ({100*n_pos/n:.1f}%)")
    L.append(f"- **EXCLUDED** (onset < {H} h / present-on-adm): {n_exc:,} ({100*n_exc/n:.1f}%) — held out")
    L.append(f"- **CONTROL** (never Sepsis-3): {n_ctrl:,} ({100*n_ctrl/n:.1f}%)")
    L.append(f"- partition check: {n_pos:,}+{n_exc:,}+{n_ctrl:,} = {n_pos+n_exc+n_ctrl:,} = N "
             f"{'✓' if n_pos+n_exc+n_ctrl == n else '✗'}")

    if mim:
        L.append("\n## 5. eICU vs MIMIC (full) — side by side\n")
        L.append("  | metric | eICU | MIMIC |")
        L.append("  |---|---|---|")
        L.append(f"  | stays | {n:,} | {mim['n']:,} |")
        L.append(f"  | median age | {df['age'].median():.0f} | {mim['median_age']:.0f} |")
        L.append(f"  | female % | {100*(df['gender']=='F').mean():.1f} | {mim['female']:.1f} |")
        L.append(f"  | in-hosp mortality % | {100*df['hospital_expire_flag'].mean():.1f} | {mim['mortality']:.1f} |")
        L.append(f"  | median LOS h | {df['los_hours'].median():.0f} | {mim['median_los']:.0f} |")
        L.append(f"  | suspicion % | {cov['has_suspicion']:.1f} | {mim['suspicion']:.1f} |")
        L.append(f"  | Sepsis-3 % | {100*len(septic)/n:.1f} | {mim['sepsis']:.1f} |")
        L.append(f"  | ICU-acquired (onset>=H) % | {100*len(icu_acq)/n:.1f} | {mim['icu_acq']:.1f} |")
    else:
        L.append("\n## 5. eICU vs MIMIC\n_(MIMIC full-mode labels not present; run the MIMIC "
                 "pipeline for a side-by-side.)_")

    L.append("\n## 6. SOFA (eICU)\n")
    L.append(f"- Mean total SOFA: **{sofa_mean:.2f}** · hours with SOFA >= 2: "
             f"{100*(sofa['sofa_total']>=2).mean():.1f}%")
    L.append("  | component | mean | % hours > 0 |")
    L.append("  |---|---|---|")
    for c in SOFA_COMPONENTS:
        m, f = comp[c]
        L.append(f"  | {c.replace('sofa_','')} | {m:.2f} | {f:.1f}% |")

    L.append("\n## 7. Real-institution federation preview\n")
    L.append(f"Sites = `hospitalid` (genuine institutions, replacing MIMIC's simulated "
             f"care-unit sites). **{n_sites_200} hospitals with >= 200 stays.** Top sites:\n")
    L.append("  | hospitalid | n | Sepsis-3 % | ICU-acq % |")
    L.append("  |---|---|---|---|")
    for hid, row in site.head(12).iterrows():
        L.append(f"  | {hid} | {int(row['n']):,} | {100*row['sepsis']:.1f}% | {100*row['icu_acq']:.1f}% |")

    # ---- hand-checkable onset validation ----
    L.append("\n## 8. Hand-checkable Sepsis-3 onset cases (SOFA trajectory)\n")
    L.append("For a random sample of POSITIVE stays: SOFA total around onset should rise by "
             ">= 2 from baseline at `onset_hour` (within the suspicion window). Eyeball below.\n")
    pos = df[(df["sepsis3"] == 1) & (df["onset_hour"] >= H)]
    samp = pos.sample(n=min(8, len(pos)), random_state=7) if len(pos) else pos
    sofa_idx = sofa.set_index(["stay_id", "hr"])["sofa_total"]
    for _, r in samp.iterrows():
        sid, on = int(r["stay_id"]), int(r["onset_hour"])
        traj = []
        for h in range(on - 3, on + 2):
            v = sofa_idx.get((sid, h))
            traj.append(f"h{h}={'.' if v is None or pd.isna(v) else int(v)}")
        L.append(f"- stay {sid}: si_hour={int(r['si_hour']) if pd.notna(r['si_hour']) else 'NA'} "
                 f"baseline={r['baseline_sofa']:.0f} onset={on} sofa@onset={r['sofa_at_onset']:.0f} "
                 f"max={r['sofa_max']:.0f} | traj [{'  '.join(traj)}]")

    L.append("\n## 9. Documented limitations (eICU harmonization)\n")
    L.append("- **Culture sparsity** -> antibiotic-based suspicion (Section 2): may include some "
             "prophylactic-antibiotic stays, but the acute-SOFA-rise gate filters non-deteriorators.")
    L.append("- **Vasopressor dose**: eICU embeds the unit in `drugname`; ml/hr (volumetric) "
             "records have no derivable dose -> treated as presence at a conservative SOFA floor.")
    L.append("- **Ventilation**: derived from respiratoryCare ventstart..ventend intervals "
             "(ventend=0 -> observed-status time), an approximation of the true vent window.")
    L.append("- **MAP**: arterial (vitalPeriodic) coalesced with NIBP (vitalAperiodic).")
    L.append("- **Age >89**: eICU '> 89' retained as 90 (MIMIC excludes its >89 band) — a minor, "
             "documented cohort difference; the frozen normalizer handles age 90 in range.")
    L.append("- **All eICU stays are the external split** (no temporal train/val/test); models "
             "are FROZEN MIMIC-trained and only evaluated here.")

    PATHS.reports_root.mkdir(parents=True, exist_ok=True)
    out = PATHS.reports_root / "eicu_report.md"
    out.write_text("\n".join(L), encoding="utf-8")
    log.info("wrote %s", out)
    log.info("  eICU: n=%d | suspicion=%.1f%% | sepsis=%.1f%% | ICU-acq=%d (%.1f%%) | POS=%d",
             n, cov["has_suspicion"], 100*len(septic)/n, len(icu_acq),
             100*len(icu_acq)/n, n_pos)
    return out


def run(cfg: CohortConfig | None = None, lcfg: LabelConfig | None = None) -> None:
    cfg = cfg or CohortConfig(mode="eicu")
    lcfg = lcfg or LabelConfig.load()
    build_report(cfg, lcfg)
