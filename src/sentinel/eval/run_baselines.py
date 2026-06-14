"""Phase 3 baseline runner: NEWS + XGBoost across the measurement ablation.

For each model x {full, physiology-only}, trains (learned models, multi-seed),
picks the operating threshold on VAL at a clinical ALARM BUDGET (alarms/patient-
day; the alarm-fatigue-aware operating point), and reports discrimination (AUPRC
primary + bootstrap CI) and clinical metrics on internal (val/test) and the
external CVICU split. Results -> outputs/reports/baselines_<mode>.md.
"""
from __future__ import annotations

import time

import numpy as np

from ..config import CohortConfig, FeatureConfig
from ..logging_utils import get_logger
from ..paths import PATHS
from ..features.dataset import ABLATIONS, load_hourly, load_split
from . import metrics as M

log = get_logger("eval.baselines")


def _eval_split(data, score, thr_arrays, budget, n_boot=500):
    d = M.discrimination(data.y, score)
    lo, hi = M.bootstrap_ci(data.y, score, data.stay_ids, "auprc", n_boot=n_boot)
    thr = M.threshold_for_budget(*thr_arrays, max_alarms_per_day=budget)
    cm = M.clinical_metrics(M.episode_summaries(
        data.stay_ids, data.hr, score, data.t_to_onset, data.ep_label, thr))
    return {"auprc": d.auprc, "auprc_lo": lo, "auprc_hi": hi, "auroc": d.auroc,
            "brier": d.brier, "prev": d.prevalence, "no_skill": M.no_skill_auprc(data.y),
            "sens": cm.sensitivity, "lead": cm.median_lead_h, "ppv": cm.ppv,
            "events_day": cm.alarm_events_per_day, "alert_h_day": cm.alert_hours_per_day,
            "fa": cm.fa_rate}


def _agg(rows: list[dict]) -> dict:
    out = {}
    for k in rows[0]:
        vals = np.array([r[k] for r in rows], dtype=float)
        out[k] = (float(np.nanmean(vals)), float(np.nanstd(vals)))
    return out


def run(cfg: CohortConfig | None = None, fcfg: FeatureConfig | None = None,
        seeds=(0, 1, 2), alarm_budget: float = 8.0, skip_torch: bool = False) -> None:
    cfg = cfg or CohortConfig.load()
    fcfg = fcfg or FeatureConfig.load()
    t0 = time.perf_counter()
    df = load_hourly(cfg)
    splits = [s for s in ("val", "test", "external") if (df["split"] == s).any()]
    log.info("baselines (mode=%s): splits=%s, seeds=%s", cfg.mode, splits, list(seeds))

    results: dict[tuple, dict] = {}

    # ---- NEWS2 (physiology-only, deterministic) ----
    from ..baselines.news import load_norm_stats, news_score
    ns = load_norm_stats(cfg)
    val = load_split(cfg, "val", df=df)
    val_scores = news_score(df[df["split"] == "val"], ns)
    for s in splits:
        data = load_split(cfg, s, df=df)
        sc = news_score(df[df["split"] == s], ns)
        thr_arrays = (val.stay_ids, val.hr, val_scores, val.t_to_onset, val.ep_label)
        results[("NEWS2", "physiology", s)] = _agg([_eval_split(data, sc, thr_arrays, alarm_budget)])
    log.info("  NEWS2 done")

    # ---- XGBoost (three-way ablation: full / no_measurement / physiology) ----
    from ..baselines.xgb import predict_xgb, train_xgb
    for ablation in ABLATIONS:
        tr = load_split(cfg, "train", df=df, ablation=ablation)
        va = load_split(cfg, "val", df=df, ablation=ablation)
        per_split_rows: dict[str, list] = {s: [] for s in splits}
        for seed in seeds:
            model = train_xgb(tr, seed=seed)
            thr_arrays = (va.stay_ids, va.hr, predict_xgb(model, va), va.t_to_onset, va.ep_label)
            for s in splits:
                data = load_split(cfg, s, df=df, ablation=ablation)
                per_split_rows[s].append(_eval_split(data, predict_xgb(model, data),
                                                     thr_arrays, alarm_budget))
        for s in splits:
            results[("XGBoost", ablation, s)] = _agg(per_split_rows[s])
        log.info("  XGBoost[%s] done", ablation)

    if not skip_torch:
        from ..baselines.gru import (predict_gru, predict_organ_ensemble,
                                     train_gru, train_organ_ensemble)
        # ---- GRU single-agent (three-way ablation) ----
        for ablation in ABLATIONS:
            tr = load_split(cfg, "train", df=df, ablation=ablation)
            va = load_split(cfg, "val", df=df, ablation=ablation)
            pw = (len(tr.y) - int(tr.y.sum())) / max(int(tr.y.sum()), 1)
            rows = {s: [] for s in splits}
            for seed in seeds:
                gm = train_gru(df, tr.feature_names, pw, seed=seed)
                gthr = (va.stay_ids, va.hr, predict_gru(gm, df, "val", tr.feature_names),
                        va.t_to_onset, va.ep_label)
                for s in splits:
                    data = load_split(cfg, s, df=df, ablation=ablation)
                    rows[s].append(_eval_split(data, predict_gru(gm, df, s, tr.feature_names),
                                               gthr, alarm_budget))
            for s in splits:
                results[("GRU-single", ablation, s)] = _agg(rows[s])
            log.info("  GRU-single[%s] done", ablation)
        # ---- independent organ-ensemble (full + physiology) ----
        for ablation in ("full", "physiology"):
            tr = load_split(cfg, "train", df=df, ablation=ablation)
            va = load_split(cfg, "val", df=df, ablation=ablation)
            pw = (len(tr.y) - int(tr.y.sum())) / max(int(tr.y.sum()), 1)
            rows = {s: [] for s in splits}
            for seed in seeds:
                em = train_organ_ensemble(df, tr.manifest, pw, seed=seed)
                ethr = (va.stay_ids, va.hr, predict_organ_ensemble(em, df, "val"),
                        va.t_to_onset, va.ep_label)
                for s in splits:
                    data = load_split(cfg, s, df=df, ablation=ablation)
                    rows[s].append(_eval_split(data, predict_organ_ensemble(em, df, s),
                                               ethr, alarm_budget))
            for s in splits:
                results[("OrganEnsemble-indep", ablation, s)] = _agg(rows[s])
            log.info("  OrganEnsemble-indep[%s] done", ablation)

    _write_report(cfg, results, splits, seeds, alarm_budget)
    log.info("baselines done in %.1fs", time.perf_counter() - t0)


def _write_report(cfg, results, splits, seeds, alarm_budget):
    L = [f"# SENTINEL — Phase 3 baselines ({cfg.mode} cohort)\n",
         f"_AUPRC primary (95% CI bootstrapped over stays); operating threshold set on val "
         f"at an alarm budget of ≤ {alarm_budget:.0f} alarms/patient-day; "
         f"{len(seeds)} seed(s) mean±std. Sens/Lead reported at that operating point._\n",
         "**Measurement ablation:** `full` = physiology + measurement/timing channels; "
         "`physiology` = physiology values + demographics only (drops `__measured`/`__mask`/"
         "`hour_idx`). The full−physiology gap = reliance on clinician behavior/timing.\n"]
    L.append("\n_Burden unit = **distinct alarm events/patient-day** (rising edge + 4h "
             "refractory; the interruption count); alert-hours/day reported alongside. "
             "PPV = per-hour precision at the operating point. AUPRC read against the "
             "no-skill line (= positive rate)._\n")
    for s in splits:
        L.append(f"\n## Split: {s}\n")
        ns = next((m["no_skill"][0] for (mm, a, sp), m in results.items() if sp == s), float("nan"))
        L.append(f"_No-skill AUPRC (chance) ≈ **{ns:.3f}**._\n")
        L.append("| Model | Ablation | AUPRC | AUROC | PPV | Sens | Lead h | Events/day | AlertH/day | FA |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for (model, abl, sp), m in results.items():
            if sp != s:
                continue
            ap = m["auprc"][0]; lo = m["auprc_lo"][0]; hi = m["auprc_hi"][0]
            L.append(f"| {model} | {abl} | {ap:.3f} [{lo:.3f}–{hi:.3f}] | "
                     f"{m['auroc'][0]:.3f} | {m['ppv'][0]:.2f} | {m['sens'][0]:.2f} | "
                     f"{m['lead'][0]:.1f} | {m['events_day'][0]:.1f} | "
                     f"{m['alert_h_day'][0]:.1f} | {m['fa'][0]:.2f} |")
    prev = next(iter(results.values()))["prev"][0]
    L.append(f"\n_Positive-hour prevalence ≈ {100*prev:.1f}%. Internal test is small "
             f"(few hundred positive episodes) — lean on the external CVICU split and the "
             f"CIs; don't over-read small between-model gaps. The full−no_measurement gap = "
             f"clinician-behavior leak; no_measurement−physiology = timing prior._\n")
    PATHS.reports_root.mkdir(parents=True, exist_ok=True)
    out = PATHS.reports_root / f"baselines_{cfg.mode}.md"
    out.write_text("\n".join(L), encoding="utf-8")
    log.info("  wrote %s", out)
