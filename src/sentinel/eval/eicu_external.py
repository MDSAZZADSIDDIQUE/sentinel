"""Phase 9 Step 5: external validation of FROZEN MIMIC-trained models on eICU.

Each model is trained ONCE on the MIMIC full `train` split and then scored,
WITHOUT any further gradient step, on (a) the MIMIC `test` split (internal
reference) and (b) the eICU `external` split. Frozen-ness is structural: eICU
data only ever reaches predict_* / team_scores (all @torch.no_grad + .eval());
tests/test_eicu_frozen.py asserts weights are unchanged by external scoring.

Operating point on eICU is reported BOTH ways the spec requires:
  * transfer  — threshold chosen on MIMIC val, applied to eICU (calibration transfer)
  * recal     — threshold re-chosen on eICU itself (secondary)
AUPRC (rank-based) is threshold-independent and read against the no-skill line.

Models: SENTINEL (organ-decomposed MAPPO; team risk = max-organ P(escalate)),
GRU-single, JointEnsemble-6, XGBoost, NEWS2.

Output: outputs/reports/eicu_external.md
"""
from __future__ import annotations

import dataclasses
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ..config import CohortConfig, FeatureConfig, MARLConfig
from ..logging_utils import get_logger
from ..paths import PATHS
from ..features.dataset import load_hourly, load_split
from . import metrics as M

log = get_logger("eval.eicu_external")

ALL_MODELS = ("NEWS2", "XGBoost", "GRU-single", "JointEnsemble-6", "SENTINEL")


def _stay_blocks(stay_ids: np.ndarray) -> list[np.ndarray]:
    """Row indices grouped by stay (O(N log N) once) for fast block-bootstrap."""
    order = np.argsort(stay_ids, kind="stable")
    _, starts = np.unique(stay_ids[order], return_index=True)
    bounds = np.append(starts, len(stay_ids))
    return [order[bounds[i]:bounds[i + 1]] for i in range(len(starts))]


def _fast_ci(y, score, stay_ids, n_boot=200, seed=0, metric="auprc") -> tuple[float, float]:
    """95% CI resampling STAYS with replacement; fast for the 122k-stay eICU set."""
    blocks = _stay_blocks(stay_ids)
    fn = average_precision_score if metric == "auprc" else roc_auc_score
    rng = np.random.default_rng(seed)
    nb = len(blocks)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, nb, size=nb)
        idx = np.concatenate([blocks[i] for i in pick])
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        vals.append(fn(yb, score[idx]))
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def _fast_summaries(blocks, stay_ids, hr, score, t_to_onset, ep_label, threshold,
                    alert_window=6, refractory=4):
    """Block-based episode summaries — O(rows), not O(stays x rows) (M.episode_summaries
    rescans the full array per stay, which is intractable for 122k eICU stays)."""
    alert = score >= threshold
    out = []
    for idx in blocks:
        hrs, al, lead = hr[idx], alert[idx], t_to_onset[idx]
        label = int(ep_label[idx][0])
        onset_hr = float(hrs[0] + lead[0]) if label == 1 and np.isfinite(lead[0]) else float("nan")
        y = (label == 1) & np.isfinite(lead) & (lead >= 1) & (lead <= alert_window)
        fa = hrs[al]
        out.append(M.EpisodeSummary(
            int(stay_ids[idx][0]), label, onset_hr,
            float(fa.min()) if fa.size else float("nan"),
            int(al.sum()), int((al & y).sum()),
            M._count_alarm_events(al, refractory), int(len(idx))))
    return out


def _fast_threshold(blocks, stay_ids, hr, score, t_to_onset, ep_label,
                    budget, alert_window=6) -> float:
    """Lowest threshold whose alarm-event burden stays within budget.

    Burden decreases monotonically with threshold, so binary-search the candidate
    grid (~7 evals) instead of a 120-step linear scan — essential at eICU scale."""
    cand = sorted(np.unique(np.round(np.quantile(score, np.linspace(0.5, 0.9995, 120)), 5)))

    def burden(thr):
        return M.clinical_metrics(_fast_summaries(blocks, stay_ids, hr, score, t_to_onset,
                                                  ep_label, thr, alert_window)).alarm_events_per_day

    lo, hi, ans = 0, len(cand) - 1, float(cand[-1])
    while lo <= hi:
        mid = (lo + hi) // 2
        if burden(cand[mid]) <= budget:
            ans = float(cand[mid]); hi = mid - 1     # try a lower (more sensitive) threshold
        else:
            lo = mid + 1
    return ans


def _eval(data, score, thr, blocks, n_boot=200) -> dict:
    """Discrimination (+ fast CI) and clinical metrics at a precomputed threshold."""
    d = M.discrimination(data.y, score)
    lo, hi = _fast_ci(data.y, score, data.stay_ids, n_boot=n_boot)
    cm = M.clinical_metrics(_fast_summaries(blocks, data.stay_ids, data.hr, score,
                                            data.t_to_onset, data.ep_label, thr))
    return {"auprc": d.auprc, "auprc_lo": lo, "auprc_hi": hi, "auroc": d.auroc,
            "brier": d.brier, "prev": d.prevalence, "no_skill": M.no_skill_auprc(data.y),
            "sens": cm.sensitivity, "lead": cm.median_lead_h, "ppv": cm.ppv,
            "events_day": cm.alarm_events_per_day, "fa": cm.fa_rate, "thr": thr}


def _agg(rows: list[dict]) -> dict:
    return {k: (float(np.nanmean([r[k] for r in rows])),
                float(np.nanstd([r[k] for r in rows]))) for k in rows[0]}


# --- per-model scorers: return (score_mimic_val, score_mimic_test, score_eicu_ext) ---
def _score_news(df_m, df_e, full_cfg, eicu_cfg, seed):
    from ..baselines.news import load_norm_stats, news_score
    ns = load_norm_stats(full_cfg)
    return (news_score(df_m[df_m["split"] == "val"], ns),
            news_score(df_m[df_m["split"] == "test"], ns),
            news_score(df_e[df_e["split"] == "external"], ns))


def _score_xgb(df_m, df_e, full_cfg, eicu_cfg, seed):
    from ..baselines.xgb import predict_xgb, train_xgb
    tr = load_split(full_cfg, "train", df=df_m)
    model = train_xgb(tr, seed=seed)
    va = load_split(full_cfg, "val", df=df_m)
    te = load_split(full_cfg, "test", df=df_m)
    ext = load_split(eicu_cfg, "external", df=df_e)
    return predict_xgb(model, va), predict_xgb(model, te), predict_xgb(model, ext)


def _score_gru(df_m, df_e, full_cfg, eicu_cfg, seed):
    from ..baselines.gru import predict_gru, train_gru
    tr = load_split(full_cfg, "train", df=df_m)
    pw = (len(tr.y) - int(tr.y.sum())) / max(int(tr.y.sum()), 1)
    cols = tr.feature_names
    gm = train_gru(df_m, cols, pw, seed=seed)
    return (predict_gru(gm, df_m, "val", cols), predict_gru(gm, df_m, "test", cols),
            predict_gru(gm, df_e, "external", cols))


def _score_jointens(df_m, df_e, full_cfg, eicu_cfg, seed):
    from ..baselines.gru import predict_joint_ensemble, train_joint_ensemble
    tr = load_split(full_cfg, "train", df=df_m)
    pw = (len(tr.y) - int(tr.y.sum())) / max(int(tr.y.sum()), 1)
    cols = tr.feature_names
    em = train_joint_ensemble(df_m, cols, pw, seed=seed, n=6)
    return (predict_joint_ensemble(em, df_m, "val", cols),
            predict_joint_ensemble(em, df_m, "test", cols),
            predict_joint_ensemble(em, df_e, "external", cols))


def _marl_scores_chunked(policy, df_sub, manifest, chunk_stays=4000):
    """team_scores in stay-chunks: EpisodeTensors pads ALL stays and moves them to
    the 4GB GPU at once — eICU's 122k stays would OOM. Chunking by consecutive
    (already stay_id,hr-sorted) stays preserves row order, so scores still align
    with load_split."""
    from ..agents.mappo import EpisodeTensors, team_scores
    sids = df_sub["stay_id"].drop_duplicates().to_numpy()
    parts = []
    for i in range(0, len(sids), chunk_stays):
        sub = df_sub[df_sub["stay_id"].isin(set(sids[i:i + chunk_stays].tolist()))]
        parts.append(team_scores(policy, EpisodeTensors(sub, manifest, "full")))
    return np.concatenate(parts).astype(np.float32)


def _score_sentinel(df_m, df_e, full_cfg, eicu_cfg, seed, marl_updates):
    from ..agents.mappo import EpisodeTensors, team_scores, train_marl
    mc = MARLConfig.load()
    mc = dataclasses.replace(mc, algo="mappo", ablation="full", seed=seed,
                             updates=marl_updates or mc.updates)
    policy, _, manifest = train_marl(full_cfg, mc, df=df_m)
    sv = team_scores(policy, EpisodeTensors(df_m[df_m["split"] == "val"], manifest, "full"))
    st = team_scores(policy, EpisodeTensors(df_m[df_m["split"] == "test"], manifest, "full"))
    se = _marl_scores_chunked(policy, df_e[df_e["split"] == "external"], manifest)
    return sv, st, se


_SCORERS = {"NEWS2": _score_news, "XGBoost": _score_xgb, "GRU-single": _score_gru,
            "JointEnsemble-6": _score_jointens, "SENTINEL": _score_sentinel}


def run(eicu_cfg: CohortConfig | None = None, full_cfg: CohortConfig | None = None,
        fcfg: FeatureConfig | None = None, seeds=(0, 1, 2), alarm_budget: float = 8.0,
        models=ALL_MODELS, marl_updates: int | None = None, n_boot: int = 200) -> None:
    eicu_cfg = eicu_cfg or CohortConfig(mode="eicu")
    full_cfg = full_cfg or CohortConfig(mode="full")
    fcfg = fcfg or FeatureConfig.load()
    t0 = time.perf_counter()
    df_m = load_hourly(full_cfg)
    df_e = load_hourly(eicu_cfg)
    val = load_split(full_cfg, "val", df=df_m)
    test = load_split(full_cfg, "test", df=df_m)
    ext = load_split(eicu_cfg, "external", df=df_e)
    log.info("external validation: MIMIC val=%d test=%d | eICU external=%d rows (%d stays)",
             len(val.y), len(test.y), len(ext.y), len(np.unique(ext.stay_ids)))
    log.info("  no-skill AUPRC: MIMIC test=%.3f | eICU external=%.3f",
             M.no_skill_auprc(test.y), M.no_skill_auprc(ext.y))

    val_blocks = _stay_blocks(val.stay_ids)
    test_blocks = _stay_blocks(test.stay_ids)
    ext_blocks = _stay_blocks(ext.stay_ids)

    # (model, target) -> aggregated metrics; target in {test, eicu_transfer, eicu_recal}
    results: dict[tuple, dict] = {}
    for name in models:
        det = name == "NEWS2"               # deterministic -> single "seed"
        use_seeds = (0,) if det else seeds
        rows = {"test": [], "eicu_transfer": [], "eicu_recal": []}
        for seed in use_seeds:
            try:
                kw = {"marl_updates": marl_updates} if name == "SENTINEL" else {}
                s_val, s_test, s_ext = _SCORERS[name](df_m, df_e, full_cfg, eicu_cfg, seed, **kw)
                thr_transfer = _fast_threshold(val_blocks, val.stay_ids, val.hr, s_val,
                                               val.t_to_onset, val.ep_label, alarm_budget)
                thr_recal = _fast_threshold(ext_blocks, ext.stay_ids, ext.hr, s_ext,
                                            ext.t_to_onset, ext.ep_label, alarm_budget)
                rows["test"].append(_eval(test, s_test, thr_transfer, test_blocks, n_boot))
                # eICU transfer: full discrimination + CI computed once on the external score
                ext_transfer = _eval(ext, s_ext, thr_transfer, ext_blocks, n_boot)
                rows["eicu_transfer"].append(ext_transfer)
                # eICU recal: SAME score -> identical AUPRC/AUROC/CI; only the operating
                # point (clinical metrics) changes -> reuse discrimination, recompute clinical
                ext_recal = dict(ext_transfer)
                cm = M.clinical_metrics(_fast_summaries(ext_blocks, ext.stay_ids, ext.hr, s_ext,
                                                        ext.t_to_onset, ext.ep_label, thr_recal))
                ext_recal.update({"sens": cm.sensitivity, "ppv": cm.ppv, "lead": cm.median_lead_h,
                                  "events_day": cm.alarm_events_per_day, "fa": cm.fa_rate, "thr": thr_recal})
                rows["eicu_recal"].append(ext_recal)
            except Exception as exc:  # noqa: BLE001 — one model's failure must not lose the run
                log.warning("  %s seed=%s FAILED: %s: %s", name, seed, type(exc).__name__, exc)
        if not rows["eicu_transfer"]:
            log.warning("  %-16s no successful seeds — skipping", name)
            continue
        for tgt in rows:
            results[(name, tgt)] = _agg(rows[tgt])
        log.info("  %-16s done | eICU AUPRC=%.3f [%.3f-%.3f] (no-skill %.3f) | MIMIC-test AUPRC=%.3f",
                 name, results[(name, "eicu_transfer")]["auprc"][0],
                 results[(name, "eicu_transfer")]["auprc_lo"][0],
                 results[(name, "eicu_transfer")]["auprc_hi"][0],
                 results[(name, "eicu_transfer")]["no_skill"][0],
                 results[(name, "test")]["auprc"][0])

    _write_report(results, models, seeds, alarm_budget, t0)
    log.info("external validation done in %.1fs", time.perf_counter() - t0)


def _write_report(results, models, seeds, alarm_budget, t0):
    def row(model, tgt):
        m = results.get((model, tgt))
        if not m:
            return None
        return (f"| {model} | {m['auprc'][0]:.3f} [{m['auprc_lo'][0]:.3f}-{m['auprc_hi'][0]:.3f}] | "
                f"{m['auroc'][0]:.3f} | {m['sens'][0]:.2f} | {m['ppv'][0]:.2f} | "
                f"{m['lead'][0]:.1f} | {m['events_day'][0]:.1f} | {m['fa'][0]:.2f} |")

    L = ["# SENTINEL — eICU External Validation: frozen MIMIC models\n",
         f"_Generated {time.strftime('%Y-%m-%d %H:%M')}. Models trained on MIMIC `train`, "
         f"FROZEN, scored on eICU `external` (no gradient step touches eICU). AUPRC primary "
         f"(95% CI over stays); operating point at ≤ {alarm_budget:.0f} alarm events/patient-day; "
         f"{len(seeds)} seeds mean±std._\n"]
    ns_test = next((m["no_skill"][0] for (mm, t), m in results.items() if t == "test"), float("nan"))
    ns_eicu = next((m["no_skill"][0] for (mm, t), m in results.items() if t == "eicu_transfer"), float("nan"))

    L.append(f"\n## 1. Internal (MIMIC test) — reference  ·  no-skill AUPRC ≈ **{ns_test:.3f}**\n")
    L.append("| Model | AUPRC | AUROC | Sens | PPV | Lead h | Events/day | FA |")
    L.append("|---|---|---|---|---|---|---|---|")
    for model in models:
        r = row(model, "test")
        if r:
            L.append(r)

    L.append(f"\n## 2. External (eICU) — threshold TRANSFERRED from MIMIC val  ·  no-skill ≈ **{ns_eicu:.3f}**\n")
    L.append("_The honest external operating point: MIMIC-chosen threshold applied as-is to eICU._\n")
    L.append("| Model | AUPRC | AUROC | Sens | PPV | Lead h | Events/day | FA |")
    L.append("|---|---|---|---|---|---|---|---|")
    for model in models:
        r = row(model, "eicu_transfer")
        if r:
            L.append(r)

    L.append("\n## 3. External (eICU) — threshold RE-CALIBRATED on eICU (secondary)\n")
    L.append("_Operating point re-chosen on eICU; isolates calibration drift from discrimination._\n")
    L.append("| Model | AUPRC | AUROC | Sens | PPV | Lead h | Events/day | FA |")
    L.append("|---|---|---|---|---|---|---|---|")
    for model in models:
        r = row(model, "eicu_recal")
        if r:
            L.append(r)

    L.append("\n## 4. Internal → external AUPRC (does discrimination hold?)\n")
    L.append("| Model | MIMIC-test AUPRC | eICU AUPRC | Δ |")
    L.append("|---|---|---|---|")
    for model in models:
        ti, te = results.get((model, "test")), results.get((model, "eicu_transfer"))
        if ti and te:
            L.append(f"| {model} | {ti['auprc'][0]:.3f} | {te['auprc'][0]:.3f} | "
                     f"{te['auprc'][0]-ti['auprc'][0]:+.3f} |")
    L.append(f"\n_eICU external positive-hour prevalence ≈ {100*ns_eicu:.1f}% (the no-skill line); "
             f"read AUPRC against it. Discrimination (AUROC) is prevalence-independent and the "
             f"cleaner cross-dataset comparison. Calibration (transfer vs recal operating points) "
             f"shows threshold drift; AUPRC/AUROC are unchanged between them (rank-based)._\n")

    PATHS.reports_root.mkdir(parents=True, exist_ok=True)
    out = PATHS.reports_root / "eicu_external.md"
    out.write_text("\n".join(L), encoding="utf-8")
    log.info("  wrote %s", out)
