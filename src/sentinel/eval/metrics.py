"""Evaluation metrics for early-warning models.

Hour-level discrimination (AUPRC primary, AUROC, Brier) with bootstrap CIs
resampled over STAYS (not hours — within-patient hours are correlated), plus
episode-level clinical metrics: detection sensitivity, median alert lead time,
control false-alarm rate, and alarm burden (alarms/patient-day). Operating
thresholds are chosen on validation (sensitivity floor or alarm budget).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


@dataclass
class Discrimination:
    auprc: float
    auroc: float
    brier: float
    prevalence: float
    n: int


def discrimination(y: np.ndarray, score: np.ndarray) -> Discrimination:
    y = np.asarray(y); score = np.asarray(score)
    pos = int(y.sum())
    if pos == 0 or pos == len(y):
        return Discrimination(float("nan"), float("nan"), float("nan"), pos / len(y), len(y))
    return Discrimination(
        auprc=float(average_precision_score(y, score)),
        auroc=float(roc_auc_score(y, score)),
        brier=float(brier_score_loss(y, np.clip(score, 0, 1))),
        prevalence=pos / len(y),
        n=len(y),
    )


def bootstrap_ci(y: np.ndarray, score: np.ndarray, stay_ids: np.ndarray,
                 metric: str = "auprc", n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% CI resampling stays with replacement."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(stay_ids)
    by_stay = {s: np.where(stay_ids == s)[0] for s in uniq}
    fn = {"auprc": average_precision_score, "auroc": roc_auc_score}[metric]
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_stay[s] for s in pick])
        yb, sb = y[idx], score[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        vals.append(fn(yb, sb))
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def reliability(y: np.ndarray, score: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Calibration curve: per-bin mean predicted vs observed frequency."""
    score = np.clip(score, 0, 1)
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for i in range(n_bins):
        m = (score >= edges[i]) & (score < edges[i + 1] if i < n_bins - 1 else score <= edges[i + 1])
        if m.sum() == 0:
            continue
        out.append({"bin": i, "n": int(m.sum()), "pred": float(score[m].mean()),
                    "obs": float(y[m].mean())})
    return out


# ----------------------------- episode level -----------------------------
@dataclass
class EpisodeSummary:
    stay_id: int
    label: int
    onset_hr: float           # nan for controls
    first_alert_hr: float     # nan if never alerted
    alert_hours: int
    n_hours: int


def episode_summaries(stay_ids, hr, score, t_to_onset, ep_label, threshold) -> list[EpisodeSummary]:
    alert = score >= threshold
    out = []
    for s in np.unique(stay_ids):
        m = stay_ids == s
        hrs, al = hr[m], alert[m]
        lead = t_to_onset[m]
        label = int(ep_label[m][0])
        onset_hr = float(hrs[0] + lead[0]) if label == 1 and np.isfinite(lead[0]) else float("nan")
        fa = hrs[al]
        first = float(fa.min()) if fa.size else float("nan")
        out.append(EpisodeSummary(int(s), label, onset_hr, first, int(al.sum()), int(m.sum())))
    return out


@dataclass
class ClinicalMetrics:
    sensitivity: float        # positives alerted before onset
    median_lead_h: float      # over detected positives
    fa_rate: float            # controls with any alarm
    alarms_per_day: float     # alert-hours per patient-day
    n_pos: int
    n_ctrl: int


def clinical_metrics(summaries: list[EpisodeSummary]) -> ClinicalMetrics:
    pos = [e for e in summaries if e.label == 1]
    ctrl = [e for e in summaries if e.label == 0]
    detected = [e for e in pos if np.isfinite(e.first_alert_hr) and e.first_alert_hr < e.onset_hr]
    leads = [e.onset_hr - e.first_alert_hr for e in detected]
    fa = [e for e in ctrl if e.alert_hours > 0]
    total_alert_h = sum(e.alert_hours for e in summaries)
    total_days = sum(e.n_hours for e in summaries) / 24.0
    return ClinicalMetrics(
        sensitivity=len(detected) / len(pos) if pos else float("nan"),
        median_lead_h=float(np.median(leads)) if leads else float("nan"),
        fa_rate=len(fa) / len(ctrl) if ctrl else float("nan"),
        alarms_per_day=total_alert_h / total_days if total_days else float("nan"),
        n_pos=len(pos), n_ctrl=len(ctrl),
    )


def threshold_for_sensitivity(stay_ids, hr, score, t_to_onset, ep_label,
                              target: float = 0.85) -> float:
    """Lowest threshold (least alarms) achieving >= target episode sensitivity on this split."""
    cand = np.unique(np.round(np.quantile(score, np.linspace(0.5, 0.999, 100)), 4))
    best = 0.0
    for thr in sorted(cand):
        cm = clinical_metrics(episode_summaries(stay_ids, hr, score, t_to_onset, ep_label, thr))
        if cm.sensitivity >= target:
            best = float(thr)
    return best


def threshold_for_budget(stay_ids, hr, score, t_to_onset, ep_label,
                         max_alarms_per_day: float = 8.0) -> float:
    """Lowest threshold (most sensitive) whose alarm burden stays within budget.

    Alarms/day decreases monotonically with threshold, so the lowest threshold
    meeting the budget gives the most-sensitive operating point at that alarm
    burden — the alarm-fatigue-aware choice the clinical framing calls for.
    """
    cand = np.unique(np.round(np.quantile(score, np.linspace(0.5, 0.9995, 120)), 5))
    for thr in sorted(cand):
        cm = clinical_metrics(episode_summaries(stay_ids, hr, score, t_to_onset, ep_label, thr))
        if cm.alarms_per_day <= max_alarms_per_day:
            return float(thr)
    return float(max(cand))
