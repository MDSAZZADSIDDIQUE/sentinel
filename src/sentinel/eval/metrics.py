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
def _count_alarm_events(alert: np.ndarray, refractory: int) -> int:
    """Distinct alarm EVENTS: rising edges debounced by a refractory window.

    A sustained alert is one event; a new event only fires on a fresh rise that
    is more than `refractory` hours after the last event (models clinician
    interruptions, not alert-hours)."""
    events, last, prev = 0, -10**9, False
    for t, a in enumerate(alert):
        if a and not prev and (t - last) > refractory:
            events += 1
            last = t
        prev = bool(a)
    return events


@dataclass
class EpisodeSummary:
    stay_id: int
    label: int
    onset_hr: float           # nan for controls
    first_alert_hr: float     # nan if never alerted
    alert_hours: int          # hours with score>=thr
    tp_alert_hours: int       # alert hours that are truly in-window (y=1)
    alarm_events: int         # refractory-debounced distinct events
    n_hours: int


def episode_summaries(stay_ids, hr, score, t_to_onset, ep_label, threshold,
                      alert_window: int = 6, refractory: int = 4) -> list[EpisodeSummary]:
    alert = score >= threshold
    out = []
    for s in np.unique(stay_ids):
        m = stay_ids == s
        hrs, al, lead = hr[m], alert[m], t_to_onset[m]
        label = int(ep_label[m][0])
        onset_hr = float(hrs[0] + lead[0]) if label == 1 and np.isfinite(lead[0]) else float("nan")
        y = (label == 1) & np.isfinite(lead) & (lead >= 1) & (lead <= alert_window)
        fa = hrs[al]
        out.append(EpisodeSummary(
            int(s), label, onset_hr,
            float(fa.min()) if fa.size else float("nan"),
            int(al.sum()), int((al & y).sum()),
            _count_alarm_events(al, refractory), int(m.sum())))
    return out


@dataclass
class ClinicalMetrics:
    sensitivity: float        # positives alerted before onset
    median_lead_h: float      # over detected positives
    fa_rate: float            # controls with any alarm
    alarm_events_per_day: float   # HEADLINE burden: distinct events / patient-day
    alert_hours_per_day: float    # reported alongside (sustained-alert burden)
    ppv: float                # per-hour precision at the operating point
    n_pos: int
    n_ctrl: int


def clinical_metrics(summaries: list[EpisodeSummary]) -> ClinicalMetrics:
    pos = [e for e in summaries if e.label == 1]
    ctrl = [e for e in summaries if e.label == 0]
    detected = [e for e in pos if np.isfinite(e.first_alert_hr) and e.first_alert_hr < e.onset_hr]
    leads = [e.onset_hr - e.first_alert_hr for e in detected]
    fa = [e for e in ctrl if e.alarm_events > 0]
    total_events = sum(e.alarm_events for e in summaries)
    total_alert_h = sum(e.alert_hours for e in summaries)
    total_tp_h = sum(e.tp_alert_hours for e in summaries)
    total_days = sum(e.n_hours for e in summaries) / 24.0
    return ClinicalMetrics(
        sensitivity=len(detected) / len(pos) if pos else float("nan"),
        median_lead_h=float(np.median(leads)) if leads else float("nan"),
        fa_rate=len(fa) / len(ctrl) if ctrl else float("nan"),
        alarm_events_per_day=total_events / total_days if total_days else float("nan"),
        alert_hours_per_day=total_alert_h / total_days if total_days else float("nan"),
        ppv=total_tp_h / total_alert_h if total_alert_h else float("nan"),
        n_pos=len(pos), n_ctrl=len(ctrl),
    )


def no_skill_auprc(y: np.ndarray) -> float:
    """The chance/reference AUPRC = positive rate (essential to read AUPRC against)."""
    return float(np.mean(y)) if len(y) else float("nan")


def threshold_for_budget(stay_ids, hr, score, t_to_onset, ep_label,
                         max_alarms_per_day: float = 8.0, alert_window: int = 6) -> float:
    """Lowest threshold (most sensitive) whose ALARM-EVENT burden stays in budget.

    Event burden decreases with threshold, so the lowest threshold meeting the
    budget is the most-sensitive alarm-fatigue-aware operating point."""
    cand = np.unique(np.round(np.quantile(score, np.linspace(0.5, 0.9995, 120)), 5))
    for thr in sorted(cand):
        cm = clinical_metrics(episode_summaries(stay_ids, hr, score, t_to_onset, ep_label,
                                                thr, alert_window))
        if cm.alarm_events_per_day <= max_alarms_per_day:
            return float(thr)
    return float(max(cand))
