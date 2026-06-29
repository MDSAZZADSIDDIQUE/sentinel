"""Generate the paper's headline figures from research_paper/figures/results.json.

Reproducible without re-running models (the numbers are committed). Saves vector
PDF + PNG into research_paper/figures/. matplotlib only (no seaborn).
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..logging_utils import get_logger
from ..paths import PATHS

log = get_logger("viz")
FIGDIR = PATHS.figures_root
SENTINEL_C = "#d62728"
# Canonical per-organ colors -- identical to the replay dashboard for consistency.
ORGAN_COLORS = {
    "respiratory": "#1f77b4", "coagulation": "#9467bd", "hepatic": "#8c564b",
    "cardiovascular": "#d62728", "neurologic": "#2ca02c", "renal": "#ff7f0e",
}
# Display names: results.json keeps the original keys; the paper uses the
# SENTINEL-Risk / SENTINEL-Alarm split, so remap row/legend labels at render time.
DISPLAY = {"SENTINEL (organ)": "SENTINEL-Risk", "SENTINEL-MAPPO": "SENTINEL-Alarm"}
# pdf/ps.fonttype 42 => embed TrueType (not Type 3) fonts: required for IEEE Xplore
# PDF compliance (matplotlib defaults to Type 3, which PDF eXpress rejects).
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42})


def _load():
    with (FIGDIR / "results.json").open() as fh:
        return json.load(fh)


def _save(fig, name):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    log.info("  wrote %s.{pdf,png}", name)


def fig_discrimination(d):
    models = ["NEWS2", "XGBoost", "GRU-single", "JointEnsemble-6", "SENTINEL (organ)", "SENTINEL-MAPPO"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    y = list(range(len(models)))
    for ax, split in zip(axes, ("test", "external")):
        vals = [d["discrimination"][m][split] for m in models]
        ap = [v[0] for v in vals]
        err = [[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]]
        colors = [SENTINEL_C if "SENTINEL (organ)" == m else "#888" for m in models]
        ax.barh(y, ap, xerr=err, color=colors, height=0.62, error_kw={"lw": 1})
        ax.axvline(d["no_skill"][split], ls="--", color="k", lw=1, label="no-skill")
        ax.set_xlabel("AUPRC"); ax.set_title("internal test" if split == "test" else "external (CVICU)")
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([DISPLAY.get(m, m) for m in models])
    axes[1].tick_params(labelleft=False)
    axes[0].invert_yaxis()
    fig.suptitle("Discrimination: organ-decomposed matches/beats monolithic", fontsize=11)
    _save(fig, "fig_discrimination")


def fig_robustness(d):
    per = d["robustness_per_organ_external"]
    organs = ["none", "respiratory", "cardiovascular", "neurologic", "renal", "coagulation", "hepatic"]
    # GRU-masked = monolithic GRU trained WITH organ-dropout augmentation (the
    # missingness-aware control): orange, distinct from the plain monolith (grey).
    models = ["SENTINEL (organ)", "JointEnsemble-6", "GRU-single", "GRU-masked"]
    colors = {"SENTINEL (organ)": SENTINEL_C, "JointEnsemble-6": "#1f77b4",
              "GRU-single": "#888", "GRU-masked": "#ff7f0e"}
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    x = range(len(organs)); w = 0.20
    for i, m in enumerate(models):
        ax.bar([xi + (i - 1.5) * w for xi in x], [per[o][m] for o in organs], w,
               label=DISPLAY.get(m, m), color=colors[m])
    ax.set_xticks(list(x)); ax.set_xticklabels(["(none)"] + organs[1:], rotation=30, ha="right")
    ax.set_ylabel("external AUPRC"); ax.set_xlabel("organ blinded at test time")
    ax.set_ylim(0, 0.36)
    ax.set_title("Missing-signal robustness is structural, not just missingness-aware training")
    ax.legend(fontsize=8, ncol=4, loc="upper center")
    _save(fig, "fig_robustness")


def fig_ablation(d):
    ab = d["ablation_test_auprc"]
    conds = ["full", "no_measurement", "physiology"]
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    x = range(len(conds)); w = 0.36
    for i, m in enumerate(("XGBoost", "GRU-single")):
        ax.bar([xi + (i - 0.5) * w for xi in x], [ab[m][c] for c in conds], w, label=m,
               color=["#1f77b4", "#888"][i])
    ax.set_xticks(list(x)); ax.set_xticklabels(["full", "−measurement\n(keep timing)", "physiology\nonly"])
    ax.set_ylabel("test AUPRC")
    ax.set_title("Non-physiology signal is the timing prior,\nnot the measurement/ordering leak")
    ax.legend(fontsize=8)
    _save(fig, "fig_ablation")


def _load_attr():
    with (FIGDIR / "attribution_example.json").open() as fh:
        return json.load(fh)


def fig_attribution(ex):
    """Per-patient interpretability: the six per-organ onset-risk trajectories for one
    ICU stay, the team max-combine, the first alarm, and the true Sepsis-3 onset. The
    organ that drives the alert is emphasized -- the decomposition's value over an
    opaque single score. Source: attribution_example.json (built by
    scripts/make_attribution_example.py)."""
    import numpy as np
    hrs = np.asarray(ex["hours"], dtype=float)
    organs = {o: np.asarray(v, dtype=float) for o, v in ex["organs"].items()}
    team = np.asarray(ex["team"], dtype=float)
    onset, thr = ex["onset_hr"], ex["threshold"]
    alarm = ex.get("first_alarm_hr")
    ref = alarm if alarm is not None else onset
    driver = max(organs, key=lambda o: organs[o][int(np.argmin(np.abs(hrs - ref)))])

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for o, y in organs.items():
        lead = o == driver
        ax.plot(hrs, y, color=ORGAN_COLORS[o], lw=2.2 if lead else 1.2,
                alpha=0.95 if lead else 0.45, zorder=4 if lead else 2, label=o)
    ax.plot(hrs, team, color="black", lw=2.2, alpha=0.85, zorder=5,
            label="team (max-combine)")
    ax.axhline(thr, ls="--", color="gray", lw=1)
    ax.text(hrs[0], thr + 0.02, "alarm threshold", fontsize=7, color="gray")
    ax.axvline(onset, color=SENTINEL_C, lw=2, zorder=3)
    ax.text(onset, 0.99, " Sepsis-3 onset", color=SENTINEL_C, fontsize=8,
            ha="left", va="top", rotation=90)
    ax.text(hrs[-1], organs[driver][-1], f" {driver}", color=ORGAN_COLORS[driver],
            fontsize=8, fontweight="bold", va="center")
    if alarm is not None:
        ai = int(np.argmin(np.abs(hrs - alarm)))
        ax.scatter([alarm], [team[ai]], color="black", s=26, zorder=6)
        ax.annotate(f"first alarm (hr {int(alarm)})", (alarm, team[ai]),
                    textcoords="offset points", xytext=(5, 9), fontsize=7.5)
        ax.annotate("", xy=(onset, 0.06), xytext=(alarm, 0.06),
                    arrowprops=dict(arrowstyle="<->", color="0.3", lw=1.2))
        ax.text((alarm + onset) / 2, 0.09, f"{onset - alarm:.0f} h early",
                ha="center", fontsize=8, color="0.2")
    if ex.get("_illustrative"):
        ax.text(0.99, 0.03, "illustrative example", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7, color="0.5", style="italic")
    ax.set_xlabel("hours since ICU admission"); ax.set_ylabel("onset risk")
    ax.set_ylim(0, 1.0); ax.set_xlim(hrs.min(), hrs.max())
    ax.set_title("Per-organ attribution: which organ system drives the alert", fontsize=10)
    ax.legend(ncol=4, fontsize=7.2, loc="upper left", framealpha=0.9)
    _save(fig, "fig_attribution")


def run() -> None:
    d = _load()
    fig_discrimination(d)
    fig_robustness(d)
    fig_ablation(d)
    if (FIGDIR / "attribution_example.json").exists():
        fig_attribution(_load_attr())
    log.info("figures -> %s", FIGDIR)
