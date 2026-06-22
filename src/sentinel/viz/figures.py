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
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


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
    for ax, split in zip(axes, ("test", "external")):
        vals = [d["discrimination"][m][split] for m in models]
        y = range(len(models))
        ap = [v[0] for v in vals]
        err = [[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]]
        colors = [SENTINEL_C if "SENTINEL (organ)" == m else "#888" for m in models]
        ax.barh(y, ap, xerr=err, color=colors, height=0.62, error_kw={"lw": 1})
        ax.axvline(d["no_skill"][split], ls="--", color="k", lw=1, label="no-skill")
        ax.set_yticks(list(y)); ax.set_yticklabels(models if split == "test" else [])
        ax.set_xlabel("AUPRC"); ax.set_title(f"{split} (internal)" if split == "test" else "external (CVICU)")
        ax.invert_yaxis(); ax.legend(fontsize=8, loc="lower right")
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
               label=m, color=colors[m])
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


def run() -> None:
    d = _load()
    fig_discrimination(d)
    fig_robustness(d)
    fig_ablation(d)
    log.info("figures -> %s", FIGDIR)
