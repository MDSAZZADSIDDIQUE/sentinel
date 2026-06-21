"""Phase 9 Step 6a: missing-signal robustness TRANSFER to eICU.

Re-runs the Phase 5 per-organ blinding curve, but scores FROZEN MIMIC-trained
models on the eICU external cohort. Tests whether SENTINEL's decomposition
robustness (graceful degradation when an organ goes dark) survives a real
external hospital network — not just the simulated CVICU held-out unit.

Train on MIMIC `train` (frozen — no gradient touches eICU), blind one organ's
inputs at test time, score eICU `external`. Same three models as Phase 5:

  SENTINEL-Ensemble — organ-decomposed (one GRU per organ + max-combine). Blinded
                      = drop the dark organ from the max (the others still fire).
  JointEnsemble-6   — capacity-matched CONTROL (6 GRUs on the joint vector +
                      max-combine, no organ split). Blinded = zero the organ cols.
  GRU-single        — monolith. Blinded = zero the organ cols.

AUPRC is the primary metric: blinding does NOT change eICU prevalence, so the Δ
from full inputs is a clean within-eICU degradation with no prevalence confound.
The decomposition hypothesis predicts SENTINEL-Ensemble Δ ≈ 0 while the control
and the monolith degrade — and that this holds on the real external network.
"""
from __future__ import annotations

import json
import time

import numpy as np

from ..config import CohortConfig, MARLConfig
from ..constants import ORGAN_SYSTEMS
from ..features.dataset import load_hourly, load_split
from ..logging_utils import get_logger
from ..paths import PATHS
from . import metrics as M
from .robustness import _gru_predict_drop  # blind = zero drop_cols, then predict_seq

log = get_logger("eval.eicu_robustness")

MODEL_ORDER = ("SENTINEL-Ensemble", "JointEnsemble-6", "GRU-single")

# Phase 5 mean Δ across organs (from robustness_full.md) for the side-by-side.
PHASE5_MEAN_DELTA = {
    "test": {"SENTINEL-Ensemble": -0.001, "JointEnsemble-6": -0.030, "GRU-single": -0.028},
    "external (CVICU)": {"SENTINEL-Ensemble": -0.002, "JointEnsemble-6": -0.056, "GRU-single": -0.041},
}


def _organ_scores_once(ens, df_e, split):
    """Score each organ predictor ONCE on eICU → {organ: per-hour score array}.

    Every blinding condition is then a max-combine over cached arrays
    (drop X = max over organs != X), so we never re-score 122k stays per drop.
    """
    from ..baselines.torch_seq import EpisodeSeqDataset, predict_seq

    sub = df_e[df_e["split"] == split]
    return {organ: predict_seq(model, EpisodeSeqDataset(sub, cols))
            for organ, (model, cols) in ens.items()}


def _seed_cache(seed: int):
    """Per-seed checkpoint path. This laptop idle-sleeps mid-run and the run is
    ~3.5 h over 3 seeds; caching each completed seed makes the run resumable so a
    sleep/death costs at most one seed. Under outputs/ (gitignored, DUA-safe)."""
    return PATHS.output_root / "cache" / f"eicu_robustness_seed{seed}.json"


def run(eicu_cfg: CohortConfig | None = None, mimic_cfg: CohortConfig | None = None,
        mcfg: MARLConfig | None = None, seeds=(0, 1, 2)) -> None:
    from ..baselines.gru import (predict_joint_ensemble, train_gru,
                                  train_joint_ensemble, train_organ_ensemble)

    eicu_cfg = eicu_cfg or CohortConfig(mode="eicu")
    mimic_cfg = mimic_cfg or CohortConfig(mode="full")
    mcfg = mcfg or MARLConfig.load()
    t0 = time.perf_counter()

    df_m = load_hourly(mimic_cfg)     # train the frozen models here
    df_e = load_hourly(eicu_cfg)      # blind + score here
    split = "external"
    if not (df_e["split"] == split).any():
        raise SystemExit("eICU has no external split — run `eicu-features` first.")

    tr = load_split(mimic_cfg, "train", df=df_m, ablation=mcfg.ablation)
    pw = (len(tr.y) - int(tr.y.sum())) / max(int(tr.y.sum()), 1)
    ext = load_split(eicu_cfg, split, df=df_e, ablation=mcfg.ablation)
    y_e = ext.y

    organs = list(ORGAN_SYSTEMS)
    conditions = [None] + organs
    no_skill = float(np.mean(y_e))
    # Signature guards the seed caches against silent reuse across a different
    # cohort / feature set (recompute if the data the cache was built on changed).
    sig = f"{len(tr.y)}-{len(y_e)}-{len(tr.feature_names)}-{mcfg.ablation}"
    log.info("robustness transfer: MIMIC train=%d | eICU external=%d stays | no-skill AUPRC=%.3f",
             len(tr.y), len(np.unique(ext.stay_ids)), no_skill)

    def _compute_seed(seed: int) -> dict:
        log.info("  seed %d: train organ-ensemble + joint-ensemble + GRU on MIMIC (frozen)", seed)
        ens = train_organ_ensemble(df_m, tr.manifest, pw, seed=seed)
        je = train_joint_ensemble(df_m, tr.feature_names, pw, seed=seed)
        gm = train_gru(df_m, tr.feature_names, pw, seed=seed)

        organ_sc = _organ_scores_once(ens, df_e, split)  # 6 scorings, reused below
        auprc = {m: {} for m in MODEL_ORDER}
        au0: dict[str, float] = {}
        for dropped in conditions:
            keep = [organ_sc[o] for o in organs if o != dropped]
            drop_cols = tr.manifest[dropped] if dropped else []
            de = M.discrimination(y_e, np.maximum.reduce(keep))
            dj = M.discrimination(y_e, predict_joint_ensemble(je, df_e, split,
                                                              tr.feature_names, drop_cols))
            dg = M.discrimination(y_e, _gru_predict_drop(gm, df_e, split,
                                                         tr.feature_names, drop_cols))
            key = "__none__" if dropped is None else dropped
            auprc["SENTINEL-Ensemble"][key] = de.auprc
            auprc["JointEnsemble-6"][key] = dj.auprc
            auprc["GRU-single"][key] = dg.auprc
            if dropped is None:
                au0 = {"SENTINEL-Ensemble": de.auroc, "JointEnsemble-6": dj.auroc,
                       "GRU-single": dg.auroc}
        return {"_sig": sig, "auprc": auprc, "auroc0": au0}

    res: dict[tuple, list] = {}
    auroc0: dict[str, list] = {}
    for seed in seeds:
        cache = _seed_cache(seed)
        payload = None
        if cache.exists():
            cached = json.loads(cache.read_text(encoding="utf-8"))
            if cached.get("_sig") == sig:
                payload = cached
                log.info("  seed %d: loaded cache %s (skip training)", seed, cache.name)
            else:
                log.info("  seed %d: cache %s stale (sig mismatch) — recomputing", seed, cache.name)
        if payload is None:
            payload = _compute_seed(seed)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(payload), encoding="utf-8")
            log.info("  seed %d done -> cached %s", seed, cache.name)
        for m, dd in payload["auprc"].items():
            for key, val in dd.items():
                res.setdefault((m, None if key == "__none__" else key), []).append(val)
        for m, val in payload["auroc0"].items():
            auroc0.setdefault(m, []).append(val)

    _write_report(eicu_cfg, res, auroc0, conditions, seeds, no_skill)
    log.info("eICU robustness transfer done in %.1fs", time.perf_counter() - t0)


def _write_report(cfg, res, auroc0, conditions, seeds, no_skill) -> None:
    def mean(key):
        return float(np.nanmean(res.get(key, [float("nan")])))

    models = [m for m in MODEL_ORDER if any(k[0] == m for k in res)]
    L = [f"# SENTINEL — Missing-signal robustness TRANSFER to eICU\n",
         f"_Frozen MIMIC-trained models; test-time organ blinding scored on the eICU "
         f"external cohort (real {208}-hospital network). Mean AUPRC over {len(seeds)} "
         f"seed(s); Δ = AUPRC drop from full inputs. Graceful = small Δ. eICU no-skill "
         f"AUPRC (positive-hour rate) ≈ **{no_skill:.3f}**. Blinding leaves prevalence "
         f"unchanged, so Δ is a clean within-eICU degradation (no prevalence confound). "
         f"**JointEnsemble-6** is the capacity-matched control (6 GRUs on the joint "
         f"vector, no organ split): if it degrades like GRU-single, the robustness is "
         f"from DECOMPOSITION, not ensembling._\n"]

    L.append("\n## eICU external — per-organ blinding\n")
    L.append("| Dropped organ | " + " | ".join(f"{m} | Δ" for m in models) + " |")
    L.append("|---|" + "---|---|" * len(models))
    base = {m: mean((m, None)) for m in models}
    for d in conditions:
        row = [f"{'(none)' if d is None else d}"]
        for m in models:
            v = mean((m, d))
            row += [f"{v:.3f}", "" if d is None else f"{v - base[m]:+.3f}"]
        L.append("| " + " | ".join(row) + " |")

    mean_delta = {m: float(np.mean([mean((m, d)) - base[m] for d in conditions if d]))
                  for m in models}
    summ = ", ".join(f"{m} {mean_delta[m]:+.3f}" for m in models)
    L.append(f"\n_Mean Δ across organs (eICU): {summ}. More-negative = less robust._")

    # (none) baseline AUROC — the prevalence-independent external discrimination.
    arow = ", ".join(f"{m} {float(np.mean(auroc0.get(m, [float('nan')]))):.3f}" for m in models)
    L.append(f"\n_Full-input eICU AUROC (no organ dropped): {arow}._")

    # Side-by-side with Phase 5 (MIMIC test + simulated CVICU external).
    L.append("\n## Mean Δ across organs — does the robustness gap transfer?\n")
    L.append("| Site | " + " | ".join(models) + " |")
    L.append("|---|" + "---|" * len(models))
    for site, dd in PHASE5_MEAN_DELTA.items():
        L.append("| " + " | ".join([f"MIMIC {site}"] + [f"{dd[m]:+.3f}" for m in models]) + " |")
    L.append("| " + " | ".join(["**eICU external (real)**"]
                                + [f"**{mean_delta[m]:+.3f}**" for m in models]) + " |")
    L.append("\n_Honest read: SENTINEL-Ensemble stays the most robust (≈0), but the large MIMIC "
             "gap ATTENUATES on real eICU — the control/monolith degrade far less here because "
             "they already sit near the signal floor externally (full-input AUPRC ≈ no-skill), "
             "leaving little for blinding to remove. The decomposition advantage survives "
             "per-organ for the high-signal respiratory/cardiovascular organs, but the aggregate "
             "gap does NOT transfer at MIMIC magnitude._")

    PATHS.reports_root.mkdir(parents=True, exist_ok=True)
    out = PATHS.reports_root / "eicu_robustness.md"
    out.write_text("\n".join(L), encoding="utf-8")
    log.info("  wrote %s", out)
