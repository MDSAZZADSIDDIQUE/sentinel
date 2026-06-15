"""Missing-signal robustness: drop one organ's inputs at test time and measure
degradation. The decomposition hypothesis — SENTINEL (per-organ agents, max-
combine) should degrade GRACEFULLY when an organ goes dark, because the other
organs still drive the alert, whereas the monolithic GRU (one network over the
joint vector) has no such structural fallback.

Reports test (and external) AUPRC with each organ blinded, for SENTINEL-Ensemble
vs GRU-single, and the degradation from the full-input baseline.
"""
from __future__ import annotations

import time

import numpy as np

from ..config import CohortConfig, MARLConfig
from ..constants import ORGAN_SYSTEMS
from ..logging_utils import get_logger
from ..paths import PATHS
from ..features.dataset import load_hourly, load_split
from . import metrics as M

log = get_logger("eval.robustness")


def _gru_predict_drop(model, df, split, feature_cols, drop_cols):
    from ..baselines.torch_seq import EpisodeSeqDataset, predict_seq
    sub = df[df["split"] == split].copy()
    for c in drop_cols:
        if c in sub.columns:
            sub[c] = 0.0
    return predict_seq(model, EpisodeSeqDataset(sub, feature_cols))


MODELS = ("SENTINEL-Ensemble", "GRU-single")


def _ensemble_predict_drop(models, df, split, drop_organ):
    """Team risk = max over organs EXCEPT the dropped (dark) one."""
    from ..baselines.torch_seq import EpisodeSeqDataset, predict_seq
    per = [predict_seq(m, EpisodeSeqDataset(df[df["split"] == split], cols))
           for organ, (m, cols) in models.items() if organ != drop_organ]
    return np.maximum.reduce(per)


def run(cfg: CohortConfig | None = None, mcfg: MARLConfig | None = None,
        seeds=(0, 1, 2), splits=("test", "external")) -> None:
    from ..baselines.gru import predict_organ_ensemble, train_gru, train_organ_ensemble

    cfg = cfg or CohortConfig.load()
    mcfg = mcfg or MARLConfig.load()
    t0 = time.perf_counter()
    df = load_hourly(cfg)
    manifest = {o: c for o, c in __import__("json").load(
        open(PATHS.features_root / f"feature_manifest_{cfg.mode}.json")).items()}
    splits = [s for s in splits if (df["split"] == s).any()]
    organs = list(ORGAN_SYSTEMS)
    conditions = [None] + organs

    # SENTINEL-Ensemble = decomposed SUPERVISED organ risk heads + max-combine (the
    # reframed system: strong like the deep model, robust by decomposition).
    # GRU-single = monolithic deep model over the joint vector.
    res: dict[tuple, list] = {}
    for seed in seeds:
        log.info("  seed %d: training SENTINEL-Ensemble + GRU-single", seed)
        tr = load_split(cfg, "train", df=df, ablation=mcfg.ablation)
        pw = (len(tr.y) - int(tr.y.sum())) / max(int(tr.y.sum()), 1)
        ens = train_organ_ensemble(df, tr.manifest, pw, seed=seed)
        gm = train_gru(df, tr.feature_names, pw, seed=seed)
        for split in splits:
            data = load_split(cfg, split, df=df, ablation=mcfg.ablation)
            for dropped in conditions:
                ap_e = M.discrimination(data.y, (predict_organ_ensemble(ens, df, split)
                       if dropped is None else _ensemble_predict_drop(ens, df, split, dropped))).auprc
                drop_cols = tr.manifest[dropped] if dropped else []
                ap_g = M.discrimination(
                    data.y, _gru_predict_drop(gm, df, split, tr.feature_names, drop_cols)).auprc
                res.setdefault(("SENTINEL-Ensemble", split, dropped), []).append(ap_e)
                res.setdefault(("GRU-single", split, dropped), []).append(ap_g)
        log.info("  seed %d done", seed)

    _write_report(cfg, res, splits, conditions, seeds)
    log.info("robustness done in %.1fs", time.perf_counter() - t0)


def _write_report(cfg, res, splits, conditions, seeds):
    def mean(key):
        v = res.get(key, [float("nan")])
        return float(np.nanmean(v))

    L = [f"# SENTINEL — Missing-signal robustness ({cfg.mode} cohort)\n",
         f"_Test-time organ blinding (inputs zeroed); mean AUPRC over {len(seeds)} seed(s). "
         "Δ = AUPRC drop from full inputs. Graceful = small Δ when an organ goes dark._\n"]
    for split in splits:
        L.append(f"\n## Split: {split}\n")
        L.append("| Dropped organ | SENTINEL-Ensemble | Δ | GRU-single | Δ |")
        L.append("|---|---|---|---|---|")
        base_m = mean(("SENTINEL-Ensemble", split, None))
        base_g = mean(("GRU-single", split, None))
        for d in conditions:
            name = "(none)" if d is None else d
            m = mean(("SENTINEL-Ensemble", split, d))
            g = mean(("GRU-single", split, d))
            dm = "" if d is None else f"{m-base_m:+.3f}"
            dg = "" if d is None else f"{g-base_g:+.3f}"
            L.append(f"| {name} | {m:.3f} | {dm} | {g:.3f} | {dg} |")
        # summary: mean degradation across organs
        md = np.mean([mean(("SENTINEL-Ensemble", split, d)) - base_m for d in conditions if d])
        gd = np.mean([mean(("GRU-single", split, d)) - base_g for d in conditions if d])
        L.append(f"\n_Mean Δ across organs: SENTINEL {md:+.3f} vs GRU-single {gd:+.3f}. "
                 "More-negative = larger degradation = less robust._")
    PATHS.reports_root.mkdir(parents=True, exist_ok=True)
    out = PATHS.reports_root / f"robustness_{cfg.mode}.md"
    out.write_text("\n".join(L), encoding="utf-8")
    log.info("  wrote %s", out)
