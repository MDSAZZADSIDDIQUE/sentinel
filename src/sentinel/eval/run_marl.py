"""Phase 4 runner: train SENTINEL-MAPPO and the IPPO control, evaluate both
through the SAME metrics pipeline as the baselines, and report the headline
MAPPO-vs-IPPO comparison at a matched alarm budget.

Parity is enforced by construction: identical actors, capacity, data, seeds,
reward, and operating-point picker — only the critic differs (centralized vs
decentralized), so any gap is coordinated credit assignment.
"""
from __future__ import annotations

import dataclasses
import time

from ..config import CohortConfig, MARLConfig
from ..env.reward import RewardConfig
from ..features.dataset import load_hourly, load_split
from ..logging_utils import get_logger
from ..paths import PATHS
from .run_baselines import _agg, _eval_split

log = get_logger("eval.marl")


def run(cfg: CohortConfig | None = None, mcfg: MARLConfig | None = None,
        rcfg: RewardConfig | None = None, seeds=(0, 1, 2), alarm_budget: float = 8.0,
        algos=("mappo", "ippo"), ablations=("full",)) -> None:
    from ..agents.mappo import EpisodeTensors, team_scores, train_marl

    cfg = cfg or CohortConfig.load()
    mcfg = mcfg or MARLConfig.load()
    rcfg = rcfg or RewardConfig()
    t0 = time.perf_counter()
    df = load_hourly(cfg)
    splits = [s for s in ("val", "test", "external") if (df["split"] == s).any()]
    log.info("MARL (mode=%s): algos=%s ablations=%s splits=%s seeds=%s",
             cfg.mode, list(algos), list(ablations), splits, list(seeds))

    results: dict[tuple, dict] = {}
    for algo in algos:
        for ablation in ablations:
            rows = {s: [] for s in splits}
            for seed in seeds:
                mc = dataclasses.replace(mcfg, algo=algo, ablation=ablation, seed=seed)
                policy, _, manifest = train_marl(cfg, mc, rcfg, df=df)
                vd = load_split(cfg, "val", df=df)
                v_t = EpisodeTensors(df[df["split"] == "val"], manifest, ablation)
                thr = (vd.stay_ids, vd.hr, team_scores(policy, v_t), vd.t_to_onset, vd.ep_label)
                for s in splits:
                    data = load_split(cfg, s, df=df)
                    st = EpisodeTensors(df[df["split"] == s], manifest, ablation)
                    rows[s].append(_eval_split(data, team_scores(policy, st), thr, alarm_budget))
            for s in splits:
                results[(f"SENTINEL-{algo.upper()}", ablation, s)] = _agg(rows[s])
            log.info("  %s[%s] done", algo, ablation)

    _write_report(cfg, results, splits, seeds, alarm_budget)
    log.info("MARL done in %.1fs", time.perf_counter() - t0)


def _write_report(cfg, results, splits, seeds, alarm_budget):
    L = [f"# SENTINEL — Phase 4 MARL ({cfg.mode} cohort)\n",
         f"_AUPRC primary (95% CI over stays); operating point at ≤ {alarm_budget:.0f} alarm "
         f"events/patient-day on val; {len(seeds)} seed(s) mean±std._\n",
         "**Headline:** SENTINEL-MAPPO (centralized critic = coordination) vs IPPO "
         "(decentralized critics, identical otherwise). A gap at matched alarm budget is "
         "the value of cooperative credit assignment; parity is a real finding, not a "
         "failure. Compare also to the supervised baselines (`baselines_<mode>.md`).\n"]
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
    L.append("\n_Dev cohort can only confirm MAPPO trains and is in the ballpark "
             "(~400 positives, wide CIs); the coordination claim is adjudicated on the "
             "full-cohort / 5-seed run, not here._\n")
    PATHS.reports_root.mkdir(parents=True, exist_ok=True)
    out = PATHS.reports_root / f"marl_{cfg.mode}.md"
    out.write_text("\n".join(L), encoding="utf-8")
    log.info("  wrote %s", out)
