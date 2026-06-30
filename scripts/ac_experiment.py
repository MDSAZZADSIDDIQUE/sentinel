"""Methodology experiment: does conditioning the actor on its own alert history
help? If action-conditioned ≈ observation-only on dev, the fixed-trajectory
one-forward-pass simplification is empirically justified.

Trains both variants (MAPPO) on dev across seeds and compares test AUPRC.
Run on CPU to avoid contending with the GPU adjudication.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from sentinel.agents.action_conditioned import team_scores_ac, train_action_conditioned
from sentinel.agents.mappo import EpisodeTensors, team_scores, train_marl
from sentinel.config import CohortConfig, MARLConfig
from sentinel.eval.metrics import discrimination
from sentinel.features.dataset import load_hourly, load_split

cfg = CohortConfig(mode="dev")
df = load_hourly(cfg)
seeds = [0, 1, 2]
variants = [
    ("observation-only", train_marl, team_scores),
    ("action-conditioned", train_action_conditioned, team_scores_ac),
]

print(f"{'variant':<20} {'test AUPRC (mean±std over %d seeds)' % len(seeds)}")
for name, train_fn, score_fn in variants:
    aps = []
    for seed in seeds:
        mc = dataclasses.replace(MARLConfig(algo="mappo"), updates=80, eval_every=80,
                                 batch_episodes=128, seed=seed)
        policy, _, manifest = train_fn(cfg, mc, df=df)
        st = EpisodeTensors(df[df["split"] == "test"], manifest, "full")
        data = load_split(cfg, "test", df=df)
        d = discrimination(data.y, score_fn(policy, st))
        aps.append(d.auprc)
    print(f"{name:<20} {np.mean(aps):.3f} ± {np.std(aps):.3f}")
print("\nIf the two rows overlap, the observation-only simplification is justified.")
