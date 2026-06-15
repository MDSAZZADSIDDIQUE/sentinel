"""Smoke tests for the MARL trainer (MAPPO + IPPO paths). Runs when features
are built; trains 2 updates and checks the team score is a valid probability."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

pytest.importorskip("torch")
from sentinel.config import CohortConfig, MARLConfig  # noqa: E402
from sentinel.paths import PATHS  # noqa: E402


def _mode():
    for m in ("dev", "full"):
        if (PATHS.features_root / f"hourly_{m}.parquet").exists():
            return m
    return None


MODE = _mode()
pytestmark = pytest.mark.skipif(MODE is None, reason="features not built")


def _train_and_score(algo: str):
    from sentinel.agents.mappo import EpisodeTensors, team_scores, train_marl
    cfg = CohortConfig(mode=MODE)
    mc = dataclasses.replace(MARLConfig(algo=algo), updates=2, eval_every=1,
                             batch_episodes=32, ppo_epochs=2)
    policy, df, manifest = train_marl(cfg, mc)
    t = EpisodeTensors(df[df["split"] == "val"], manifest, "full")
    s = team_scores(policy, t)
    assert s.ndim == 1 and s.shape[0] > 0
    assert np.isfinite(s).all() and (s >= 0).all() and (s <= 1).all()
    return s


def test_mappo_trains_and_scores():
    _train_and_score("mappo")


def test_ippo_trains_and_scores():
    _train_and_score("ippo")


def test_team_alert_is_max_over_agents():
    """Sanity: the env's effective action is the most-alarmed organ."""
    from sentinel.env.sentinel_env import SentinelEnv
    env = SentinelEnv(CohortConfig(mode=MODE), split="val", seed=0)
    env.reset(episode_idx=0)
    actions = {a: 0 for a in env.agents}
    actions[env.agents[2]] = 2
    _, _, _, info = env.step(actions)
    assert info["effective_action"] == 2
