"""Smoke test for the Dec-POMDP environment (runs when features are built)."""
from __future__ import annotations

import pytest

pytest.importorskip("pandas")
from sentinel.config import CohortConfig  # noqa: E402
from sentinel.constants import ORGAN_SYSTEMS  # noqa: E402
from sentinel.paths import PATHS  # noqa: E402


def _mode():
    for m in ("dev", "full"):  # prefer the small cohort for a fast test
        if (PATHS.features_root / f"hourly_{m}.parquet").exists():
            return m
    return None


MODE = _mode()
pytestmark = pytest.mark.skipif(MODE is None, reason="features not built")


def test_env_runs_one_episode():
    from sentinel.env.sentinel_env import SentinelEnv

    env = SentinelEnv(CohortConfig(mode=MODE), split="train", seed=0)
    assert set(env.agents) == set(ORGAN_SYSTEMS)
    obs = env.reset(episode_idx=0)
    assert set(obs) == set(env.agents)
    for a in env.agents:
        assert obs[a].shape == (env.obs_dims[a],)

    total, steps, done = 0.0, 0, False
    while not done and steps < 1000:
        actions = {a: (steps % 3) for a in env.agents}  # cycle maintain/watch/escalate
        obs, r, done, info = env.step(actions)
        assert isinstance(r, float)
        total += r
        steps += 1
    assert done and steps >= 1
    assert "effective_action" in info


def test_effective_action_is_team_max():
    from sentinel.env.sentinel_env import SentinelEnv

    env = SentinelEnv(CohortConfig(mode=MODE), split="train", seed=1)
    env.reset(episode_idx=0)
    actions = {a: 0 for a in env.agents}
    next(iter(actions))  # ensure non-empty
    one = env.agents[0]
    actions[one] = 2
    _, _, _, info = env.step(actions)
    assert info["effective_action"] == 2  # most-alarmed organ drives the team
