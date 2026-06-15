"""MARL correctness gates (no data needed — always run):

Gate 1  the MAPPO critic ingests the JOINT state while IPPO critics are per-organ
        (otherwise MAPPO == IPPO by construction and the headline is meaningless).
Gate 2  the vectorized `batched_reward` (training) equals the env `reward.py`
        path (eval) EXACTLY on the same episodes (else train/eval optimize
        different objectives).
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from sentinel.config import MARLConfig  # noqa: E402
from sentinel.constants import ORGAN_SYSTEMS  # noqa: E402
from sentinel.env.reward import RewardConfig, step_reward, terminal_reward  # noqa: E402


# ---------------------------------------------------------------- Gate 1: critic
def _build(algo):
    from sentinel.agents.mappo import SentinelMARL
    organ_dims = {o: 5 + i for i, o in enumerate(ORGAN_SYSTEMS)}   # distinct per organ
    joint_dim = sum(organ_dims.values())
    return SentinelMARL(organ_dims, joint_dim, MARLConfig(algo=algo)), organ_dims, joint_dim


def test_mappo_critic_is_centralized():
    policy, organ_dims, joint_dim = _build("mappo")
    assert policy.critic.gru.input_size == joint_dim
    assert joint_dim > max(organ_dims.values()), "joint state must exceed any single organ"


def test_ippo_critics_are_decentralized():
    policy, organ_dims, joint_dim = _build("ippo")
    assert not hasattr(policy, "critic"), "IPPO must not have a centralized critic"
    for o in ORGAN_SYSTEMS:
        assert policy.critics[o].gru.input_size == organ_dims[o]


def test_actors_are_always_decentralized():
    for algo in ("mappo", "ippo"):
        policy, organ_dims, _ = _build(algo)
        for o in ORGAN_SYSTEMS:
            assert policy.actors[o].gru.input_size == organ_dims[o]


# --------------------------------------------------- Gate 2: reward equivalence
def _env_episode_reward(label, actions, leads, cfg: RewardConfig) -> float:
    """Reference per-episode reward via the env/reward.py functions (eval path)."""
    alerted, prev, total = False, -1, 0.0
    for t in range(len(actions)):
        e = int(actions[t])
        new_alarm = e == 2 and prev != 2
        r, fired = step_reward(label, e, float(leads[t]), alerted, cfg, new_alarm=new_alarm)
        alerted = alerted or fired
        prev = e
        total += r
    total += terminal_reward(label, alerted, cfg)
    return total


def test_batched_reward_equals_env_path():
    from sentinel.agents.mappo import batched_reward
    cfg = RewardConfig()
    rng = np.random.default_rng(0)
    for trial in range(200):
        T = int(rng.integers(6, 40))
        valid = int(rng.integers(6, T + 1))
        label = int(rng.integers(0, 2))
        onset = int(rng.integers(6, valid + 6)) if label == 1 else -1
        actions = rng.integers(0, 3, size=T)
        leads = np.full(T, np.inf, np.float32)
        if label == 1:
            for t in range(valid):
                leads[t] = onset - t
        mask = np.zeros(T, np.float32)
        mask[:valid] = 1.0
        # batched (1 episode); padded hours masked out
        eff = actions[None, :].copy()
        rb = batched_reward(eff, np.array([label]), leads[None, :], mask[None, :], cfg)
        got = float(rb.sum())
        # reference: only the valid hours, leads on those hours
        expected = _env_episode_reward(label, actions[:valid], leads[:valid], cfg)
        assert abs(got - expected) < 1e-5, f"trial {trial}: batched {got} != env {expected}"
