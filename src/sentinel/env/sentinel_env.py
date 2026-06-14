"""SENTINEL Dec-POMDP early-warning environment.

One episode = one ICU stay, streamed hour by hour over its (leakage-safe)
observation window. Six organ-system agents each observe only their own organ's
features plus a small shared context (partial observability). Each hour the team
picks a joint escalation; the **effective team alert = the most alarmed organ**
(max over agents), which keeps the decision interpretable (you can see which
organ raised it). The reward is the shared early-warning utility (see reward.py).

Built on the precomputed feature parquet (no raw-table access at train time).
Numpy-only; agents/training add torch.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import CohortConfig, FeatureConfig
from ..constants import ORGAN_SYSTEMS, SHARED_GROUP
from ..logging_utils import get_logger
from ..paths import PATHS
from .reward import RewardConfig, step_reward, terminal_reward

log = get_logger("env")


class SentinelEnv:
    """Single-episode multi-agent env over the feature tensor for one data split."""

    def __init__(self, cfg: CohortConfig, split: str = "train",
                 rcfg: RewardConfig | None = None, fcfg: FeatureConfig | None = None,
                 seed: int = 0):
        self.cfg = cfg
        self.split = split
        self.rcfg = rcfg or RewardConfig()
        self.fcfg = fcfg or FeatureConfig.load()
        self.rng = np.random.default_rng(seed)
        self.agents = list(ORGAN_SYSTEMS)

        with (PATHS.features_root / f"feature_manifest_{cfg.mode}.json").open() as fh:
            manifest = json.load(fh)
        df = pd.read_parquet(PATHS.features_root / f"hourly_{cfg.mode}.parquet")
        df = df[df["split"] == split].sort_values(["stay_id", "hr"]).reset_index(drop=True)
        if df.empty:
            raise ValueError(f"no episodes for split={split!r} (mode={cfg.mode})")

        shared_cols = manifest.get(SHARED_GROUP, [])
        # each agent sees its organ features + shared context
        self.agent_cols = {a: manifest[a] + shared_cols for a in self.agents}
        self.obs_dims = {a: len(self.agent_cols[a]) for a in self.agents}
        self.n_actions = 3

        # precompute per-episode arrays
        all_cols = sorted({c for cols in self.agent_cols.values() for c in cols})
        self._col_idx = {a: [all_cols.index(c) for c in self.agent_cols[a]] for a in self.agents}
        self.episodes = []
        for sid, ep in df.groupby("stay_id", sort=False):
            self.episodes.append({
                "stay_id": int(sid),
                "X": ep[all_cols].to_numpy(dtype=np.float32),
                "y": ep["y"].to_numpy(dtype=np.int64),
                "t_to_onset": ep["t_to_onset"].to_numpy(dtype=np.float32),
                "label": int(ep["label"].iloc[0]),
            })
        log.info("env[%s/%s]: %d episodes | agents=%d | obs dims=%s",
                 cfg.mode, split, len(self.episodes), len(self.agents), self.obs_dims)
        self._ep = None
        self._t = 0

    def __len__(self) -> int:
        return len(self.episodes)

    def _obs(self) -> dict[str, np.ndarray]:
        row = self._ep["X"][self._t]
        return {a: row[self._col_idx[a]] for a in self.agents}

    def reset(self, episode_idx: int | None = None) -> dict[str, np.ndarray]:
        idx = self.rng.integers(len(self.episodes)) if episode_idx is None else episode_idx
        self._ep = self.episodes[idx]
        self._t = 0
        self._alerted = False
        return self._obs()

    def step(self, actions: dict[str, int]):
        ep = self._ep
        effective = max(int(actions[a]) for a in self.agents)  # most-alarmed organ
        lead = float(ep["t_to_onset"][self._t])
        if not np.isfinite(lead):
            lead = np.inf
        r, fired = step_reward(ep["label"], effective, lead, self._alerted, self.rcfg)
        if fired:
            self._alerted = True

        self._t += 1
        done = self._t >= len(ep["y"])
        if done:
            r += terminal_reward(ep["label"], self._alerted, self.rcfg)
            obs = {a: np.zeros(self.obs_dims[a], np.float32) for a in self.agents}
        else:
            obs = self._obs()
        info = {"effective_action": effective, "actions": dict(actions),
                "y": int(ep["y"][self._t - 1]), "lead": lead, "label": ep["label"],
                "stay_id": ep["stay_id"]}
        return obs, float(r), done, info
