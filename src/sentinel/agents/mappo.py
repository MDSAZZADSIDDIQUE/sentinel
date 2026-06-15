"""SENTINEL cooperative MARL: MAPPO (CTDE) with an IPPO control.

Six per-organ GRU actors choose escalation each hour; the team alert = the most-
alarmed organ (max-pool, interpretable). Training is CTDE PPO on a shared TEAM
reward. The ONLY difference between SENTINEL/MAPPO and the IPPO control is the
critic:
  * MAPPO  — one CENTRALIZED critic over the joint state (coordinated credit),
  * IPPO   — six DECENTRALIZED per-organ critics on own obs (no coordination).
Same actors, capacity, data, seeds, reward, operating-point picker → the critic
isolates the value of coordination.

Because the action is alerting (not treatment), observations are a FIXED
trajectory per episode (actions don't change the patient), so we forward each
episode once, sample actions, and compute rewards/PPO post-hoc — recurrent PPO
without environment stepping. FP32 on the GTX 1650.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from ..config import CohortConfig, MARLConfig
from ..constants import ORGAN_SYSTEMS, SHARED_GROUP
from ..env.reward import RewardConfig
from ..features.dataset import feature_columns, load_hourly
from ..logging_utils import get_logger
from ..paths import PATHS

log = get_logger("agents.mappo")
N_ACT = 3  # maintain / watch / escalate


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- data
class EpisodeTensors:
    """Padded per-organ + joint observation tensors for one split."""

    def __init__(self, df, manifest: dict, ablation: str):
        from ..features.dataset import col_dropped
        df = df.sort_values(["stay_id", "hr"])
        shared = [c for c in manifest[SHARED_GROUP] if not col_dropped(c, ablation)]
        self.organ_cols = {o: [c for c in manifest[o] if not col_dropped(c, ablation)] + shared
                           for o in ORGAN_SYSTEMS}
        self.joint_cols, _ = feature_columns(manifest, ablation)

        groups = list(df.groupby("stay_id", sort=False))
        N = len(groups)
        T = max(len(ep) for _, ep in groups)
        self.N, self.T = N, T
        self.mask = np.zeros((N, T), np.float32)
        self.labels = np.zeros(N, np.float32)
        self.leads = np.full((N, T), np.inf, np.float32)
        self.lengths = np.zeros(N, np.int64)
        self.stay_ids = np.zeros(N, np.int64)
        self.organ = {o: np.zeros((N, T, len(c)), np.float32) for o, c in self.organ_cols.items()}
        self.joint = np.zeros((N, T, len(self.joint_cols)), np.float32)
        for i, (sid, ep) in enumerate(groups):
            t = len(ep)
            self.mask[i, :t] = 1.0
            self.labels[i] = ep["label"].iloc[0]
            self.leads[i, :t] = ep["t_to_onset"].to_numpy(np.float32)
            self.lengths[i] = t
            self.stay_ids[i] = int(sid)
            self.joint[i, :t] = ep[self.joint_cols].to_numpy(np.float32)
            for o, cols in self.organ_cols.items():
                self.organ[o][i, :t] = ep[cols].to_numpy(np.float32)

    def to(self, dev, idx=None):
        sl = slice(None) if idx is None else idx
        organ = {o: torch.as_tensor(v[sl], device=dev) for o, v in self.organ.items()}
        return (organ,
                torch.as_tensor(self.joint[sl], device=dev),
                torch.as_tensor(self.labels[sl], device=dev),
                torch.as_tensor(self.leads[sl], device=dev),
                torch.as_tensor(self.mask[sl], device=dev))


# ------------------------------------------------------------------------ networks
class GRUNet(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x):
        h, _ = self.gru(x)
        return self.head(h)


class SentinelMARL(nn.Module):
    def __init__(self, organ_dims: dict, joint_dim: int, mcfg: MARLConfig):
        super().__init__()
        self.algo = mcfg.algo
        self.agents = list(ORGAN_SYSTEMS)
        self.actors = nn.ModuleDict(
            {o: GRUNet(organ_dims[o], mcfg.hidden_actor, N_ACT) for o in self.agents})
        if mcfg.algo == "mappo":
            self.critic = GRUNet(joint_dim, mcfg.hidden_critic, 1)       # centralized
        else:  # ippo: per-organ decentralized critics
            self.critics = nn.ModuleDict(
                {o: GRUNet(organ_dims[o], mcfg.hidden_critic, 1) for o in self.agents})

    def actor_logits(self, organ_obs):
        return {o: self.actors[o](organ_obs[o]) for o in self.agents}

    def values(self, organ_obs, joint_obs):
        if self.algo == "mappo":
            v = self.critic(joint_obs).squeeze(-1)                       # [B,T]
            return {o: v for o in self.agents}                          # shared
        return {o: self.critics[o](organ_obs[o]).squeeze(-1) for o in self.agents}


# ------------------------------------------------------------------------- reward
def batched_reward(effective: np.ndarray, labels: np.ndarray, leads: np.ndarray,
                   mask: np.ndarray, rc: RewardConfig) -> np.ndarray:
    """Team reward [B,T] from sampled effective actions on the fixed trajectory."""
    B, T = effective.shape
    r = np.zeros((B, T), np.float32)
    alerted = np.zeros(B, bool)
    prev = np.full(B, -1)
    strength = np.array([0.0, rc.watch_frac, 1.0], np.float32)
    for t in range(T):
        e = effective[:, t]
        st = strength[e]
        lead = leads[:, t]
        timely = (labels == 1) & np.isfinite(lead) & (lead >= 1) & (lead <= rc.alert_window_hours)
        new_alarm = (e == 2) & (prev != 2)
        rt = np.where((e == 2) & new_alarm, -rc.c_escalate, 0.0).astype(np.float32)
        rt += np.where(timely, rc.r_detect * st, -rc.c_false_alarm * st)
        first = timely & (st > 0) & (~alerted)
        rt += np.where(first, rc.b_lead * np.where(np.isfinite(lead), lead, 0.0), 0.0)
        alerted |= first
        valid = mask[:, t] > 0
        prev = np.where(valid, e, prev)
        r[:, t] = rt * valid
    # terminal miss penalty at the last valid hour
    last = (mask.sum(1) - 1).astype(int)
    miss = (labels == 1) & (~alerted)
    r[np.arange(B), last] += np.where(miss, -rc.r_miss, 0.0)
    return r


def _gae(rewards, values, mask, gamma, lam):
    B, T = rewards.shape
    adv = np.zeros((B, T), np.float32)
    last = np.zeros(B, np.float32)
    for t in reversed(range(T)):
        nv = values[:, t + 1] if t + 1 < T else np.zeros(B, np.float32)
        delta = rewards[:, t] + gamma * nv * mask[:, t] - values[:, t]
        last = delta + gamma * lam * last * mask[:, t]
        adv[:, t] = last
    return adv * mask, (adv + values) * mask


def _manifest(cfg: CohortConfig) -> dict:
    with (PATHS.features_root / f"feature_manifest_{cfg.mode}.json").open() as fh:
        return json.load(fh)


# ------------------------------------------------------------------------- train
def train_marl(cfg: CohortConfig, mcfg: MARLConfig, rcfg: RewardConfig | None = None,
               df=None):
    rcfg = rcfg or RewardConfig()
    torch.manual_seed(mcfg.seed)
    np.random.seed(mcfg.seed)
    dev = _device()
    df = df if df is not None else load_hourly(cfg)
    manifest = _manifest(cfg)
    train_t = EpisodeTensors(df[df["split"] == "train"], manifest, mcfg.ablation)
    val_t = EpisodeTensors(df[df["split"] == "val"], manifest, mcfg.ablation)
    organ_dims = {o: len(c) for o, c in train_t.organ_cols.items()}
    policy = SentinelMARL(organ_dims, len(train_t.joint_cols), mcfg).to(dev)
    opt = torch.optim.Adam(policy.parameters(), lr=mcfg.lr)
    mappo = mcfg.algo == "mappo"
    agents = policy.agents
    log.info("  MARL[%s/%s ablation=%s] train=%d val=%d episodes, dev=%s",
             mcfg.algo, cfg.mode, mcfg.ablation, train_t.N, val_t.N, dev.type)

    for update in range(mcfg.updates):
        policy.train()  # team_scores() leaves it in eval; cudnn RNN backward needs train
        idx = np.random.choice(train_t.N, min(mcfg.batch_episodes, train_t.N), replace=False)
        organ_obs, joint_obs, labels, leads, mask = train_t.to(dev, idx)
        mask_t = mask
        with torch.no_grad():
            logits = policy.actor_logits(organ_obs)
            dists = {o: Categorical(logits=logits[o]) for o in agents}
            actions = {o: dists[o].sample() for o in agents}
            old_lp = {o: dists[o].log_prob(actions[o]) for o in agents}
            values = policy.values(organ_obs, joint_obs)
            eff = torch.stack([actions[o] for o in agents], -1).amax(-1)
        rew = batched_reward(eff.cpu().numpy(), labels.cpu().numpy(),
                             leads.cpu().numpy(), mask.cpu().numpy(), rcfg)
        m_np = mask.cpu().numpy()

        adv, ret = {}, {}
        for o in agents:
            a, r = _gae(rew, values[o].cpu().numpy(), m_np, mcfg.gamma, mcfg.gae_lambda)
            at = torch.as_tensor(a, device=dev)
            sel = mask_t > 0
            at = (at - at[sel].mean()) / (at[sel].std() + 1e-6)
            adv[o] = at
            ret[o] = torch.as_tensor(r, device=dev)
            if mappo:
                break  # shared critic -> identical for all agents
        if mappo:
            adv = {o: adv[agents[0]] for o in agents}
            ret_shared = ret[agents[0]]

        for _ in range(mcfg.ppo_epochs):
            logits = policy.actor_logits(organ_obs)
            values_new = policy.values(organ_obs, joint_obs)
            ploss = torch.zeros((), device=dev)
            ent = torch.zeros((), device=dev)
            denom = mask_t.sum()
            for o in agents:
                d = Categorical(logits=logits[o])
                ratio = torch.exp(d.log_prob(actions[o]) - old_lp[o])
                s1 = ratio * adv[o]
                s2 = torch.clamp(ratio, 1 - mcfg.clip, 1 + mcfg.clip) * adv[o]
                ploss += (-torch.min(s1, s2) * mask_t).sum() / denom
                ent += (d.entropy() * mask_t).sum() / denom
            if mappo:
                vloss = (((values_new[agents[0]] - ret_shared) ** 2) * mask_t).sum() / denom
            else:
                vloss = sum((((values_new[o] - ret[o]) ** 2) * mask_t).sum() / denom
                            for o in agents)
            loss = ploss + mcfg.value_coef * vloss - mcfg.entropy_coef * ent
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), mcfg.grad_clip)
            opt.step()

        if update % mcfg.eval_every == 0 or update == mcfg.updates - 1:
            from .mappo import quick_eval_auprc
            log.info("    update %2d/%d | loss=%.3f | mean reward=%.3f | val AUPRC=%.3f",
                     update, mcfg.updates, float(loss), float(rew.sum() / m_np.sum()),
                     quick_eval_auprc(policy, val_t))
    return policy, df, manifest


# ------------------------------------------------------------------------- eval
@torch.no_grad()
def team_scores(policy: SentinelMARL, tensors: EpisodeTensors) -> np.ndarray:
    """Per-hour team risk = max over organs of P(escalate); row order = (stay_id, hr)."""
    dev = _device()
    policy = policy.to(dev).eval()
    organ_obs, joint_obs, _, _, mask = tensors.to(dev)
    logits = policy.actor_logits(organ_obs)
    p_esc = torch.stack([torch.softmax(logits[o], -1)[..., 2] for o in policy.agents], -1).amax(-1)
    p = p_esc.cpu().numpy()
    out = []
    for i in range(tensors.N):
        out.append(p[i, : tensors.lengths[i]])
    return np.concatenate(out).astype(np.float32)


@torch.no_grad()
def quick_eval_auprc(policy: SentinelMARL, tensors: EpisodeTensors) -> float:
    from sklearn.metrics import average_precision_score
    score = team_scores(policy, tensors)
    # reconstruct per-hour y in the same order
    y = []
    for i in range(tensors.N):
        lead = tensors.leads[i, : tensors.lengths[i]]
        lab = tensors.labels[i]
        y.append(((lab == 1) & np.isfinite(lead) & (lead >= 1) & (lead <= 6)).astype(int))
    y = np.concatenate(y)
    if y.sum() == 0:
        return 0.0
    return float(average_precision_score(y, score))
