"""Action-history-conditioned actor variant (methodology justification).

The observation-only SENTINEL actor cannot condition on its own alert history,
yet the reward couples actions over time (refractory, one-time lead bonus,
terminal miss). This variant augments each actor's input with [already-alerted,
hours-since-last-alarm], which depend on the policy's own sampled actions — so
the ROLLOUT is sequential (step-by-step). During the PPO UPDATE the actions (and
thus the alert features) are fixed, so the augmented observations are fixed and
the update stays a single batched forward.

If this variant ≈ the observation-only policy on dev, the one-forward-pass
simplification is empirically justified (a far stronger claim than asserting it).
Reuses the MAPPO networks/reward/GAE.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Categorical

from ..config import CohortConfig, MARLConfig
from ..env.reward import RewardConfig
from ..features.dataset import load_hourly
from ..logging_utils import get_logger
from .mappo import (EpisodeTensors, SentinelMARL, _device, _gae, _manifest,
                    batched_reward)

log = get_logger("agents.ac")
ALERT_FEAT = 2


def _aug_dims(organ_dims, joint_dim):
    return {o: d + ALERT_FEAT for o, d in organ_dims.items()}, joint_dim + ALERT_FEAT


@torch.no_grad()
def _rollout_alert(policy: SentinelMARL, organ_obs: dict, T: int, greedy: bool):
    """Sequential rollout to determine actions + alert-history features [B,T,2]."""
    dev = next(policy.parameters()).device
    B = next(iter(organ_obs.values())).shape[0]
    agents = policy.agents
    h = {o: None for o in agents}
    last_alarm = torch.full((B,), -1e9, device=dev)
    alerted = torch.zeros(B, dtype=torch.bool, device=dev)
    actions = {o: torch.zeros(B, T, dtype=torch.long, device=dev) for o in agents}
    alert_feat = torch.zeros(B, T, ALERT_FEAT, device=dev)
    for t in range(T):
        since = torch.clamp(t - last_alarm, 0, 24) / 24.0
        af = torch.stack([alerted.float(), since], -1)            # [B,2]
        alert_feat[:, t] = af
        a_t = []
        for o in agents:
            x = torch.cat([organ_obs[o][:, t], af], -1).unsqueeze(1)   # [B,1,F+2]
            out, h[o] = policy.actors[o].gru(x, h[o])
            logits = policy.actors[o].head(out).squeeze(1)            # [B,3]
            a = logits.argmax(-1) if greedy else Categorical(logits=logits).sample()
            actions[o][:, t] = a
            a_t.append(a)
        eff = torch.stack(a_t, -1).amax(-1)                          # [B]
        is_esc = eff == 2
        last_alarm = torch.where(is_esc, torch.full_like(last_alarm, float(t)), last_alarm)
        alerted = alerted | is_esc
    return actions, alert_feat


def _aug(organ_obs, joint_obs, alert_feat):
    organ = {o: torch.cat([v, alert_feat], -1) for o, v in organ_obs.items()}
    return organ, torch.cat([joint_obs, alert_feat], -1)


def train_action_conditioned(cfg: CohortConfig, mcfg: MARLConfig,
                             rcfg: RewardConfig | None = None, df=None):
    rcfg = rcfg or RewardConfig()
    torch.manual_seed(mcfg.seed)
    np.random.seed(mcfg.seed)
    dev = _device()
    df = df if df is not None else load_hourly(cfg)
    manifest = _manifest(cfg)
    train_t = EpisodeTensors(df[df["split"] == "train"], manifest, mcfg.ablation)
    val_t = EpisodeTensors(df[df["split"] == "val"], manifest, mcfg.ablation)
    organ_dims = {o: len(c) for o, c in train_t.organ_cols.items()}
    a_organ, a_joint = _aug_dims(organ_dims, len(train_t.joint_cols))
    policy = SentinelMARL(a_organ, a_joint, mcfg).to(dev)
    opt = torch.optim.Adam(policy.parameters(), lr=mcfg.lr)
    mappo = mcfg.algo == "mappo"
    agents = policy.agents
    log.info("  MARL-AC[%s] train=%d val=%d episodes", mcfg.algo, train_t.N, val_t.N)

    for update in range(mcfg.updates):
        policy.train()
        idx = np.random.choice(train_t.N, min(mcfg.batch_episodes, train_t.N), replace=False)
        organ_obs, joint_obs, labels, leads, mask = train_t.to(dev, idx)
        T = organ_obs[agents[0]].shape[1]
        actions, alert_feat = _rollout_alert(policy, organ_obs, T, greedy=False)
        aorgan, ajoint = _aug(organ_obs, joint_obs, alert_feat)
        with torch.no_grad():
            logits = policy.actor_logits(aorgan)
            old_lp = {o: Categorical(logits=logits[o]).log_prob(actions[o]) for o in agents}
            values = policy.values(aorgan, ajoint)
            eff = torch.stack([actions[o] for o in agents], -1).amax(-1)
        rew = batched_reward(eff.cpu().numpy(), labels.cpu().numpy(),
                             leads.cpu().numpy(), mask.cpu().numpy(), rcfg)
        adv, ret = {}, {}
        for o in agents:
            a, r = _gae(rew, values[o].cpu().numpy(), mask.cpu().numpy(),
                        mcfg.gamma, mcfg.gae_lambda)
            at = torch.as_tensor(a, device=dev)
            sel = mask > 0
            adv[o] = (at - at[sel].mean()) / (at[sel].std() + 1e-6)
            ret[o] = torch.as_tensor(r, device=dev)
            if mappo:
                break
        if mappo:
            adv = {o: adv[agents[0]] for o in agents}
            ret_shared = ret[agents[0]]
        for _ in range(mcfg.ppo_epochs):
            logits = policy.actor_logits(aorgan)
            values_new = policy.values(aorgan, ajoint)
            ploss = torch.zeros((), device=dev)
            ent = torch.zeros((), device=dev)
            denom = mask.sum()
            for o in agents:
                d = Categorical(logits=logits[o])
                ratio = torch.exp(d.log_prob(actions[o]) - old_lp[o])
                s1, s2 = ratio * adv[o], torch.clamp(ratio, 1 - mcfg.clip, 1 + mcfg.clip) * adv[o]
                ploss += (-torch.min(s1, s2) * mask).sum() / denom
                ent += (d.entropy() * mask).sum() / denom
            if mappo:
                vloss = (((values_new[agents[0]] - ret_shared) ** 2) * mask).sum() / denom
            else:
                vloss = sum((((values_new[o] - ret[o]) ** 2) * mask).sum() / denom for o in agents)
            loss = ploss + mcfg.value_coef * vloss - mcfg.entropy_coef * ent
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), mcfg.grad_clip)
            opt.step()
        if update % mcfg.eval_every == 0 or update == mcfg.updates - 1:
            log.info("    AC update %d/%d | val AUPRC=%.3f",
                     update, mcfg.updates, ac_eval_auprc(policy, val_t))
    return policy, df, manifest


@torch.no_grad()
def team_scores_ac(policy: SentinelMARL, tensors: EpisodeTensors) -> np.ndarray:
    dev = _device()
    policy = policy.to(dev).eval()
    organ_obs, joint_obs, _, _, _ = tensors.to(dev)
    T = organ_obs[policy.agents[0]].shape[1]
    _, alert_feat = _rollout_alert(policy, organ_obs, T, greedy=True)
    aorgan, _ = _aug(organ_obs, joint_obs, alert_feat)
    logits = policy.actor_logits(aorgan)
    p = torch.stack([torch.softmax(logits[o], -1)[..., 2] for o in policy.agents], -1).amax(-1)
    p = p.cpu().numpy()
    return np.concatenate([p[i, : tensors.lengths[i]] for i in range(tensors.N)]).astype(np.float32)


@torch.no_grad()
def ac_eval_auprc(policy, tensors) -> float:
    from sklearn.metrics import average_precision_score
    score = team_scores_ac(policy, tensors)
    y = []
    for i in range(tensors.N):
        lead = tensors.leads[i, : tensors.lengths[i]]
        y.append(((tensors.labels[i] == 1) & np.isfinite(lead) & (lead >= 1) & (lead <= 6)).astype(int))
    y = np.concatenate(y)
    return float(average_precision_score(y, score)) if y.sum() else 0.0
