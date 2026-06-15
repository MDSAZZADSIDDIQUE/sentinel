"""SENTINEL-Hybrid: per-organ supervised risk heads + an RL alarming policy.

The full-RL system hit a discrimination ceiling (~0.06 AUPRC) because RL-for-
utility on decomposed observations is a weak *discriminator*. The redesign splits
the two jobs cleanly:

  * PERCEPTION — six per-organ GRU **risk heads** trained by supervision (BCE on
    the onset label). Team risk = max over organs → a strong, interpretable, and
    (by max-combine) missing-signal-robust onset score. This is what's compared
    on AUPRC.
  * DECISION — a small **RL policy** over [the six organ risks + alert history]
    chooses maintain/watch/escalate, trained on the clinical utility (alarm-event
    budget, lead-time bonus, miss penalty). This is what's compared on the
    clinical operating-point metrics, vs a fixed threshold on the risk.

Keeps decomposition + interpretability + robustness, recovers discrimination, and
isolates what the RL actually buys (the operating policy, not the risk score).
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
from ..features.dataset import col_dropped, load_hourly
from ..logging_utils import get_logger
from ..paths import PATHS
from .mappo import EpisodeTensors, _device, _gae, _manifest, batched_reward

log = get_logger("agents.hybrid")


def _risk_cols(manifest, ablation):
    shared = [c for c in manifest[SHARED_GROUP] if not col_dropped(c, ablation)]
    return {o: [c for c in manifest[o] if not col_dropped(c, ablation)] + shared
            for o in ORGAN_SYSTEMS}


class OrganRiskHeads(nn.Module):
    """Six per-organ GRU risk heads (the agents' perception)."""

    def __init__(self, organ_dims: dict, hidden: int = 48):
        super().__init__()
        self.nets = nn.ModuleDict(
            {o: nn.GRU(d, hidden, batch_first=True) for o, d in organ_dims.items()})
        self.heads = nn.ModuleDict({o: nn.Linear(hidden, 1) for o in organ_dims})

    def organ_risk(self, organ_obs: dict) -> dict:
        out = {}
        for o in ORGAN_SYSTEMS:
            h, _ = self.nets[o](organ_obs[o])
            out[o] = self.heads[o](h).squeeze(-1)        # [B,T] logits
        return out

    def team_risk_logit(self, organ_obs: dict) -> torch.Tensor:
        r = self.organ_risk(organ_obs)
        return torch.stack([r[o] for o in ORGAN_SYSTEMS], -1).amax(-1)   # max-combine


class AlarmPolicy(nn.Module):
    """RL policy over [6 organ risks + alert history] -> action + value."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(len(ORGAN_SYSTEMS) + 2, hidden, batch_first=True)
        self.pi = nn.Linear(hidden, 3)
        self.v = nn.Linear(hidden, 1)


def _bce_masked(logits, y, mask, pos_weight):
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, y, pos_weight=pos_weight, reduction="none")
    return (loss * mask).sum() / mask.sum()


def train_risk_heads(cfg, mcfg, df, manifest):
    """Stage 1: supervised per-organ risk heads."""
    dev = _device()
    tt = EpisodeTensors(df[df["split"] == "train"], manifest, mcfg.ablation)
    vt = EpisodeTensors(df[df["split"] == "val"], manifest, mcfg.ablation)
    organ_dims = {o: len(c) for o, c in tt.organ_cols.items()}
    model = OrganRiskHeads(organ_dims).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    npos = int(sum(((tt.labels == 1)[:, None] & (np.isfinite(tt.leads) &
              (tt.leads >= 1) & (tt.leads <= 6))).sum() for _ in [0]))
    pw = torch.tensor([max((tt.mask.sum() - npos) / max(npos, 1), 1.0)], device=dev)

    def y_of(t):
        lead = t.leads
        return ((t.labels[:, None] == 1) & np.isfinite(lead) & (lead >= 1) & (lead <= 6)).astype(np.float32)

    best, best_state, bad = -1.0, None, 0
    for ep in range(40):
        model.train()
        idx = np.random.permutation(tt.N)
        for s in range(0, tt.N, 256):
            b = idx[s:s + 256]
            oo, _, _, _, mask = tt.to(dev, b)
            y = torch.as_tensor(y_of(tt)[b], device=dev)
            risk = model.organ_risk(oo)
            loss = sum(_bce_masked(risk[o], y, mask, pw) for o in ORGAN_SYSTEMS)
            opt.zero_grad(); loss.backward(); opt.step()
        auprc = _risk_auprc(model, vt)
        if auprc > best:
            best, bad = auprc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 5:
                break
    if best_state:
        model.load_state_dict(best_state)
    log.info("    risk heads: best val AUPRC=%.3f", best)
    return model


@torch.no_grad()
def risk_scores(model: OrganRiskHeads, tensors: EpisodeTensors) -> np.ndarray:
    dev = _device()
    model = model.to(dev).eval()
    oo, _, _, _, _ = tensors.to(dev)
    p = torch.sigmoid(model.team_risk_logit(oo)).cpu().numpy()
    return np.concatenate([p[i, : tensors.lengths[i]] for i in range(tensors.N)]).astype(np.float32)


@torch.no_grad()
def _risk_auprc(model, tensors) -> float:
    from sklearn.metrics import average_precision_score
    score = risk_scores(model, tensors)
    y = []
    for i in range(tensors.N):
        lead = tensors.leads[i, : tensors.lengths[i]]
        y.append(((tensors.labels[i] == 1) & np.isfinite(lead) & (lead >= 1) & (lead <= 6)).astype(int))
    y = np.concatenate(y)
    return float(average_precision_score(y, score)) if y.sum() else 0.0


def train_hybrid(cfg: CohortConfig, mcfg: MARLConfig, rcfg: RewardConfig | None = None, df=None):
    """Stage 1 (risk heads) then Stage 2 (RL alarm policy over the risks)."""
    rcfg = rcfg or RewardConfig()
    torch.manual_seed(mcfg.seed); np.random.seed(mcfg.seed)
    dev = _device()
    df = df if df is not None else load_hourly(cfg)
    manifest = _manifest(cfg)
    risk = train_risk_heads(cfg, mcfg, df, manifest)

    tt = EpisodeTensors(df[df["split"] == "train"], manifest, mcfg.ablation)
    policy = AlarmPolicy().to(dev)
    opt = torch.optim.Adam(policy.parameters(), lr=mcfg.lr)

    @torch.no_grad()
    def risk_feat(tensors, idx):
        oo, _, _, _, _ = tensors.to(dev, idx)
        r = risk.to(dev).eval().organ_risk(oo)
        return torch.stack([torch.sigmoid(r[o]) for o in ORGAN_SYSTEMS], -1)   # [B,T,6]

    for update in range(mcfg.updates):
        policy.train()
        idx = np.random.choice(tt.N, min(mcfg.batch_episodes, tt.N), replace=False)
        _, _, labels, leads, mask = tt.to(dev, idx)
        rf = risk_feat(tt, idx)                       # [B,T,6]
        T = rf.shape[1]
        # sequential rollout: alert history depends on sampled actions
        h = None
        alerted = torch.zeros(len(idx), dtype=torch.bool, device=dev)
        last = torch.full((len(idx),), -1e9, device=dev)
        acts = torch.zeros(len(idx), T, dtype=torch.long, device=dev)
        alert_feat = torch.zeros(len(idx), T, 2, device=dev)
        with torch.no_grad():
            for t in range(T):
                af = torch.stack([alerted.float(), torch.clamp(t - last, 0, 24) / 24], -1)
                alert_feat[:, t] = af
                x = torch.cat([rf[:, t], af], -1).unsqueeze(1)
                out, h = policy.gru(x, h)
                a = Categorical(logits=policy.pi(out.squeeze(1))).sample()
                acts[:, t] = a
                esc = a == 2
                last = torch.where(esc, torch.full_like(last, float(t)), last)
                alerted = alerted | esc
        obs = torch.cat([rf, alert_feat], -1)         # [B,T,8] fixed for the update
        with torch.no_grad():
            ho, _ = policy.gru(obs)
            old_lp = Categorical(logits=policy.pi(ho)).log_prob(acts)
            values = policy.v(ho).squeeze(-1)
        rew = batched_reward(acts.cpu().numpy(), labels.cpu().numpy(),
                             leads.cpu().numpy(), mask.cpu().numpy(), rcfg)
        adv_np, ret_np = _gae(rew, values.cpu().numpy(), mask.cpu().numpy(),
                              mcfg.gamma, mcfg.gae_lambda)
        adv = torch.as_tensor(adv_np, device=dev)
        sel = mask > 0
        adv = (adv - adv[sel].mean()) / (adv[sel].std() + 1e-6)
        ret = torch.as_tensor(ret_np, device=dev)
        for _ in range(mcfg.ppo_epochs):
            ho, _ = policy.gru(obs)
            d = Categorical(logits=policy.pi(ho))
            ratio = torch.exp(d.log_prob(acts) - old_lp)
            s1, s2 = ratio * adv, torch.clamp(ratio, 1 - mcfg.clip, 1 + mcfg.clip) * adv
            ploss = (-torch.min(s1, s2) * mask).sum() / mask.sum()
            vloss = (((policy.v(ho).squeeze(-1) - ret) ** 2) * mask).sum() / mask.sum()
            ent = (d.entropy() * mask).sum() / mask.sum()
            loss = ploss + mcfg.value_coef * vloss - mcfg.entropy_coef * ent
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), mcfg.grad_clip); opt.step()
        if update % mcfg.eval_every == 0 or update == mcfg.updates - 1:
            log.info("    hybrid policy update %d/%d | mean reward=%.3f",
                     update, mcfg.updates, float(rew.sum() / mask.cpu().numpy().sum()))
    return risk, policy, df, manifest


@torch.no_grad()
def policy_alarm_scores(risk: OrganRiskHeads, policy: AlarmPolicy,
                        tensors: EpisodeTensors) -> np.ndarray:
    """P(escalate) from the RL policy along its greedy trajectory (clinical decisions)."""
    dev = _device()
    risk = risk.to(dev).eval(); policy = policy.to(dev).eval()
    oo, _, _, _, _ = tensors.to(dev)
    r = risk.organ_risk(oo)
    rf = torch.stack([torch.sigmoid(r[o]) for o in ORGAN_SYSTEMS], -1)
    T = rf.shape[1]
    h = None
    B = rf.shape[0]
    alerted = torch.zeros(B, dtype=torch.bool, device=dev)
    last = torch.full((B,), -1e9, device=dev)
    pesc = torch.zeros(B, T, device=dev)
    for t in range(T):
        af = torch.stack([alerted.float(), torch.clamp(t - last, 0, 24) / 24], -1)
        out, h = policy.gru(torch.cat([rf[:, t], af], -1).unsqueeze(1), h)
        probs = torch.softmax(policy.pi(out.squeeze(1)), -1)
        pesc[:, t] = probs[:, 2]
        a = probs.argmax(-1)
        esc = a == 2
        last = torch.where(esc, torch.full_like(last, float(t)), last)
        alerted = alerted | esc
    p = pesc.cpu().numpy()
    return np.concatenate([p[i, : tensors.lengths[i]] for i in range(tensors.N)]).astype(np.float32)
