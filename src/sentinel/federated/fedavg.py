"""Federated training across care-unit sites (privacy story, Phase 5).

Sites = `first_careunit` — genuinely non-IID populations (MICU ~54% sepsis vs
Neuro ~16%). FedAvg trains a shared team-risk model WITHOUT pooling raw data:
each site trains locally, only weights are averaged (sample-weighted) each round.
We compare federated vs centralized (pooled) on test + external CVICU; the AUPRC
gap is the **privacy cost** of never centralizing patient data.

Self-contained FedAvg (no Flower) per the hardware addendum. Uses the monolithic
team-risk GRU as the shared model (one model to average cleanly across sites).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..config import CohortConfig
from ..features.dataset import load_hourly, load_split
from ..features.partition import partition_path
from ..logging_utils import get_logger
from ..paths import PATHS
from ..baselines.gru import GRUClassifier
from ..baselines.torch_seq import EpisodeSeqDataset, _collate, device, predict_seq

log = get_logger("federated")
MIN_SITE_STAYS = 200


def _sites(cfg: CohortConfig, df) -> dict[str, list]:
    """Map site -> list of train stay_ids (internal sites only; external held out)."""
    part = pd.read_parquet(partition_path(cfg))
    tr = part[(part["split"] == "train") & part["label"].notna()]
    out = {}
    for site, g in tr.groupby("site"):
        if len(g) >= MIN_SITE_STAYS:
            out[site] = g["stay_id"].tolist()
    return out


def _local_update(state, ds, feature_cols, pw, epochs, lr, dev):
    model = GRUClassifier(len(feature_cols)).to(dev)
    model.load_state_dict(state)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=dev), reduction="none")
    dl = DataLoader(ds, batch_size=64, shuffle=True, collate_fn=_collate, num_workers=0)
    for _ in range(epochs):
        for X, Y, M, _ in dl:
            X, Y, M = X.to(dev), Y.to(dev), M.to(dev)
            opt.zero_grad()
            loss = (loss_fn(model(X), Y) * M).sum() / M.sum()
            loss.backward()
            opt.step()
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def _fedavg(states, weights):
    total = sum(weights)
    avg = {}
    for k in states[0]:
        avg[k] = sum(w * s[k] for s, w in zip(states, weights)) / total
    return avg


def train_federated(cfg, df, feature_cols, pw, rounds=15, local_epochs=1, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    dev = device()
    sites = _sites(cfg, df)
    log.info("  federated: %d sites (%s)", len(sites),
             ", ".join(f"{s.split('(')[-1].rstrip(')')}:{len(ids)}" for s, ids in sites.items()))
    # one dataset per site (train episodes of that site)
    site_ds, site_n = {}, {}
    for site, ids in sites.items():
        sub = df[(df["split"] == "train") & df["stay_id"].isin(ids)]
        site_ds[site] = EpisodeSeqDataset(sub, feature_cols)
        site_n[site] = len(ids)
    g = GRUClassifier(len(feature_cols)).to(dev)
    state = {k: v.detach().cpu() for k, v in g.state_dict().items()}
    for r in range(rounds):
        states, weights = [], []
        for site in sites:
            states.append(_local_update(state, site_ds[site], feature_cols, pw,
                                        local_epochs, lr, dev))
            weights.append(site_n[site])
        state = _fedavg(states, weights)
    g.load_state_dict(state)
    return g


def run(cfg: CohortConfig | None = None, seeds=(0, 1, 2), rounds=15) -> None:
    from ..baselines.gru import train_gru

    cfg = cfg or CohortConfig.load()
    t0 = time.perf_counter()
    df = load_hourly(cfg)
    tr = load_split(cfg, "train", df=df)
    pw = (len(tr.y) - int(tr.y.sum())) / max(int(tr.y.sum()), 1)
    splits = [s for s in ("test", "external") if (df["split"] == s).any()]

    from ..eval.metrics import discrimination
    res = {}
    for seed in seeds:
        log.info("seed %d: centralized + federated", seed)
        cen = train_gru(df, tr.feature_names, pw, seed=seed)          # pooled
        fed = train_federated(cfg, df, tr.feature_names, pw, rounds=rounds, seed=seed)
        for split in splits:
            data = load_split(cfg, split, df=df)
            from ..baselines.gru import predict_gru
            res.setdefault(("Centralized", split), []).append(
                discrimination(data.y, predict_gru(cen, df, split, tr.feature_names)).auprc)
            res.setdefault(("Federated", split), []).append(
                discrimination(data.y, predict_seq(fed, EpisodeSeqDataset(
                    df[df["split"] == split], tr.feature_names))).auprc)

    L = [f"# SENTINEL — Federated vs centralized ({cfg.mode} cohort)\n",
         f"_FedAvg over care-unit sites ({rounds} rounds), no data pooling; mean AUPRC over "
         f"{len(seeds)} seed(s). Gap = privacy cost of never centralizing data._\n",
         "| Split | Centralized | Federated | Privacy cost (Δ) |", "|---|---|---|---|"]
    for split in splits:
        c = float(np.mean(res[("Centralized", split)]))
        f = float(np.mean(res[("Federated", split)]))
        L.append(f"| {split} | {c:.3f} | {f:.3f} | {f-c:+.3f} |")
    PATHS.reports_root.mkdir(parents=True, exist_ok=True)
    out = PATHS.reports_root / f"federated_{cfg.mode}.md"
    out.write_text("\n".join(L), encoding="utf-8")
    log.info("wrote %s (%.1fs)", out, time.perf_counter() - t0)
