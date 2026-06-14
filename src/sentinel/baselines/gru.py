"""GRU sequence baselines.

`GRUClassifier` — single-agent model over the full feature vector (the monolithic
deep baseline SENTINEL must match/beat).

`OrganEnsemble` — six per-organ GRUs trained *independently* (no shared
parameters, no shared critic) and combined with the same max-rule SENTINEL uses
at execution. This is the load-bearing control: if jointly-trained SENTINEL beats
this independent ensemble at a matched alarm budget, coordination adds value; if
not, the decomposition is just an ensemble and we say so.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..config import CohortConfig
from ..constants import ORGAN_SYSTEMS
from ..logging_utils import get_logger
from .torch_seq import EpisodeSeqDataset, predict_seq, train_seq

log = get_logger("baselines.gru")


class GRUClassifier(nn.Module):
    def __init__(self, n_features: int, hidden: int = 96, layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:  # X [B,T,F] -> [B,T]
        h, _ = self.gru(X)
        return self.head(h).squeeze(-1)


def train_gru(df, feature_cols, pos_weight, seed=0, hidden=96, epochs=25):
    tr = EpisodeSeqDataset(df[df["split"] == "train"], feature_cols)
    va = EpisodeSeqDataset(df[df["split"] == "val"], feature_cols)
    model = GRUClassifier(len(feature_cols), hidden=hidden)
    return train_seq(model, tr, va, pos_weight, epochs=epochs, seed=seed)


def predict_gru(model, df, split, feature_cols) -> np.ndarray:
    ds = EpisodeSeqDataset(df[df["split"] == split], feature_cols)
    return predict_seq(model, ds)


# --------------------------- independent ensemble ---------------------------
def train_organ_ensemble(df, manifest: dict, pos_weight: float, seed: int = 0,
                         hidden: int = 48, epochs: int = 25):
    """Train one independent GRU per organ on its own features + shared context."""
    from ..constants import SHARED_GROUP
    shared = manifest.get(SHARED_GROUP, [])
    models = {}
    for organ in ORGAN_SYSTEMS:
        cols = [c for c in (manifest[organ] + shared) if c in df.columns]
        tr = EpisodeSeqDataset(df[df["split"] == "train"], cols)
        va = EpisodeSeqDataset(df[df["split"] == "val"], cols)
        m = GRUClassifier(len(cols), hidden=hidden)
        log.info("    organ-ensemble: training %s (%d feats)", organ, len(cols))
        models[organ] = (train_seq(m, tr, va, pos_weight, epochs=epochs, seed=seed), cols)
    return models


def predict_organ_ensemble(models: dict, df, split: str) -> np.ndarray:
    """Per-hour team score = max over independently-trained organ predictors."""
    per_organ = []
    for organ, (model, cols) in models.items():
        ds = EpisodeSeqDataset(df[df["split"] == split], cols)
        per_organ.append(predict_seq(model, ds))
    return np.maximum.reduce(per_organ)
