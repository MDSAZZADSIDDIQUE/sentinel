"""Shared PyTorch sequence-model harness (GRU baselines + reused in Phase 4).

Episodes are variable-length hourly sequences; we pad per batch and mask the
loss. GRUs are causal (left-to-right), so the prediction at hour t uses only
hours <= t — no leakage by construction. FP32 on the GTX 1650 (no mixed
precision; no tensor cores). Predictions are returned in the same row order as
`features.dataset.load_split` (sorted by stay_id, hr) so metrics align.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..logging_utils import get_logger

log = get_logger("baselines.torch")


def quick_auprc(y: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    if y.sum() == 0 or y.sum() == len(y):
        return 0.0
    return float(average_precision_score(y, score))


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EpisodeSeqDataset(Dataset):
    """Per-episode (X[T,F], y[T]) from a split dataframe + feature columns."""

    def __init__(self, df, feature_cols: list[str]):
        df = df.sort_values(["stay_id", "hr"])
        self.episodes = []
        self.stay_order = []
        for sid, ep in df.groupby("stay_id", sort=False):
            X = ep[feature_cols].to_numpy(dtype=np.float32)
            y = ep["y"].to_numpy(dtype=np.float32)
            self.episodes.append((X, y))
            self.stay_order.append(int(sid))
        self.n_features = len(feature_cols)

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, i):
        X, y = self.episodes[i]
        return torch.from_numpy(X), torch.from_numpy(y)


def _collate(batch):
    lengths = torch.tensor([x.shape[0] for x, _ in batch])
    T = int(lengths.max())
    F = batch[0][0].shape[1]
    X = torch.zeros(len(batch), T, F)
    Y = torch.zeros(len(batch), T)
    M = torch.zeros(len(batch), T)
    for i, (x, y) in enumerate(batch):
        t = x.shape[0]
        X[i, :t] = x
        Y[i, :t] = y
        M[i, :t] = 1.0
    return X, Y, M, lengths


def _mask_groups_(X: torch.Tensor, groups, mask_prob: float) -> torch.Tensor:
    """Augmentation: with prob `mask_prob`, zero ONE random feature group per
    episode (in place), mimicking the test-time organ-blinding stress. Used to
    give the monolithic GRU a missingness-aware training signal, so that any
    residual robustness gap can be attributed to organ DECOMPOSITION rather than
    merely to whether the model saw input dropouts during training. `groups` is a
    list of LongTensor column-index sets (one per organ). No-op when disabled."""
    if not groups or mask_prob <= 0.0:
        return X
    drop = torch.rand(X.shape[0], device=X.device) < mask_prob
    for bi in torch.nonzero(drop, as_tuple=True)[0].tolist():
        g = int(torch.randint(0, len(groups), (1,)).item())
        X[bi, :, groups[g]] = 0.0
    return X


def train_seq(model: nn.Module, train_ds, val_ds, pos_weight: float,
              epochs: int = 25, batch_size: int = 64, lr: float = 1e-3,
              patience: int = 5, seed: int = 0,
              mask_groups: list | None = None, mask_prob: float = 0.0) -> nn.Module:
    torch.manual_seed(seed)
    dev = device()
    model = model.to(dev)
    groups = ([torch.as_tensor(g, dtype=torch.long, device=dev) for g in mask_groups]
              if mask_groups else None)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pw = torch.tensor([pos_weight], device=dev)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")
    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate, num_workers=0)

    best_auprc, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        for X, Y, Msk, _ in tl:
            X, Y, Msk = X.to(dev), Y.to(dev), Msk.to(dev)
            X = _mask_groups_(X, groups, mask_prob)
            opt.zero_grad()
            logits = model(X)
            loss = (loss_fn(logits, Y) * Msk).sum() / Msk.sum()
            loss.backward()
            opt.step()
        # val AUPRC for early stopping
        vy, vs = predict_seq(model, val_ds, return_truth=True)
        auprc = quick_auprc(vy, vs)
        if auprc > best_auprc:
            best_auprc, bad = auprc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    log.info("    trained %d ep, best val AUPRC=%.3f", ep + 1, best_auprc)
    return model


@torch.no_grad()
def predict_seq(model: nn.Module, ds, return_truth: bool = False, batch_size: int = 128):
    dev = device()
    model = model.to(dev).eval()
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate, num_workers=0)
    scores, truth = [], []
    for X, Y, Msk, lengths in dl:
        X = X.to(dev)
        p = torch.sigmoid(model(X)).cpu().numpy()
        for i, t in enumerate(lengths.tolist()):
            scores.append(p[i, :t])
            if return_truth:
                truth.append(Y[i, :t].numpy())
    scores = np.concatenate(scores).astype(np.float32)
    if return_truth:
        return np.concatenate(truth).astype(np.int64), scores
    return scores
