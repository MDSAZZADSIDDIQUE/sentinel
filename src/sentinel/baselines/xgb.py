"""XGBoost baseline on per-hour feature vectors (point-in-time prediction).

Predicts the per-hour early-warning label y(t) from the features at hour t.
Small/CPU-hist to fit the hardware; class imbalance handled via
``scale_pos_weight``. Supports the measurement ablation through the feature set
passed in (physiology-only vs full).
"""
from __future__ import annotations

import numpy as np

from ..features.dataset import SplitData


def train_xgb(train: SplitData, seed: int = 0, n_estimators: int = 300,
              max_depth: int = 5):
    from xgboost import XGBClassifier

    n_pos = int(train.y.sum())
    n_neg = len(train.y) - n_pos
    spw = (n_neg / max(n_pos, 1))
    model = XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        tree_method="hist", n_jobs=6, random_state=seed,
        eval_metric="aucpr", objective="binary:logistic",
    )
    model.fit(train.X, train.y)
    return model


def predict_xgb(model, data: SplitData) -> np.ndarray:
    return model.predict_proba(data.X)[:, 1].astype(np.float32)
