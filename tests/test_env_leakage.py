"""Spec-D guards on the built feature tensor: no peeking at/after onset, and
episode length cannot proxy the label. Runs only when features are built.
"""
from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from sentinel.config import CohortConfig, FeatureConfig  # noqa: E402
from sentinel.labels.sepsis3 import sepsis3_path  # noqa: E402
from sentinel.paths import PATHS  # noqa: E402


def _mode():
    for m in ("full", "dev"):
        if (PATHS.features_root / f"hourly_{m}.parquet").exists():
            return m
    return None


MODE = _mode()
pytestmark = pytest.mark.skipif(MODE is None, reason="features not built")


@pytest.fixture(scope="module")
def feat():
    cfg = CohortConfig(mode=MODE)
    df = pd.read_parquet(PATHS.features_root / f"hourly_{MODE}.parquet")
    onset = pd.read_parquet(sepsis3_path(cfg))[["stay_id", "onset_hour"]]
    return df.merge(onset, on="stay_id", how="left")


def test_no_observation_at_or_after_onset(feat):
    """The hard leakage guard: positives are never observed at hr >= onset."""
    pos = feat[feat["label"] == 1]
    bad = pos[pos["hr"] >= pos["onset_hour"]]
    assert len(bad) == 0, f"{len(bad)} positive rows observed at/after onset (leakage)"


def test_positive_labels_only_in_alert_window(feat):
    fcfg = FeatureConfig.load()
    lead = feat["onset_hour"] - feat["hr"]
    should = (feat["label"] == 1) & (lead >= 1) & (lead <= fcfg.alert_window_hours)
    assert (feat["y"] == should.astype(int)).all(), "per-hour label != alert-window rule"
    assert (feat.loc[feat["label"] == 0, "y"] == 0).all(), "controls must have y=0"


def test_controls_have_no_onset_leak(feat):
    assert feat.loc[feat["label"] == 0, "t_to_onset"].isna().all()


def test_episode_length_not_a_class_proxy(feat):
    """LOS-proxy guard: positive and control episode lengths must overlap."""
    elen = feat.groupby("stay_id").agg(label=("label", "first"), n=("hr", "size"))
    pos_med = elen.loc[elen.label == 1, "n"].median()
    ctrl_med = elen.loc[elen.label == 0, "n"].median()
    assert 0.5 <= pos_med / ctrl_med <= 2.0, (
        f"episode length differs by class (pos={pos_med}, ctrl={ctrl_med}) — LOS proxy risk")
