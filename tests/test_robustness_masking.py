"""Tier C control — organ-block masking augmentation (`_mask_groups_`).

The missingness-aware monolith ("GRU-masked") trains the single GRU with the
SAME perturbation the robustness harness applies at test time: zero one random
organ's columns per episode. These gates pin the augmentation's contract so the
control is fair — all-or-nothing per organ group, at most one group per episode,
zeroed across the full time axis, and a true no-op when disabled. CPU-only, no
data, no model: always run.
"""
from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from sentinel.baselines.torch_seq import _mask_groups_  # noqa: E402

GROUPS = [[0, 1], [2, 3], [4, 5]]  # 3 organs x 2 cols over F=6


def _tensor_groups():
    return [torch.as_tensor(g, dtype=torch.long) for g in GROUPS]


def test_mask_prob_zero_is_noop():
    X = torch.ones(8, 5, 6)
    out = _mask_groups_(X, _tensor_groups(), mask_prob=0.0)
    assert torch.equal(out, torch.ones(8, 5, 6))


def test_no_groups_is_noop():
    X = torch.ones(4, 5, 6)
    assert torch.equal(_mask_groups_(X, None, mask_prob=1.0), torch.ones(4, 5, 6))


def test_mask_prob_one_zeros_exactly_one_whole_group_per_episode():
    torch.manual_seed(0)
    B, T, F = 64, 5, 6
    out = _mask_groups_(torch.ones(B, T, F), _tensor_groups(), mask_prob=1.0)
    group_cols = [set(g) for g in GROUPS]
    for bi in range(B):
        # a column is "dark" iff it is zero across the full time axis
        zero_cols = set((out[bi].abs().sum(0) == 0).nonzero(as_tuple=True)[0].tolist())
        assert zero_cols in group_cols, f"row {bi}: {zero_cols} is not one whole organ group"


def test_masking_is_in_place():
    X = torch.ones(4, 3, 6)
    assert _mask_groups_(X, _tensor_groups(), mask_prob=1.0) is X
