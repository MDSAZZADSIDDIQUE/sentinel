"""Unit tests for the pure SOFA component scoring (Vincent 1996 / Sepsis-3).

Clinical truth tables hand-derived from the SOFA definition. Also a property
test that the vectorized implementations agree with the scalar wrappers.
"""
from __future__ import annotations

import numpy as np
import pytest

from sentinel.labels import scoring as s


@pytest.mark.parametrize("pf,vent,expected", [
    (450, False, 0), (450, True, 0),
    (350, False, 1), (350, True, 1),
    (250, False, 2), (250, True, 2),
    (150, True, 3), (150, False, 2),   # 3/4 require ventilation; else capped at 2
    (90, True, 4), (90, False, 2),
    (399, False, 1), (299, False, 2), (199, True, 3), (99, True, 4),
])
def test_respiration(pf, vent, expected):
    assert s.score_respiration_one(pf, vent) == expected


@pytest.mark.parametrize("plt,expected", [
    (200, 0), (150, 0), (149, 1), (100, 1), (99, 2), (50, 2),
    (49, 3), (20, 3), (19, 4), (5, 4),
])
def test_coagulation(plt, expected):
    assert s.score_coagulation_one(plt) == expected


@pytest.mark.parametrize("bili,expected", [
    (0.5, 0), (1.19, 0), (1.2, 1), (1.9, 1), (2.0, 2), (5.9, 2),
    (6.0, 3), (11.9, 3), (12.0, 4), (20.0, 4),
])
def test_liver(bili, expected):
    assert s.score_liver_one(bili) == expected


@pytest.mark.parametrize("kwargs,expected", [
    (dict(map_mmhg=80), 0),
    (dict(map_mmhg=65), 1),
    (dict(map_mmhg=60, dopamine=3.0), 2),
    (dict(map_mmhg=60, dobutamine=5.0), 2),
    (dict(map_mmhg=60, dopamine=10.0), 3),
    (dict(map_mmhg=60, norepinephrine=0.05), 3),
    (dict(map_mmhg=60, epinephrine=0.10), 3),
    (dict(map_mmhg=60, norepinephrine=0.20), 4),
    (dict(map_mmhg=60, epinephrine=0.15), 4),
    (dict(map_mmhg=60, dopamine=20.0), 4),
])
def test_cardiovascular(kwargs, expected):
    assert s.score_cardiovascular_one(**kwargs) == expected


@pytest.mark.parametrize("gcs,expected", [
    (15, 0), (14, 1), (13, 1), (12, 2), (10, 2), (9, 3), (6, 3), (5, 4), (3, 4),
])
def test_cns(gcs, expected):
    assert s.score_cns_one(gcs) == expected


@pytest.mark.parametrize("creat,urine,expected", [
    (1.0, np.nan, 0), (1.2, np.nan, 1), (2.0, np.nan, 2),
    (3.5, np.nan, 3), (5.0, np.nan, 4),
    (1.0, 400, 3), (1.0, 150, 4),       # urine drives when creatinine is low
    (5.0, 450, 4),                       # worst of the two
])
def test_renal(creat, urine, expected):
    assert s.score_renal_one(creat, urine) == expected


def test_missing_inputs_score_zero():
    assert s.score_coagulation_one(np.nan) == 0
    assert s.score_liver_one(np.nan) == 0
    assert s.score_cns_one(np.nan) == 0
    assert s.score_cardiovascular_one(np.nan) == 0
    assert s.score_renal_one(np.nan, np.nan) == 0
    assert s.score_respiration_one(np.nan, True) == 0


def test_components_in_range():
    rng = np.random.default_rng(0)
    n = 5000
    out = [
        s.score_respiration(rng.uniform(50, 500, n), rng.integers(0, 2, n).astype(bool)),
        s.score_coagulation(rng.uniform(1, 400, n)),
        s.score_liver(rng.uniform(0, 30, n)),
        s.score_cardiovascular(rng.uniform(40, 100, n), rng.uniform(0, 25, n),
                               rng.uniform(0, 10, n), rng.uniform(0, 0.3, n),
                               rng.uniform(0, 0.3, n)),
        s.score_cns(rng.uniform(3, 15, n)),
        s.score_renal(rng.uniform(0.5, 6, n), rng.uniform(100, 2000, n)),
    ]
    for arr in out:
        assert arr.min() >= 0 and arr.max() <= 4


def test_vectorized_matches_scalar():
    rng = np.random.default_rng(1)
    for _ in range(200):
        plt = rng.uniform(1, 400)
        assert s.score_coagulation(np.array([plt]))[0] == s.score_coagulation_one(plt)
        bili = rng.uniform(0, 30)
        assert s.score_liver(np.array([bili]))[0] == s.score_liver_one(bili)
