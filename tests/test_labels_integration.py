"""Integration tests on the *built* labels (validates the derivation incl. SQL).

These run only if a label parquet exists (full or dev); otherwise they skip, so
the suite still passes on a fresh checkout without MIMIC. They check structural
invariants and tie the persisted SOFA back to the unit-tested pure scoring
functions (closing the loop between the builder and `scoring.py`).
"""
from __future__ import annotations

import numpy as np
import pytest

duckdb = pytest.importorskip("duckdb")

from sentinel.config import CohortConfig, LabelConfig  # noqa: E402
from sentinel.labels import scoring  # noqa: E402
from sentinel.labels.sepsis3 import sepsis3_path  # noqa: E402
from sentinel.labels.sofa import sofa_path  # noqa: E402
from sentinel.labels.suspicion import suspicion_path  # noqa: E402


def _available_mode() -> str | None:
    for mode in ("full", "dev"):
        cfg = CohortConfig(mode=mode)
        if sofa_path(cfg).exists() and sepsis3_path(cfg).exists():
            return mode
    return None


MODE = _available_mode()
pytestmark = pytest.mark.skipif(MODE is None, reason="labels not built; run build-labels")


@pytest.fixture(scope="module")
def cfg():
    return CohortConfig(mode=MODE)


def test_sofa_invariants(cfg):
    p = sofa_path(cfg).as_posix()
    con = duckdb.connect()
    comps = "+".join(scoring.SOFA_COMPONENTS)
    bad_sum, bad_range, neg_hr, bad_total = con.execute(f"""
        SELECT
          SUM(CASE WHEN sofa_total <> ({comps}) THEN 1 ELSE 0 END),
          SUM(CASE WHEN sofa_respiration NOT BETWEEN 0 AND 4
                    OR sofa_renal NOT BETWEEN 0 AND 4
                    OR sofa_cns NOT BETWEEN 0 AND 4 THEN 1 ELSE 0 END),
          SUM(CASE WHEN hr < 0 THEN 1 ELSE 0 END),
          SUM(CASE WHEN sofa_total NOT BETWEEN 0 AND 24 THEN 1 ELSE 0 END)
        FROM read_parquet('{p}')
    """).fetchone()
    assert bad_sum == 0, "sofa_total must equal the sum of its components"
    assert bad_range == 0, "components must be in [0, 4]"
    assert neg_hr == 0, "no negative hour indices (leakage guard)"
    assert bad_total == 0, "total SOFA must be in [0, 24]"


def test_sepsis3_invariants(cfg):
    p = sepsis3_path(cfg).as_posix()
    lcfg = LabelConfig.load()
    thr = lcfg.sofa_increase_threshold
    con = duckdb.connect()
    no_susp, no_onset, low_rise, out_window = con.execute(f"""
        SELECT
          SUM(CASE WHEN sepsis3=1 AND has_suspicion<>1 THEN 1 ELSE 0 END),
          SUM(CASE WHEN sepsis3=1 AND onset_hour IS NULL THEN 1 ELSE 0 END),
          SUM(CASE WHEN sepsis3=1 AND (sofa_at_onset - baseline_sofa) < {thr} THEN 1 ELSE 0 END),
          SUM(CASE WHEN sepsis3=1 AND (onset_hour < si_hour - {lcfg.si_sofa_lookback_hours}
                                    OR onset_hour > si_hour + {lcfg.si_sofa_lookahead_hours})
               THEN 1 ELSE 0 END)
        FROM read_parquet('{p}')
    """).fetchone()
    assert no_susp == 0, "sepsis3 requires suspicion of infection"
    assert no_onset == 0, "septic stays must have an onset hour"
    assert low_rise == 0, "acute SOFA rise at onset must meet the threshold"
    assert out_window == 0, "onset must lie within the suspicion window"


def test_persisted_sofa_matches_pure_scoring(cfg):
    """Recompute components from stored driving values; must match the builder."""
    p = sofa_path(cfg).as_posix()
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT platelets, bilirubin_total, gcs_total,
               sofa_coagulation, sofa_liver, sofa_cns
        FROM read_parquet('{p}') USING SAMPLE 20000 ROWS (reservoir, 42)
    """).df()
    assert (scoring.score_coagulation(df["platelets"].to_numpy())
            == df["sofa_coagulation"].to_numpy()).all()
    assert (scoring.score_liver(df["bilirubin_total"].to_numpy())
            == df["sofa_liver"].to_numpy()).all()
    assert (scoring.score_cns(df["gcs_total"].to_numpy())
            == df["sofa_cns"].to_numpy()).all()


def test_suspicion_present_for_septic(cfg):
    sp = suspicion_path(cfg).as_posix()
    con = duckdb.connect()
    n, n_susp = con.execute(
        f"SELECT COUNT(*), SUM(has_suspicion) FROM read_parquet('{sp}')"
    ).fetchone()
    assert 0 < n_susp <= n
