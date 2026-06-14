"""Gold-standard end-to-end check of Sepsis-3 onset.

An *independent* pandas re-implementation of the onset rule (baseline + acute
rise + earliest-hour-in-window) is run on a sample of stays and must agree with
the production DuckDB pipeline, flag for flag and hour for hour. Two independent
implementations agreeing is what catches a subtly wrong derivation; a divergence
fails loudly. Runs only when the labels are built; otherwise skips.
"""
from __future__ import annotations

import numpy as np
import pytest

duckdb = pytest.importorskip("duckdb")
import pandas as pd  # noqa: E402

from sentinel.config import CohortConfig, LabelConfig  # noqa: E402
from sentinel.labels.sepsis3 import sepsis3_path  # noqa: E402
from sentinel.labels.sofa import sofa_path  # noqa: E402
from sentinel.labels.suspicion import suspicion_path  # noqa: E402


def _mode():
    for m in ("full", "dev"):
        cfg = CohortConfig(mode=m)
        if sofa_path(cfg).exists() and sepsis3_path(cfg).exists() and suspicion_path(cfg).exists():
            return m
    return None


MODE = _mode()
pytestmark = pytest.mark.skipif(MODE is None, reason="labels not built")


def _independent_onset(sofa_hours: pd.DataFrame, si_hour: int, lcfg: LabelConfig):
    """Re-derive (sepsis, onset_hour) for one stay from its hourly SOFA."""
    s = sofa_hours.set_index("hr")["sofa_total"]
    # baseline = min SOFA over the pre-suspicion admission window [0, min(si,W)]
    hi = min(max(si_hour, 0), lcfg.baseline_window_hours)
    pre = s[(s.index >= 0) & (s.index <= hi)]
    base = float(pre.min()) if len(pre) else float(lcfg.sofa_baseline)
    lo_w, hi_w = si_hour - lcfg.si_sofa_lookback_hours, si_hour + lcfg.si_sofa_lookahead_hours
    win = s[(s.index >= lo_w) & (s.index <= hi_w)]
    rise = win[(win - base) >= lcfg.sofa_increase_threshold]
    if len(rise) == 0:
        return 0, None
    return 1, int(rise.index.min())


def test_onset_matches_independent_recomputation():
    cfg = CohortConfig(mode=MODE)
    lcfg = LabelConfig.load()
    con = duckdb.connect()

    # sample suspected stays (only these can be septic), reproducibly
    susp = con.execute(f"""
        SELECT stay_id, si_hour FROM read_parquet('{suspicion_path(cfg).as_posix()}')
        WHERE has_suspicion = 1 USING SAMPLE 250 ROWS (reservoir, 7)
    """).df()
    assert len(susp) > 50
    ids = ",".join(str(int(s)) for s in susp["stay_id"])

    sofa = con.execute(f"""
        SELECT stay_id, hr, sofa_total FROM read_parquet('{sofa_path(cfg).as_posix()}')
        WHERE stay_id IN ({ids})
    """).df()
    sep = con.execute(f"""
        SELECT stay_id, sepsis3, onset_hour FROM read_parquet('{sepsis3_path(cfg).as_posix()}')
        WHERE stay_id IN ({ids})
    """).df().set_index("stay_id")

    n_checked = n_septic = 0
    for _, row in susp.iterrows():
        sid = int(row["stay_id"])
        exp_sep, exp_onset = _independent_onset(
            sofa[sofa["stay_id"] == sid], int(row["si_hour"]), lcfg)
        got = sep.loc[sid]
        assert int(got["sepsis3"]) == exp_sep, f"stay {sid}: sepsis flag mismatch"
        got_onset = None if pd.isna(got["onset_hour"]) else int(got["onset_hour"])
        assert got_onset == exp_onset, f"stay {sid}: onset {got_onset} != expected {exp_onset}"
        n_checked += 1
        n_septic += exp_sep
    assert n_checked > 50 and n_septic > 0  # exercised both classes
