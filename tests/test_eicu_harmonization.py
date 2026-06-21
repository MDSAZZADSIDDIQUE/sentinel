"""Phase 9 eICU harmonization invariants (Step 4).

Asserts the external-validation contract the spec requires:
  * the eICU hourly feature frame's schema EQUALS the saved MIMIC manifest
    (identical organ grouping, column names + order), and
  * the eICU normalizer's parameters ARE the MIMIC training-split ones
    (never refit on eICU).

Skips cleanly if the eICU features have not been built yet.
"""
from __future__ import annotations

import json

import pytest

pd = pytest.importorskip("pandas")

from sentinel.config import CohortConfig  # noqa: E402
from sentinel.features.build import hourly_path  # noqa: E402
from sentinel.features.dataset import feature_columns  # noqa: E402
from sentinel.paths import PATHS  # noqa: E402

EICU = CohortConfig(mode="eicu")
FULL = CohortConfig(mode="full")


def _mpath(mode: str, kind: str):
    return PATHS.features_root / f"{kind}_{mode}.json"


_BUILT = (hourly_path(EICU).exists()
          and _mpath("eicu", "feature_manifest").exists()
          and _mpath("eicu", "norm_stats").exists()
          and _mpath("full", "feature_manifest").exists()
          and _mpath("full", "norm_stats").exists())
pytestmark = pytest.mark.skipif(
    not _BUILT, reason="eICU features / MIMIC full normalizer not built")


def _load(mode: str, kind: str) -> dict:
    with _mpath(mode, kind).open() as fh:
        return json.load(fh)


def test_manifest_equals_mimic():
    """feature_manifest_eicu == feature_manifest_full (schema parity by name+order)."""
    assert _load("eicu", "feature_manifest") == _load("full", "feature_manifest")


def test_normalizer_is_mimic():
    """norm_stats_eicu params ARE the MIMIC train-split stats (never refit)."""
    eicu, full = _load("eicu", "norm_stats"), _load("full", "norm_stats")
    assert eicu.keys() == full.keys()
    for k in full:
        assert eicu[k]["mean"] == pytest.approx(full[k]["mean"], rel=0, abs=0), k
        assert eicu[k]["std"] == pytest.approx(full[k]["std"], rel=0, abs=0), k


def test_hourly_schema_parity():
    """hourly_eicu has every manifest feature column (in manifest order) + meta."""
    manifest = _load("eicu", "feature_manifest")
    flat = [c for cols in manifest.values() for c in cols]
    cols = pd.read_parquet(hourly_path(EICU)).columns.tolist()

    meta = ["stay_id", "hr", "split", "label", "y", "t_to_onset"]
    for m in meta:
        assert m in cols, f"missing meta column {m}"
    for f in flat:
        assert f in cols, f"missing manifest feature {f}"
    # feature columns appear in the SAME relative order as the manifest
    present = [c for c in cols if c in set(flat)]
    assert present == flat, "feature column order diverges from the MIMIC manifest"


def test_all_external_split():
    df = pd.read_parquet(hourly_path(EICU), columns=["split"])
    assert (df["split"] == "external").all(), "all eICU stays must be the external split"


def test_loadable_external_split():
    """load_split's feature set is identical to MIMIC's (frozen model can score it)."""
    e_cols, _ = feature_columns(_load("eicu", "feature_manifest"))
    f_cols, _ = feature_columns(_load("full", "feature_manifest"))
    assert e_cols == f_cols


def test_normalized_features_finite():
    """Frozen-normalizer-applied continuous features are finite (no inf/NaN)."""
    import numpy as np

    manifest = _load("eicu", "feature_manifest")
    flat = [c for cols in manifest.values() for c in cols]
    df = pd.read_parquet(hourly_path(EICU), columns=flat)
    bad = {c: int((~np.isfinite(df[c].to_numpy(dtype="float64"))).sum())
           for c in flat if not np.isfinite(df[c].to_numpy(dtype="float64")).all()}
    assert not bad, f"non-finite normalized features: {bad}"
