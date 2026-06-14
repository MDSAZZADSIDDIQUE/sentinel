"""Gating B (feature/label separation) + external-site lock.

These are structural guards that don't need the MIMIC data: they read the
committed configs (itemids.yaml, cohort.yaml).
"""
from __future__ import annotations

import pytest

from sentinel.constants import SOURCES

itemids = pytest.importorskip("sentinel.data.itemids")
from sentinel.config import CohortConfig  # noqa: E402
from sentinel.data.itemids import variable_meta  # noqa: E402

# label-defining raw tables — must never be a feature source
LABEL_ONLY_TABLES = {"prescriptions", "microbiologyevents"}
BANNED_FEATURE_TOKENS = ("antibiotic", "abx", "culture", "microbio", "suspicion", "sepsis")


def _meta():
    try:
        return variable_meta()
    except FileNotFoundError:
        pytest.skip("itemids.yaml not generated")


def test_features_are_pure_physiology():
    """Every model-input variable comes from a physiology event table."""
    meta = _meta()
    for var, m in meta.items():
        assert m["source"] in set(SOURCES), f"{var}: non-physiology source {m['source']}"


def test_no_infection_label_sources_in_features():
    """Antibiotics / cultures (which DEFINE the label) are not features (Gating B)."""
    meta = _meta()
    feature_sources = {m["source"] for m in meta.values()}
    assert not (feature_sources & LABEL_ONLY_TABLES), (
        "label-defining tables leaked into features")
    for var in meta:
        assert not any(tok in var.lower() for tok in BANNED_FEATURE_TOKENS), (
            f"feature variable '{var}' looks like an infection/label signal")


def test_organ_coverage_complete():
    """All six SOFA organ systems are represented in the feature set."""
    from sentinel.constants import ORGAN_SYSTEMS

    organs = {m["organ"] for m in _meta().values()}
    for o in ORGAN_SYSTEMS:
        assert o in organs, f"no features for organ system '{o}'"


def test_external_site_is_held_out_of_dev():
    """The external validation site must not also be a dev/training care unit."""
    cfg = CohortConfig.load()
    assert cfg.external_site, "external_site must be set"
    assert cfg.external_site not in cfg.dev_careunits, (
        "external validation site leaks into the dev/training care units")
