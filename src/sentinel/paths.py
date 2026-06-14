"""Centralised, OS-agnostic path resolution (pathlib everywhere).

All paths derive from the repository root, which is located relative to this
file (``src/sentinel/paths.py`` -> repo root is ``parents[2]``). Data and output
roots can be redirected via environment variables so heavy artifacts can live
*outside* the OneDrive-synced tree if desired (see CLAUDE.md, governance note):

    SENTINEL_DATA_ROOT     -> processed parquet + derived tables  (default: ./data)
    SENTINEL_OUTPUT_ROOT   -> figures, reports, checkpoints, logs  (default: ./outputs)
    SENTINEL_MIMIC_ROOT    -> raw MIMIC-IV v3.1 root               (default: ./mimic-iv-3.1)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_path(var: str, default: Path) -> Path:
    val = os.environ.get(var)
    return Path(val).expanduser().resolve() if val else default


@dataclass(frozen=True)
class Paths:
    repo_root: Path = _REPO_ROOT
    mimic_root: Path = field(default_factory=lambda: _env_path("SENTINEL_MIMIC_ROOT", _REPO_ROOT / "mimic-iv-3.1"))
    data_root: Path = field(default_factory=lambda: _env_path("SENTINEL_DATA_ROOT", _REPO_ROOT / "data"))
    output_root: Path = field(default_factory=lambda: _env_path("SENTINEL_OUTPUT_ROOT", _REPO_ROOT / "outputs"))
    config_root: Path = _REPO_ROOT / "config"
    paper_root: Path = _REPO_ROOT / "research_paper"

    # --- MIMIC raw table groups -------------------------------------------
    @property
    def hosp(self) -> Path:
        return self.mimic_root / "hosp"

    @property
    def icu(self) -> Path:
        return self.mimic_root / "icu"

    # --- derived / processed ----------------------------------------------
    @property
    def parquet_root(self) -> Path:
        return self.data_root / "parquet"

    @property
    def cohort_root(self) -> Path:
        return self.data_root / "cohort"

    @property
    def labels_root(self) -> Path:
        return self.data_root / "labels"

    @property
    def features_root(self) -> Path:
        return self.data_root / "features"

    @property
    def duckdb_tmp(self) -> Path:
        return self.output_root / "duckdb_tmp"

    @property
    def logs_root(self) -> Path:
        return self.output_root / "logs"

    @property
    def reports_root(self) -> Path:
        return self.output_root / "reports"

    @property
    def figures_root(self) -> Path:
        return self.paper_root / "figures"

    # --- raw MIMIC csv.gz locations ---------------------------------------
    # Maps logical table name -> (group, filename). Only tables we touch.
    _MIMIC_TABLES = {
        "icustays": ("icu", "icustays.csv.gz"),
        "d_items": ("icu", "d_items.csv.gz"),
        "chartevents": ("icu", "chartevents.csv.gz"),
        "inputevents": ("icu", "inputevents.csv.gz"),
        "outputevents": ("icu", "outputevents.csv.gz"),
        "patients": ("hosp", "patients.csv.gz"),
        "admissions": ("hosp", "admissions.csv.gz"),
        "d_labitems": ("hosp", "d_labitems.csv.gz"),
        "labevents": ("hosp", "labevents.csv.gz"),
        "microbiologyevents": ("hosp", "microbiologyevents.csv.gz"),
        "prescriptions": ("hosp", "prescriptions.csv.gz"),
    }

    def mimic_csv(self, table: str) -> Path:
        if table not in self._MIMIC_TABLES:
            raise KeyError(f"Unknown MIMIC table '{table}'. Known: {sorted(self._MIMIC_TABLES)}")
        group, fname = self._MIMIC_TABLES[table]
        return self.mimic_root / group / fname

    def parquet(self, table: str) -> Path:
        """Path to a converted/derived parquet (file or directory)."""
        return self.parquet_root / f"{table}.parquet"

    def ensure_dirs(self) -> None:
        for p in (self.data_root, self.parquet_root, self.cohort_root, self.labels_root,
                  self.features_root, self.output_root, self.duckdb_tmp, self.logs_root,
                  self.reports_root):
            p.mkdir(parents=True, exist_ok=True)


PATHS = Paths()
