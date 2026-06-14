"""Phase 0 data verification: confirm the MIMIC path and count rows.

Counting scans the gzipped CSV (DuckDB streams it; nothing is materialized in
RAM), so the two huge tables take a few minutes each. Use ``quick=True`` to skip
them during routine checks.
"""
from __future__ import annotations

import time

from ..duck import connect, count_rows
from ..logging_utils import get_logger
from ..paths import PATHS

log = get_logger("data.verify")

# Tables SENTINEL actually consumes, grouped by approximate scan cost.
SMALL_TABLES = [
    "icustays", "patients", "admissions", "d_items", "d_labitems",
    "outputevents", "microbiologyevents",
]
MEDIUM_TABLES = ["inputevents", "prescriptions"]
HUGE_TABLES = ["chartevents", "labevents"]
ALL_TABLES = SMALL_TABLES + MEDIUM_TABLES + HUGE_TABLES


def check_paths() -> list[tuple[str, bool, float]]:
    """Return (table, exists, size_MB) for every needed raw table."""
    rows = []
    for t in ALL_TABLES:
        p = PATHS.mimic_csv(t)
        exists = p.exists()
        size_mb = p.stat().st_size / 1e6 if exists else 0.0
        rows.append((t, exists, size_mb))
    return rows


def verify(quick: bool = False) -> dict[str, int]:
    """Verify data path and return {table: row_count}.

    Logs sizes and per-table scan timing. With ``quick`` the two huge tables
    (chartevents, labevents) are skipped.
    """
    if not PATHS.mimic_root.exists():
        raise FileNotFoundError(f"MIMIC root not found: {PATHS.mimic_root}")

    log.info("MIMIC root: %s", PATHS.mimic_root)
    missing = []
    for t, exists, size_mb in check_paths():
        flag = "ok " if exists else "MISSING"
        log.info("  [%s] %-20s %8.1f MB", flag, t, size_mb)
        if not exists:
            missing.append(t)
    if missing:
        raise FileNotFoundError(f"Missing required tables: {missing}")

    tables = SMALL_TABLES + MEDIUM_TABLES + ([] if quick else HUGE_TABLES)
    counts: dict[str, int] = {}
    con = connect()
    try:
        for t in tables:
            t0 = time.perf_counter()
            n = count_rows(con, t)
            dt = time.perf_counter() - t0
            counts[t] = n
            log.info("  rows %-20s %14s  (%.1fs)", t, f"{n:,}", dt)
    finally:
        con.close()
    if quick:
        log.info("  (skipped chartevents, labevents — quick mode)")
    return counts
