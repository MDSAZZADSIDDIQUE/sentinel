"""DuckDB connection factory tuned for the target machine.

Hard memory/thread limits (HARDWARE ADDENDUM) so DuckDB *spills to disk* rather
than OOM-ing the 13.9 GB box when scanning the multi-GB gzipped CSVs in place.
We never decompress CSVs to disk and never read the big tables into pandas.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb

from .paths import PATHS


@dataclass(frozen=True)
class DuckSettings:
    memory_limit: str = "8GB"
    threads: int = 6
    preserve_insertion_order: bool = False  # lower memory for big scans


def connect(settings: DuckSettings | None = None) -> duckdb.DuckDBPyConnection:
    """Return a configured in-memory DuckDB connection.

    Spills to ``outputs/duckdb_tmp`` so large group-bys/sorts survive the RAM
    ceiling. The connection is in-memory (metadata only); table *data* is read
    on demand straight from the ``.csv.gz`` files.
    """
    s = settings or DuckSettings()
    PATHS.duckdb_tmp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{s.memory_limit}';")
    con.execute(f"SET threads={s.threads};")
    con.execute(f"SET temp_directory='{PATHS.duckdb_tmp.as_posix()}';")
    con.execute("SET preserve_insertion_order=%s;" % str(s.preserve_insertion_order).lower())
    con.execute("SET enable_progress_bar=false;")
    return con


def src(path: Path | str, *, sample_size: int = 200_000) -> str:
    """SQL fragment that scans a gzipped CSV in place with robust typing.

    ``all_varchar`` is *not* used; we let DuckDB sniff types from a large sample
    but keep ``ignore_errors`` off so schema surprises surface loudly. The
    explicit ``compression='gzip'`` avoids re-sniffing the codec each scan.
    """
    p = Path(path).as_posix()
    return (
        f"read_csv_auto('{p}', compression='gzip', sample_size={sample_size}, "
        f"parallel=true)"
    )


def mimic(table: str) -> str:
    """SQL fragment scanning a raw MIMIC table by logical name."""
    return src(PATHS.mimic_csv(table))


def count_rows(con: duckdb.DuckDBPyConnection, table: str) -> int:
    """Row count of a raw MIMIC table without materializing it."""
    return con.execute(f"SELECT COUNT(*) FROM {mimic(table)}").fetchone()[0]
