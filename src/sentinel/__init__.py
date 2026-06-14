"""SENTINEL: cooperative multi-agent RL for early warning of ICU deterioration.

Importing this package eagerly applies the runtime environment fixes needed on
the target Windows / Anaconda machine (notably the duplicate-OpenMP workaround),
*before* any heavy library (torch, sklearn, duckdb) gets imported transitively.
"""
from __future__ import annotations

import os

__version__ = "0.1.0"


def apply_runtime_env() -> None:
    """Set environment variables that must precede heavy-library imports.

    - ``KMP_DUPLICATE_LIB_OK``: Anaconda ships multiple OpenMP runtimes
      (MKL + libomp). torch/duckdb/sklearn each link one, and Windows aborts
      with ``OMP: Error #15`` when a second copy initializes. Allowing the
      duplicate is the documented workaround and is safe for our workloads.
    """
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    # Keep BLAS/OMP thread pools from oversubscribing the 4-core / 8-thread CPU.
    os.environ.setdefault("OMP_NUM_THREADS", "6")


apply_runtime_env()
