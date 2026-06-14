"""Label pipeline orchestrator: hourly SOFA -> suspicion -> Sepsis-3 onset.

SOFA is the expensive stage; it is skipped if its parquet already exists unless
``rebuild_sofa`` is set.
"""
from __future__ import annotations

import time

from ..config import CohortConfig, LabelConfig
from ..logging_utils import get_logger
from . import sepsis3, sofa, suspicion

log = get_logger("labels.pipeline")


def run(cfg: CohortConfig | None = None, lcfg: LabelConfig | None = None,
        rebuild_sofa: bool = False) -> None:
    cfg = cfg or CohortConfig.load()
    lcfg = lcfg or LabelConfig.load()
    t0 = time.perf_counter()

    if rebuild_sofa or not sofa.sofa_path(cfg).exists():
        sofa.run(cfg, lcfg)
    else:
        log.info("SOFA parquet present (%s) — skipping rebuild", sofa.sofa_path(cfg).name)

    suspicion.run(cfg, lcfg)
    sepsis3.run(cfg, lcfg)
    log.info("label pipeline done in %.1fs (mode=%s)", time.perf_counter() - t0, cfg.mode)
