"""eICU label pipeline (Phase 9): hourly SOFA -> suspicion -> Sepsis-3 -> split.

Mirrors labels/pipeline.py but uses the eICU SOFA + suspicion builders, then
REUSES labels/sepsis3.py:build_sepsis3 and labels/splits.py:build_splits
unchanged (mode-keyed parquet -> parquet) so the Sepsis-3 onset logic (dynamic
baseline, acute SOFA rise >= threshold within the suspicion window) and the
three-way analysis split are byte-for-byte the MIMIC logic. SOFA is the
expensive stage; skipped if sofa_eicu.parquet exists unless rebuild_sofa.
"""
from __future__ import annotations

import time

from ..config import CohortConfig, LabelConfig
from ..logging_utils import get_logger
from . import eicu_sofa, eicu_suspicion, sepsis3, splits

log = get_logger("labels.eicu_pipeline")


def run(cfg: CohortConfig | None = None, lcfg: LabelConfig | None = None,
        rebuild_sofa: bool = False) -> None:
    cfg = cfg or CohortConfig(mode="eicu")
    lcfg = lcfg or LabelConfig.load()
    t0 = time.perf_counter()

    if rebuild_sofa or not eicu_sofa.sofa_path(cfg).exists():
        eicu_sofa.run(cfg, lcfg)
    else:
        log.info("eICU SOFA parquet present (%s) — skipping rebuild",
                 eicu_sofa.sofa_path(cfg).name)

    eicu_suspicion.run(cfg, lcfg)
    sepsis3.run(cfg, lcfg)   # REUSED: dynamic-baseline acute SOFA rise onset
    splits.run(cfg, lcfg)    # REUSED: three-way analysis split (Gating A)
    log.info("eICU label pipeline done in %.1fs (mode=%s)",
             time.perf_counter() - t0, cfg.mode)
