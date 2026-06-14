"""Baseline sensitivity analysis: dynamic admission baseline vs strict mimic-code
baseline-0, reported side by side.

The dynamic baseline (acute SOFA rise over the pre-suspicion admission state) is
our primary definition; baseline-0 (Seymour et al. / mimic-code, SOFA>=2 with
assumed baseline 0) is the canonical comparator. Reporting both pre-empts the
"that's cohort engineering" critique — the reader sees exactly what each choice
buys and that 11% ICU-acquired is consistent with hospital-onset-sepsis figures.
"""
from __future__ import annotations

import dataclasses
import time

import pandas as pd

from ..config import CohortConfig, LabelConfig
from ..logging_utils import get_logger
from .sepsis3 import build_sepsis3, sepsis3_path
from .splits import CONTROL, EXCLUDED, POSITIVE, assign_groups

log = get_logger("labels.sensitivity")

BASELINE0_TAG = "_baseline0"


def build_baseline0(cfg: CohortConfig, lcfg: LabelConfig) -> None:
    """Build the strict baseline-0 Sepsis-3 variant (kept beside the primary)."""
    lcfg0 = dataclasses.replace(lcfg, dynamic_baseline=False)
    build_sepsis3(cfg, lcfg0, tag=BASELINE0_TAG)


def _metrics(sep: pd.DataFrame, H: int, n_cohort: int) -> dict:
    septic = sep[sep["sepsis3"] == 1]
    onset = septic["onset_hour"]
    groups = assign_groups(sep, H).value_counts()
    return {
        "Sepsis-3 prevalence": f"{len(septic):,} ({100*len(septic)/n_cohort:.1f}%)",
        "Present-on-admission (<=0h)": f"{int((onset <= 0).sum()):,} "
                                       f"({100*(onset<=0).mean():.1f}% of septic)",
        "Onset hour median [IQR]": (f"{onset.median():.0f} "
                                    f"[{onset.quantile(.25):.0f}-{onset.quantile(.75):.0f}]"
                                    if len(septic) else "n/a"),
        f"POSITIVE (onset>={H}h)": f"{int(groups.get(POSITIVE,0)):,} "
                                   f"({100*int(groups.get(POSITIVE,0))/n_cohort:.1f}%)",
        "EXCLUDED (prevalent/early)": f"{int(groups.get(EXCLUDED,0)):,} "
                                      f"({100*int(groups.get(EXCLUDED,0))/n_cohort:.1f}%)",
        "CONTROL (never Sepsis-3)": f"{int(groups.get(CONTROL,0)):,} "
                                    f"({100*int(groups.get(CONTROL,0))/n_cohort:.1f}%)",
    }


def compare(cfg: CohortConfig, lcfg: LabelConfig) -> pd.DataFrame:
    """Return a metric x {dynamic, baseline0} comparison table."""
    from ..data.cohort import load_cohort

    n = len(load_cohort(cfg))
    H = lcfg.horizon_hours
    primary = pd.read_parquet(sepsis3_path(cfg))
    base0_path = sepsis3_path(cfg, BASELINE0_TAG)
    if not base0_path.exists():
        build_baseline0(cfg, lcfg)
    base0 = pd.read_parquet(base0_path)

    md = _metrics(primary, H, n)
    mb = _metrics(base0, H, n)
    return pd.DataFrame({
        "metric": list(md.keys()),
        "Dynamic baseline (primary)": list(md.values()),
        "Baseline-0 (mimic-code)": [mb[k] for k in md],
    })


def run(cfg: CohortConfig | None = None, lcfg: LabelConfig | None = None) -> None:
    cfg = cfg or CohortConfig.load()
    lcfg = lcfg or LabelConfig.load()
    t0 = time.perf_counter()
    build_baseline0(cfg, lcfg)
    tbl = compare(cfg, lcfg)
    for _, r in tbl.iterrows():
        log.info("  %-30s | dyn=%-22s | base0=%s",
                 r["metric"], r["Dynamic baseline (primary)"], r["Baseline-0 (mimic-code)"])
    log.info("sensitivity done in %.1fs", time.perf_counter() - t0)
