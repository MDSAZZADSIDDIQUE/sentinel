"""Sepsis-3 onset = suspicion of infection + acute SOFA rise >= threshold.

For each stay with suspicion of infection at hour ``si_hour``, we search the
window [si_hour - lookback, si_hour + lookahead] for the earliest hour at which
total SOFA reaches the threshold (acute rise from an assumed baseline of 0, per
the mimic-code Sepsis-3 convention). That hour is the **deterioration onset**
the early-warning system must anticipate.

Output: data/labels/sepsis3_<mode>.parquet — one row per cohort stay with
has_suspicion, si_hour, sepsis3, onset_hour, sofa_at_onset, sofa_max.
"""
from __future__ import annotations

import time

from ..config import CohortConfig, LabelConfig
from ..duck import connect
from ..logging_utils import get_logger
from ..paths import PATHS
from .sofa import sofa_path
from .suspicion import suspicion_path

log = get_logger("labels.sepsis3")


def sepsis3_path(cfg: CohortConfig):
    return PATHS.labels_root / f"sepsis3_{cfg.mode}.parquet"


def build_sepsis3(cfg: CohortConfig, lcfg: LabelConfig) -> "object":
    sofa_p = sofa_path(cfg).as_posix()
    susp_p = suspicion_path(cfg).as_posix()
    thr = lcfg.sofa_increase_threshold

    # baseline per suspected stay: dynamic (min SOFA over the pre-suspicion
    # admission window) or a constant (strict mimic-code). Sepsis requires an
    # acute rise: sofa_total - baseline_sofa >= threshold.
    if lcfg.dynamic_baseline:
        base_cte = f"""
        base AS (
            SELECT s.stay_id,
                   COALESCE(min(f.sofa_total), {lcfg.sofa_baseline}) AS baseline_sofa
            FROM susp s LEFT JOIN sofa f ON f.stay_id = s.stay_id
                 AND f.hr BETWEEN 0 AND least(greatest(0, s.si_hour),
                                              {lcfg.baseline_window_hours})
            GROUP BY s.stay_id
        )"""
    else:
        base_cte = f"""
        base AS (SELECT stay_id, {lcfg.sofa_baseline} AS baseline_sofa FROM susp)"""

    con = connect()
    try:
        q = f"""
        WITH susp AS (
            SELECT stay_id, hadm_id, si_hour FROM read_parquet('{susp_p}')
            WHERE has_suspicion = 1
        ),
        sofa AS (SELECT stay_id, hr, sofa_total FROM read_parquet('{sofa_p}')),
        {base_cte},
        onset AS (
            SELECT s.stay_id, min(f.hr) AS onset_hour
            FROM susp s
            JOIN base b ON b.stay_id = s.stay_id
            JOIN sofa f ON f.stay_id = s.stay_id
            WHERE f.sofa_total - b.baseline_sofa >= {thr}
              AND f.hr BETWEEN s.si_hour - {lcfg.si_sofa_lookback_hours}
                          AND s.si_hour + {lcfg.si_sofa_lookahead_hours}
            GROUP BY s.stay_id
        ),
        sofa_stay AS (SELECT stay_id, max(sofa_total) AS sofa_max FROM sofa GROUP BY stay_id),
        susp_all AS (SELECT stay_id, hadm_id, si_hour, has_suspicion
                     FROM read_parquet('{susp_p}'))
        SELECT a.stay_id, a.hadm_id, a.has_suspicion, a.si_hour,
               CASE WHEN o.onset_hour IS NOT NULL THEN 1 ELSE 0 END AS sepsis3,
               o.onset_hour,
               fo.sofa_total AS sofa_at_onset,
               b.baseline_sofa,
               ss.sofa_max
        FROM susp_all a
        LEFT JOIN base b    ON b.stay_id = a.stay_id
        LEFT JOIN onset o   ON o.stay_id = a.stay_id
        LEFT JOIN sofa fo   ON fo.stay_id = a.stay_id AND fo.hr = o.onset_hour
        LEFT JOIN sofa_stay ss ON ss.stay_id = a.stay_id
        """
        df = con.execute(q).df()
    finally:
        con.close()

    PATHS.labels_root.mkdir(parents=True, exist_ok=True)
    out = sepsis3_path(cfg)
    df.to_parquet(out, index=False)

    n = len(df)
    n_sep = int(df["sepsis3"].sum())
    onset = df.loc[df["sepsis3"] == 1, "onset_hour"]
    present_on_adm = int((onset <= 0).sum())
    log.info("  Sepsis-3: %d/%d stays septic (%.1f%%)", n_sep, n, 100 * n_sep / max(n, 1))
    if n_sep:
        log.info("  onset hour: median=%.0f  IQR=[%.0f, %.0f]  present-on-admission(<=0h)=%d (%.1f%%)",
                 onset.median(), onset.quantile(.25), onset.quantile(.75),
                 present_on_adm, 100 * present_on_adm / n_sep)
    log.info("  wrote %s", out)
    return out


def run(cfg: CohortConfig | None = None, lcfg: LabelConfig | None = None) -> None:
    cfg = cfg or CohortConfig.load()
    lcfg = lcfg or LabelConfig.load()
    t0 = time.perf_counter()
    build_sepsis3(cfg, lcfg)
    log.info("sepsis3 done in %.1fs", time.perf_counter() - t0)
