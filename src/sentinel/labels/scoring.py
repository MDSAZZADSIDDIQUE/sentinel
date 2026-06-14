"""Pure SOFA component scoring (Vincent 1996; Singer 2016 / Sepsis-3).

Vectorized numpy implementations are the single source of truth; scalar
``*_one`` wrappers exist for readability and unit tests. Each component returns
an integer 0–4. Missing inputs (NaN) score 0 for that component (conservative;
missingness is tracked separately in the feature pipeline).

Dose units: vasopressor rates in µg/kg/min (norepinephrine, epinephrine,
dopamine, dobutamine); dopamine thresholds are the classic 5/15 cut-offs.
Urine output in mL/day. Bilirubin/creatinine in mg/dL. Platelets in ×10³/µL.
PaO2/FiO2 in mmHg (FiO2 as a fraction in the ratio).
"""
from __future__ import annotations

import numpy as np

ArrayLike = np.ndarray


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype="float64")


def score_respiration(pf_ratio, ventilated) -> np.ndarray:
    """PaO2/FiO2 ratio (mmHg). Scores 3–4 require respiratory support."""
    pf = _arr(pf_ratio)
    vent = np.asarray(ventilated, dtype=bool)
    base = np.select(
        [pf < 100, pf < 200, pf < 300, pf < 400],
        [4, 3, 2, 1],
        default=0,
    ).astype(int)
    # Without mechanical ventilation, cap the contribution at 2.
    base = np.where((~vent) & (base > 2), 2, base)
    return base


def score_coagulation(platelets) -> np.ndarray:
    plt = _arr(platelets)
    return np.select([plt < 20, plt < 50, plt < 100, plt < 150], [4, 3, 2, 1], default=0).astype(int)


def score_liver(bilirubin) -> np.ndarray:
    b = _arr(bilirubin)
    return np.select([b >= 12.0, b >= 6.0, b >= 2.0, b >= 1.2], [4, 3, 2, 1], default=0).astype(int)


def score_cardiovascular(map_mmhg, dopamine=np.nan, dobutamine=np.nan,
                         epinephrine=np.nan, norepinephrine=np.nan) -> np.ndarray:
    """MAP + vasopressor doses (µg/kg/min). dobutamine at any dose => 2."""
    mp = _arr(map_mmhg)
    dop = _arr(dopamine)
    dob = _arr(dobutamine)
    epi = _arr(epinephrine)
    nor = _arr(norepinephrine)
    sc4 = (dop > 15) | (epi > 0.1) | (nor > 0.1)
    sc3 = (dop > 5) | ((epi > 0) & (epi <= 0.1)) | ((nor > 0) & (nor <= 0.1))
    sc2 = ((dop > 0) & (dop <= 5)) | (dob > 0)
    sc1 = mp < 70
    return np.select([sc4, sc3, sc2, sc1], [4, 3, 2, 1], default=0).astype(int)


def score_cns(gcs) -> np.ndarray:
    g = _arr(gcs)
    # GCS NaN -> 0 (assume no neuro dysfunction if unmeasured).
    return np.select([g < 6, g < 10, g < 13, g < 15], [4, 3, 2, 1], default=0).astype(int)


def score_renal(creatinine, urine_ml_day=np.nan) -> np.ndarray:
    """Worst of creatinine- and urine-output-based subscores."""
    cr = _arr(creatinine)
    uo = _arr(urine_ml_day)
    cr_sc = np.select([cr >= 5.0, cr >= 3.5, cr >= 2.0, cr >= 1.2], [4, 3, 2, 1], default=0)
    uo_sc = np.select([uo < 200, uo < 500], [4, 3], default=0)
    return np.maximum(cr_sc, uo_sc).astype(int)


# --- scalar wrappers (for tests / readability) ----------------------------
def _one(fn, *args) -> int:
    return int(fn(*[np.array([a]) for a in args])[0])


def score_respiration_one(pf_ratio: float, ventilated: bool) -> int:
    return _one(score_respiration, pf_ratio, ventilated)


def score_coagulation_one(platelets: float) -> int:
    return _one(score_coagulation, platelets)


def score_liver_one(bilirubin: float) -> int:
    return _one(score_liver, bilirubin)


def score_cardiovascular_one(map_mmhg, dopamine=np.nan, dobutamine=np.nan,
                             epinephrine=np.nan, norepinephrine=np.nan) -> int:
    return _one(score_cardiovascular, map_mmhg, dopamine, dobutamine, epinephrine, norepinephrine)


def score_cns_one(gcs: float) -> int:
    return _one(score_cns, gcs)


def score_renal_one(creatinine: float, urine_ml_day: float = np.nan) -> int:
    return _one(score_renal, creatinine, urine_ml_day)


# component name -> organ system (for per-organ SOFA in the dashboard/agents)
COMPONENT_ORGAN = {
    "sofa_respiration": "respiratory",
    "sofa_coagulation": "coagulation",
    "sofa_liver": "hepatic",
    "sofa_cardiovascular": "cardiovascular",
    "sofa_cns": "neurologic",
    "sofa_renal": "renal",
}
SOFA_COMPONENTS = tuple(COMPONENT_ORGAN.keys())
