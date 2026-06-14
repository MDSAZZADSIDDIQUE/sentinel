"""Quantify Sepsis-3 prevalence under alternative, defensible operationalizations,
to choose how to bring the headline number in line with the literature.
"""
from __future__ import annotations

import duckdb

from sentinel.config import CohortConfig
from sentinel.data.cohort import cohort_path
from sentinel.labels.sofa import sofa_path
from sentinel.labels.suspicion import suspicion_path

cfg = CohortConfig(mode="full")
sofa = sofa_path(cfg).as_posix()
susp = suspicion_path(cfg).as_posix()
coh = cohort_path(cfg).as_posix()
con = duckdb.connect()
con.execute("SET memory_limit='8GB'; SET threads=6;")

N = con.execute(f"SELECT COUNT(*) FROM read_parquet('{coh}')").fetchone()[0]


def pct(n):
    return f"{n:,} ({100*n/N:.1f}%)"


# (a) current: baseline 0, any onset in window
a = con.execute(f"""
WITH susp AS (SELECT stay_id, si_hour FROM read_parquet('{susp}') WHERE has_suspicion=1),
sofa AS (SELECT stay_id, hr, sofa_total FROM read_parquet('{sofa}'))
SELECT COUNT(DISTINCT s.stay_id) FROM susp s JOIN sofa f ON f.stay_id=s.stay_id
WHERE f.sofa_total>=2 AND f.hr BETWEEN s.si_hour-48 AND s.si_hour+24
""").fetchone()[0]

# (b) baseline 0 but onset strictly after admission (onset_hour >= 1)
b = con.execute(f"""
WITH susp AS (SELECT stay_id, si_hour FROM read_parquet('{susp}') WHERE has_suspicion=1),
sofa AS (SELECT stay_id, hr, sofa_total FROM read_parquet('{sofa}')),
o AS (SELECT s.stay_id, MIN(f.hr) onset FROM susp s JOIN sofa f ON f.stay_id=s.stay_id
      WHERE f.sofa_total>=2 AND f.hr BETWEEN s.si_hour-48 AND s.si_hour+24 GROUP BY 1)
SELECT COUNT(*) FROM o WHERE onset >= 1
""").fetchone()[0]

# (c) acute rise >=2 vs dynamic baseline = min SOFA over [0, si_hour]
c = con.execute(f"""
WITH susp AS (SELECT stay_id, si_hour FROM read_parquet('{susp}') WHERE has_suspicion=1),
sofa AS (SELECT stay_id, hr, sofa_total FROM read_parquet('{sofa}')),
base AS (SELECT s.stay_id, s.si_hour, COALESCE(MIN(f.sofa_total),0) base
         FROM susp s LEFT JOIN sofa f ON f.stay_id=s.stay_id
              AND f.hr BETWEEN 0 AND greatest(0,s.si_hour) GROUP BY 1,2)
SELECT COUNT(DISTINCT b.stay_id) FROM base b JOIN sofa f ON f.stay_id=b.stay_id
WHERE f.sofa_total - b.base >= 2 AND f.hr BETWEEN b.si_hour-48 AND b.si_hour+24
""").fetchone()[0]

# (d) acute rise vs baseline = SOFA at admission (hr 0)
d = con.execute(f"""
WITH susp AS (SELECT stay_id, si_hour FROM read_parquet('{susp}') WHERE has_suspicion=1),
sofa AS (SELECT stay_id, hr, sofa_total FROM read_parquet('{sofa}')),
base AS (SELECT s.stay_id, s.si_hour, COALESCE(MAX(CASE WHEN f.hr=0 THEN f.sofa_total END),0) base
         FROM susp s LEFT JOIN sofa f ON f.stay_id=s.stay_id GROUP BY 1,2)
SELECT COUNT(DISTINCT b.stay_id) FROM base b JOIN sofa f ON f.stay_id=b.stay_id
WHERE f.sofa_total - b.base >= 2 AND f.hr BETWEEN b.si_hour-48 AND b.si_hour+24
""").fetchone()[0]

print(f"cohort N = {N:,}\n")
print(f"(a) baseline 0, any onset (CURRENT, mimic-code) : {pct(a)}")
print(f"(b) baseline 0, onset >= 1h (excl present-on-adm): {pct(b)}")
print(f"(c) acute rise>=2 vs min SOFA[0..si]            : {pct(c)}")
print(f"(d) acute rise>=2 vs SOFA at admission (hr0)     : {pct(d)}")

# component breakdown after tightening
print("\n=== SOFA components after tightening ===")
for cc in ["sofa_respiration","sofa_coagulation","sofa_liver","sofa_cardiovascular","sofa_cns","sofa_renal"]:
    m, fr = con.execute(f"SELECT avg({cc}), avg(CASE WHEN {cc}>0 THEN 1.0 ELSE 0 END) FROM read_parquet('{sofa}')").fetchone()
    print(f"  {cc:<22} mean={m:.2f}  %hours>0={100*fr:.1f}")
