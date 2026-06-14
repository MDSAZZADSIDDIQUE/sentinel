# CLAUDE.md — SENTINEL conventions, data dictionary, decisions log

SENTINEL is a cooperative multi-agent RL system for **early warning of ICU
deterioration (Sepsis-3)**. The patient is decomposed into **organ-system
specialist agents aligned with the SOFA score** (cardiovascular, respiratory,
renal, hepatic, coagulation, neurologic). Each agent sees only its own organ
subsystem (partial observability → Dec-POMDP) and the team chooses an
**escalation action** (`maintain` / `watch` / `escalate-alert`) each hour.
**Action space is alerting/escalation, not treatment dosing** — deliberate, to
keep evaluation valid and avoid offline-RL off-policy-evaluation pitfalls.

This file is the living engineering log. Keep the decisions section updated.

---

## How to run (Windows / this machine)

```powershell
.\run.ps1 setup            # pip install -e . --no-deps
.\run.ps1 verify-data      # Phase 0 row counts
.\run.ps1 phase1           # to-parquet -> itemids -> cohort -> labels -> report
.\run.ps1 test             # pytest
# anything else is forwarded to the `sentinel` Typer CLI:
.\run.ps1 to-parquet --mode dev
sentinel --help
```

`run.bat` is a cmd.exe wrapper. A `Makefile` mirrors the stages for Linux/macOS.

---

## Environment

- **Python 3.13.9, Anaconda base env** (`C:\ProgramData\anaconda3`). We use the
  existing env rather than a fresh venv on purpose — the CUDA build of PyTorch
  (`torch 2.6.0+cu124`) is already installed and reinstalling it in a clean venv
  is a multi-GB download. `pip install -e . --no-deps` registers the package
  without touching the verified dependency set. Versions are pinned in
  `pyproject.toml` for reproducibility.
- **GPU**: NVIDIA GTX 1650, 4 GB VRAM, compute 7.5, **no tensor cores** → FP32
  only, no mixed precision. CUDA available via torch; auto-fallback to CPU.
- **CPU/RAM**: Ryzen 5 3550H (4c/8t), 13.9 GB RAM. DuckDB capped at 8 GB with
  disk spill; never load `chartevents`/`labevents` into pandas.

### `KMP_DUPLICATE_LIB_OK=TRUE` (important)
Anaconda links multiple OpenMP runtimes (MKL + libomp); importing torch after
duckdb/sklearn triggers `OMP: Error #15` and aborts on Windows. The package sets
this env var in `sentinel/__init__.py:apply_runtime_env()` **before** any heavy
import, and `run.ps1`/`run.bat` set it too. If you invoke Python directly, set
it yourself.

---

## Repository layout

```
src/sentinel/
  __init__.py        runtime-env fixes (KMP), version
  paths.py           pathlib path resolution (env-overridable roots)
  duck.py            DuckDB connection factory (8GB / 6 threads / spill)
  config.py          plain-YAML typed config (CohortConfig, LabelConfig)
  logging_utils.py   stage() ctx mgr: wall-clock + peak RAM + peak VRAM -> CSV
  cli.py             Typer CLI, one command per stage
  data/    labels/   features/  env/  agents/  baselines/
  safety/  fairness/ federated/ eval/  viz/
config/   tests/   scripts/   dashboard/
data/        (gitignored) processed parquet, cohort, labels, features
outputs/     (gitignored) reports, figures, checkpoints, logs, duckdb_tmp
research_paper/  IEEEtran template + (later) sentinel_becithcon2026.tex
```

Paths are configurable via env vars (see `paths.py`):
`SENTINEL_DATA_ROOT`, `SENTINEL_OUTPUT_ROOT`, `SENTINEL_MIMIC_ROOT`.

---

## Coding conventions

- **pathlib** for all paths; never hardcode separators. Absolute paths in code
  that touches data come from `PATHS` (`sentinel.paths`).
- Big-table access **only** through DuckDB (`sentinel.duck`), querying the
  `.csv.gz` in place; write only small, cohort/itemid-filtered parquet. Do not
  decompress CSVs to disk.
- Wrap every pipeline stage in `with stage("name"):` so runtime + peak RAM/VRAM
  land in `outputs/logs/stages.csv` (needed for the paper's compute table).
- **No data leakage**: features at decision time `t` use only data with
  timestamp `≤ t`; labels predict onset at `t + H` (default `H=6h`). Splits are
  time-based; one care unit is held out as the external site.
- **Reproducibility**: fixed seeds, config-driven. Report real numbers; ≥5 seeds
  (3 during dev) with mean±std. Never fabricate results or citations.
- DataLoader `num_workers=0` on Windows (or guard multiprocessing with
  `if __name__ == "__main__":`).

---

## ⚠️ Data governance (READ)

- MIMIC-IV is **credentialed data under a PhysioNet DUA**. Never commit raw or
  derived row-level patient data. `.gitignore` excludes `mimic-iv-3.1/`, `data/`,
  `outputs/`, and all `*.parquet`/`*.csv`/`*.csv.gz`.
- **OneDrive caveat**: the repo lives under `…\OneDrive\Documents\GitHub\sentinel`.
  OneDrive may sync the raw MIMIC files and derived parquet to consumer cloud
  storage, which is **in tension with the DUA**. Mitigations: pause OneDrive sync
  for this folder, mark `mimic-iv-3.1/`, `data/`, `outputs/` as "Always keep on
  this device" but exclude from upload, or redirect heavy artifacts off OneDrive
  via `SENTINEL_DATA_ROOT` / `SENTINEL_OUTPUT_ROOT`. **Flagged for the user.**

---

## Data dictionary (tables we use)

All raw tables are gzipped CSV under `mimic-iv-3.1/{hosp,icu}/`. Schemas verified
2026-06-14 via DuckDB `DESCRIBE`.

| Table | Grain / key | Columns we use |
|---|---|---|
| `icu/icustays` | one ICU stay (`stay_id`) | subject_id, hadm_id, stay_id, first_careunit, last_careunit, intime, outtime, los (days) |
| `hosp/patients` | one patient (`subject_id`) | gender, anchor_age, anchor_year, anchor_year_group, dod |
| `hosp/admissions` | one admission (`hadm_id`) | admittime, dischtime, deathtime, race, insurance, marital_status, hospital_expire_flag, edregtime |
| `icu/chartevents` | charted obs (huge) | stay_id, charttime, itemid, valuenum, valueuom |
| `hosp/labevents` | lab result (huge) | **hadm_id only (no stay_id)**, charttime, itemid, valuenum, valueuom |
| `icu/inputevents` | infusions | stay_id, starttime, endtime, itemid, rate, rateuom, amount, patientweight |
| `icu/outputevents` | outputs | stay_id, charttime, itemid, value (urine) |
| `hosp/microbiologyevents` | cultures | hadm_id, chartdate, charttime, spec_type_desc |
| `hosp/prescriptions` | meds | hadm_id, starttime, stoptime, drug, route |
| `icu/d_items` | dict (chart/input/output) | itemid, label, abbreviation, linksto, category, unitname |
| `hosp/d_labitems` | dict (labs) | itemid, label, fluid, category |

### Notable facts / gotchas
- **Age** = `anchor_age + (year(admittime) − anchor_year)`. MIMIC caps ages >89,
  reporting `anchor_age=91`; we exclude that band (`max_age=89`).
- **labevents has no `stay_id`**: attribute labs to a stay via `hadm_id` and
  `charttime ∈ [intime, outtime]`.
- **Vasopressors** for cardiovascular SOFA come from `inputevents.rate`
  (norepinephrine-equivalent µg/kg/min); use `patientweight` when rate is in
  µg/min. Resolve dopamine/epinephrine/norepinephrine/dobutamine itemids.
- **GCS** components are in `chartevents` (GCS-Eye/Verbal/Motor or GCS-Total).
- **PaO2/FiO2** for respiratory SOFA: PaO2 from `labevents` (blood gas), FiO2
  from `chartevents`; ventilation status modifies the SOFA cut-offs.
- Sepsis-3 onset uses suspicion-of-infection time (`microbiologyevents` +
  antibiotic `prescriptions`) combined with an acute SOFA rise ≥2.

---

## Key decisions log

- **2026-06-14** Use existing Anaconda env + `--no-deps` editable install to
  preserve the CUDA torch build. Pin versions in `pyproject.toml`.
- **2026-06-14** `KMP_DUPLICATE_LIB_OK=TRUE` set pre-import to fix OMP Error #15.
- **2026-06-14** Plain YAML configs instead of Hydra (Windows path robustness,
  smaller reproducibility surface).
- **2026-06-14** DuckDB for all raw access: `memory_limit=8GB`, `threads=6`,
  `temp_directory=outputs/duckdb_tmp` (spill, not OOM). Convert once to parquet,
  then work from parquet.
- **2026-06-14** Self-contained MAPPO/QMIX in-repo (no epymarl/pymarl2/SMAC) per
  hardware addendum (Windows spawn issues, reproducibility).
- **2026-06-14** Two-tier cohort (`dev` ~4k stays from MICU+SICU / `full`); cache
  features to parquet so training never recomputes from raw tables.
- **2026-06-14** Federated "sites" = `first_careunit`; external test site =
  CVICU (held out). Sites are simulated via care units, not real institutions —
  stated as a limitation in the paper.

- **2026-06-14 (post-checkpoint)** Per user review, tightened Sepsis-3 toward
  literature prevalence: (a) **dynamic baseline** — sepsis = acute SOFA rise ≥ 2
  over the pre-suspicion admission baseline (min SOFA in [0, min(si_hour, 24)]),
  the literal Sepsis-3 "acute change"; ~37% prevalence vs ~44% for strict
  mimic-code baseline-0 (kept available via `dynamic_baseline: false`).
  (b) **bounded ffill** (labs 48 h, vitals 24 h), **ventilation-aware GCS**
  (intubated verbal=1T not penalized), **physiologic range filtering**. CNS
  de-inflated (mean 1.30→0.84). OneDrive/DUA: user manages sync themselves.

## Open assumptions to revisit
- Default prediction horizon `H = 6h`; SOFA worst-value rolling window 24h.
- `min_los_hours = 6`, adults 18–89, first ICU stay per admission only.
- Dynamic baseline window 24h; worst-in-hour aggregation (min MAP/GCS, max
  bilirubin/creatinine). Respiratory PF needs both PaO2 (~66%) and FiO2 (~50%)
  → consider SpO2/FiO2 fallback in Phase 2.
