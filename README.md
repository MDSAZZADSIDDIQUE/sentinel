# SENTINEL

**Organ-system multi-agent reinforcement learning for early warning of ICU
patient deterioration (Sepsis-3).**

SENTINEL decomposes the patient into six **organ-system specialist agents
aligned with the SOFA score** (cardiovascular, respiratory, renal, hepatic,
coagulation, neurologic). Each agent observes only its own subsystem (a
Dec-POMDP) and the team coordinates to choose an **escalation action**
(`maintain` / `watch` / `escalate-alert`) every hour. The claim: a decomposed,
coordinating agent team matches or beats a monolithic deep model on
early-warning utility while being **interpretable** (you can see which organ
drove an alert), **robust to missing signals**, and **trainable without
centralizing raw patient data** (federated across simulated sites).

> Action space is **alerting/escalation, not treatment dosing** — by design.

This is a reproducible research artifact targeting IEEE BECITHCON 2026. See
`PLAN.md` for the phased build and `CLAUDE.md` for conventions/decisions.

## Status
Phase 0 (scaffold) complete; Phase 1 (data → cohort → labels) in progress. Build
pauses for human review at the end of Phase 1.

## Requirements
- Windows 11 (developed on Ryzen 5 3550H, 13.9 GB RAM, GTX 1650 4 GB).
- Python 3.11+ (developed on Anaconda Python 3.13.9).
- **MIMIC-IV v3.1** (credentialed; PhysioNet DUA) placed at `./mimic-iv-3.1/`
  with `hosp/` and `icu/` subfolders of gzipped CSVs. **Not** included; never
  committed (see data governance below).
- PyTorch **CUDA build** for the GPU path (already present here as
  `torch 2.6.0+cu124`). To (re)install: 
  `pip install torch --index-url https://download.pytorch.org/whl/cu124`
  (auto-falls back to CPU if CUDA is unavailable).

## Setup
```powershell
.\run.ps1 setup           # pip install -e . --no-deps  (protects CUDA torch)
sentinel info             # show env + GPU
```
`run.ps1`/`run.bat` set `KMP_DUPLICATE_LIB_OK=TRUE` (Anaconda OpenMP fix). If you
call Python directly, set it yourself.

## Running the pipeline
On Windows use `run.ps1` (no `make` needed); a `Makefile` mirrors it elsewhere.

| Stage | Command | Notes / rough runtime* |
|---|---|---|
| Phase 0 verify | `.\run.ps1 verify-data` | full row counts incl. chartevents/labevents (~minutes, gzip scan) |
| Phase 1 parquet | `.\run.ps1 to-parquet` | filters big tables to cohort + itemids |
| Phase 1 itemids | `.\run.ps1 itemids` | writes `config/itemids.yaml` |
| Phase 1 cohort | `.\run.ps1 cohort` | ICU cohort table |
| Phase 1 labels | `.\run.ps1 labels` | SOFA + suspicion + Sepsis-3 onset |
| Phase 1 report | `.\run.ps1 report` | `outputs/reports/cohort_report.md` |
| all of Phase 1 | `.\run.ps1 phase1` | runs the five above in order |
| tests | `.\run.ps1 test` | pytest |

\*Runtimes are filled in as stages are implemented; see `outputs/logs/stages.csv`
(wall-clock + peak RAM/VRAM logged automatically per stage).

Use `--mode dev` for a fast ~4k-stay subset (default) or `--mode full` for the
complete cohort used in final reported numbers:
```powershell
.\run.ps1 to-parquet --mode full
```

## Data governance ⚠️
MIMIC-IV is credentialed data under a PhysioNet DUA. Raw data, derived
row-level patient data, and any PHI are **never** committed — `.gitignore`
excludes `mimic-iv-3.1/`, `data/`, `outputs/`, and all `*.parquet`/`*.csv(.gz)`.
**Note:** this repo sits under a OneDrive-synced path; pause/exclude OneDrive
sync for the data folders or redirect them via `SENTINEL_DATA_ROOT` /
`SENTINEL_OUTPUT_ROOT` to avoid syncing credentialed data to the cloud. See
`CLAUDE.md`.

## Layout
`src/sentinel/{data,labels,features,env,agents,baselines,safety,fairness,federated,eval,viz}`,
`config/`, `tests/`, `dashboard/`, `research_paper/`. Processed data → `data/`;
reports/figures/logs/checkpoints → `outputs/` (both gitignored).

## License
Code: MIT. Data: governed by the PhysioNet Credentialed Health Data License.
