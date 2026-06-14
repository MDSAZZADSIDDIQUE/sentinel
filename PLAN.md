# PLAN.md — SENTINEL build plan & status

Status legend: ✅ done · 🟡 in progress · ⬜ not started · ⏸ awaiting review

> Build a thin working vertical slice first, then scale. Commit after each phase.
> **Hard stop for human review after Phase 1.**

## Phase 0 — Scaffold ✅
- [x] Package tree `src/sentinel/...`, `pyproject.toml` (pinned), `.gitignore` (data governance)
- [x] Core utils: `paths.py`, `duck.py` (8GB/6thr/spill), `config.py` (YAML), `logging_utils.py` (stage RAM/VRAM/wall-clock)
- [x] Typer CLI `sentinel`, `run.ps1`/`run.bat`, `Makefile`
- [x] `CLAUDE.md`, `PLAN.md`, `README.md`
- [x] Verify data path + DuckDB row counts (no in-RAM load) → `outputs/logs/verify_data.log`

## Phase 1 — Data → cohort → labels ⏸ (CHECKPOINT)
- [ ] `to-parquet`: convert needed tables; big tables filtered to cohort `stay_id`/`hadm_id` + curated itemids
- [ ] `resolve-itemids`: query `d_items`/`d_labitems` by label text → `config/itemids.yaml` (no unverified hardcoding; cross-check mimic-code)
- [ ] `build-cohort`: adults 18–89, LOS ≥ 6h, first ICU stay/admission; `dev`/`full`; record `first_careunit` site → `config/cohort.yaml`
- [ ] `build-labels`: hourly SOFA (6 components), suspicion-of-infection (abx+culture window), Sepsis-3 onset (acute SOFA rise ≥2)
- [ ] Unit tests: SOFA component scoring + Sepsis-3 onset on hand-verified stays; leakage guard
- [ ] `cohort-report`: size, sepsis prevalence, onset-time dist, missingness, site partition → `outputs/reports/cohort_report.md`
- [ ] **PAUSE — summarize for review**

## Phase 2 — Features + environment ⬜
- [ ] Hourly feature tensors per stay; explicit missingness indicators; clinically-bounded ffill; normalize on train stats only
- [ ] Dec-POMDP env (Gymnasium/PettingZoo-style): per-agent organ observations + minimal shared context; joint escalation action; episode = one stay streamed hourly
- [ ] Reward = early-warning utility (ramped pre-onset reward, big miss penalty, small false-alarm + per-escalate cost); configurable; unit-tested on synthetic episodes

## Phase 3 — Baselines ⬜
- [ ] NEWS/MEWS (rule-based), XGBoost (windowed features), GRU/Transformer single-agent

## Phase 4 — MARL core ⬜
- [ ] Organ-system agents; MAPPO (centralized critic, decentralized actors); QMIX/VDN ablation; train on internal sites

## Phase 5 — Safety / fairness / federated ⬜
- [ ] Safety: min sensitivity ≥0.85 via Lagrangian + decision shield; safety vs false-alarm trade-off
- [ ] Fairness: parity across sex/age/race; optional reweighting; before/after
- [ ] Federated: partition by `first_careunit`; FedAvg (Flower); CVICU external test site; centralized vs federated gap

## Phase 6 — Evaluation + figures ⬜
- [ ] Full matrix (≥5 seeds, bootstrap CIs); subgroup + external-site; AUROC/AUPRC/sens/spec/lead-time/false-alarm/utility/calibration
- [ ] All paper figures generated programmatically → `research_paper/figures/`

## Phase 7 — Dashboard ⬜
- [ ] Streamlit replay of a stay: live risk, per-organ-agent contributions, escalation timeline, onset marker

## Phase 8 — Paper ⬜
- [ ] `research_paper/sentinel_becithcon2026.tex` (IEEEtran), real figures, verified `references.bib`; build via MiKTeX or script (Overleaf fallback — no local pdflatex yet)

---

## Experiment matrix (Phase 6 target)
Models × {internal test, external site}: NEWS/MEWS, XGBoost, GRU/Transformer, SENTINEL-MAPPO, SENTINEL-QMIX.
Ablations: (1) multi vs single agent @ matched alert budget; (2) missing-signal robustness; (3) safety on/off; (4) centralized vs federated; (5) fairness pre/post reweighting.
