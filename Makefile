# SENTINEL pipeline. On Windows prefer .\run.ps1 (no make needed); this Makefile
# mirrors the same stages for Linux/macOS or Windows-with-make reproducibility.
# `make all` (after `make to-parquet`) reproduces cohort -> labels -> ... -> paper.

export KMP_DUPLICATE_LIB_OK := TRUE
export PYTHONUTF8 := 1
PY := python

.PHONY: setup verify-data to-parquet itemids cohort labels report \
        features env-test train-baselines train-marl eval figures \
        dashboard paper test clean phase1 all

setup:            ## install package (editable, no deps — protects CUDA torch)
	$(PY) -m pip install -e . --no-deps

verify-data:      ## Phase 0: data path + row counts
	sentinel verify-data

to-parquet:       ## Phase 1: convert/filter MIMIC tables to parquet
	sentinel to-parquet

itemids:          ## Phase 1: resolve itemids -> config/itemids.yaml
	sentinel resolve-itemids

cohort:           ## Phase 1: build ICU cohort
	sentinel build-cohort

labels:           ## Phase 1: SOFA + suspicion + Sepsis-3 onset
	sentinel build-labels

report:           ## Phase 1: cohort report
	sentinel cohort-report

phase1: itemids cohort to-parquet labels report   ## full Phase 1 (dependency order)

features:         ## Phase 2
	sentinel build-features

env-test:         ## Phase 2: env smoke test
	$(PY) -m pytest tests/test_env.py

train-baselines:  ## Phase 3
	sentinel train-baselines

train-marl:       ## Phase 4
	sentinel train-marl

eval:             ## Phase 6
	sentinel evaluate

figures:          ## Phase 6
	sentinel figures

dashboard:        ## Phase 7
	streamlit run dashboard/app.py

paper:            ## Phase 8: compile the IEEE paper
	$(PY) scripts/build_paper.py

test:             ## run unit tests
	$(PY) -m pytest

clean:            ## remove duckdb temp + pyc
	rm -rf outputs/duckdb_tmp __pycache__ .pytest_cache

all: cohort labels features train-baselines train-marl eval figures paper
