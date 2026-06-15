"""SENTINEL command-line interface (Typer).

One subcommand per pipeline stage. Heavy/optional modules are imported lazily
inside each command so the CLI loads fast and works even before later phases are
built. Run ``sentinel --help`` for the full list.
"""
from __future__ import annotations

import platform
import sys

import typer

from . import __version__
from .config import CohortConfig, LabelConfig
from .logging_utils import get_logger, stage
from .paths import PATHS

app = typer.Typer(
    name="sentinel",
    add_completion=False,
    no_args_is_help=True,
    help="SENTINEL: organ-system multi-agent RL for ICU deterioration early warning.",
)
log = get_logger("cli")


@app.command()
def info() -> None:
    """Print environment, paths and GPU status."""
    typer.echo(f"SENTINEL v{__version__}")
    typer.echo(f"  python      : {platform.python_version()} ({sys.executable})")
    typer.echo(f"  platform    : {platform.platform()}")
    typer.echo(f"  repo root   : {PATHS.repo_root}")
    typer.echo(f"  mimic root  : {PATHS.mimic_root}  (exists={PATHS.mimic_root.exists()})")
    typer.echo(f"  data root   : {PATHS.data_root}")
    typer.echo(f"  output root : {PATHS.output_root}")
    try:
        import torch

        cuda = torch.cuda.is_available()
        dev = torch.cuda.get_device_name(0) if cuda else "CPU"
        typer.echo(f"  torch       : {torch.__version__} | cuda={cuda} | {dev}")
    except Exception as exc:  # pragma: no cover
        typer.echo(f"  torch       : unavailable ({exc})")


@app.command("verify-data")
def verify_data(
    quick: bool = typer.Option(False, help="Skip the two huge tables (chartevents, labevents)."),
) -> None:
    """Phase 0: verify the MIMIC path and print row counts via DuckDB."""
    from .data.verify import verify

    with stage("verify-data"):
        verify(quick=quick)


@app.command("to-parquet")
def to_parquet(
    mode: str = typer.Option(None, help="Override cohort mode: dev | full."),
) -> None:
    """Phase 1: convert/filter needed MIMIC tables to Parquet."""
    from .data.parquet import run as run_to_parquet

    cohort = CohortConfig.load()
    if mode:
        cohort = CohortConfig(**{**cohort.__dict__, "mode": mode})
    with stage("to-parquet"):
        run_to_parquet(cohort)


@app.command("resolve-itemids")
def resolve_itemids() -> None:
    """Phase 1: resolve itemids from d_items/d_labitems into config/itemids.yaml."""
    from .data.itemids import run as run_resolve

    with stage("resolve-itemids"):
        run_resolve()


@app.command("build-cohort")
def build_cohort(
    mode: str = typer.Option(None, help="Override cohort mode: dev | full."),
) -> None:
    """Phase 1: build the ICU cohort table from icustays/patients/admissions."""
    from .data.cohort import run as run_cohort

    cohort = CohortConfig.load()
    if mode:
        cohort = CohortConfig(**{**cohort.__dict__, "mode": mode})
    with stage("build-cohort"):
        run_cohort(cohort)


@app.command("build-labels")
def build_labels(
    mode: str = typer.Option(None, help="Override cohort mode: dev | full."),
    rebuild_sofa: bool = typer.Option(False, help="Force recompute of hourly SOFA."),
) -> None:
    """Phase 1: derive hourly SOFA, suspicion-of-infection and Sepsis-3 onset."""
    from .labels.pipeline import run as run_labels

    cohort = CohortConfig.load()
    if mode:
        cohort = CohortConfig(**{**cohort.__dict__, "mode": mode})
    with stage("build-labels"):
        run_labels(cohort, LabelConfig.load(), rebuild_sofa=rebuild_sofa)


@app.command("cohort-report")
def cohort_report(
    mode: str = typer.Option(None, help="Override cohort mode: dev | full."),
) -> None:
    """Phase 1: write outputs/reports/cohort_report.md."""
    from .labels.report import run as run_report

    cohort = CohortConfig.load()
    if mode:
        cohort = CohortConfig(**{**cohort.__dict__, "mode": mode})
    with stage("cohort-report"):
        run_report(cohort, LabelConfig.load())


# --- later-phase placeholders (wired as phases land) ----------------------
def _todo(phase: str) -> None:
    typer.echo(f"Not implemented yet — arrives in {phase}.")
    raise typer.Exit(code=2)


@app.command("build-partition")
def build_partition_cmd(
    mode: str = typer.Option(None, help="Override cohort mode: dev | full."),
) -> None:
    """Phase 2: assign train/val/test (temporal) + external-site partition."""
    from .features.partition import run as run_partition

    cohort = CohortConfig.load()
    if mode:
        cohort = CohortConfig(**{**cohort.__dict__, "mode": mode})
    with stage("build-partition"):
        run_partition(cohort, LabelConfig.load())


@app.command("build-features")
def build_features(
    mode: str = typer.Option(None, help="Override cohort mode: dev | full."),
) -> None:
    """Phase 2: hourly per-organ feature tensors (partition + features)."""
    from .config import FeatureConfig
    from .features.build import run as run_features
    from .features.partition import run as run_partition

    cohort = CohortConfig.load()
    if mode:
        cohort = CohortConfig(**{**cohort.__dict__, "mode": mode})
    lcfg, fcfg = LabelConfig.load(), FeatureConfig.load()
    with stage("build-features"):
        run_partition(cohort, lcfg)
        run_features(cohort, lcfg, fcfg)


@app.command("train-baselines")
def train_baselines(
    mode: str = typer.Option(None, help="Override cohort mode: dev | full."),
    seeds: int = typer.Option(3, help="Number of seeds for learned baselines."),
    skip_torch: bool = typer.Option(False, help="Skip GRU + organ-ensemble (NEWS+XGBoost only)."),
) -> None:
    """Phase 3: NEWS/MEWS, XGBoost, GRU + independent organ-ensemble (+ ablation)."""
    from .config import FeatureConfig
    from .eval.run_baselines import run as run_baselines

    cohort = CohortConfig.load()
    if mode:
        cohort = CohortConfig(**{**cohort.__dict__, "mode": mode})
    with stage("train-baselines"):
        run_baselines(cohort, FeatureConfig.load(), seeds=tuple(range(seeds)),
                      skip_torch=skip_torch)


@app.command("train-marl")
def train_marl(
    mode: str = typer.Option(None, help="Override cohort mode: dev | full."),
    seeds: int = typer.Option(3, help="Number of seeds."),
    updates: int = typer.Option(None, help="Override PPO updates (e.g. 300 for full)."),
    algos: str = typer.Option("mappo,ippo", help="Comma list: mappo,ippo,qmix."),
) -> None:
    """Phase 4: SENTINEL-MAPPO vs IPPO control (CTDE MARL)."""
    from .config import MARLConfig
    from .eval.run_marl import run as run_marl

    cohort = CohortConfig.load()
    if mode:
        cohort = CohortConfig(**{**cohort.__dict__, "mode": mode})
    mcfg = MARLConfig.load()
    if updates:
        mcfg = MARLConfig(**{**mcfg.__dict__, "updates": updates})
    with stage("train-marl"):
        run_marl(cohort, mcfg, seeds=tuple(range(seeds)),
                 algos=tuple(a.strip() for a in algos.split(",")))


@app.command("robustness")
def robustness(
    mode: str = typer.Option(None, help="Override cohort mode: dev | full."),
    seeds: int = typer.Option(3, help="Number of seeds."),
) -> None:
    """Phase 5: missing-signal robustness (drop one organ; SENTINEL vs GRU-single)."""
    from .config import MARLConfig
    from .eval.robustness import run as run_robust

    cohort = CohortConfig.load()
    if mode:
        cohort = CohortConfig(**{**cohort.__dict__, "mode": mode})
    with stage("robustness"):
        run_robust(cohort, MARLConfig.load(), seeds=tuple(range(seeds)))


@app.command("evaluate")
def evaluate() -> None:
    """Phase 6: full evaluation matrix with bootstrap CIs."""
    _todo("Phase 6")


@app.command()
def figures() -> None:
    """Phase 6: generate paper figures into research_paper/figures/."""
    _todo("Phase 6")


if __name__ == "__main__":
    app()
