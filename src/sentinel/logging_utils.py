"""Logging + per-stage resource accounting (wall-clock, peak RAM, peak VRAM).

Every pipeline stage runs inside ``stage(...)`` which:
  * logs start/end at INFO,
  * samples process RSS in a background thread to capture *peak* RAM,
  * reads peak VRAM from torch **only if torch is already imported** (so pure
    data stages don't pay the multi-second torch import cost),
  * appends a row to ``outputs/logs/stages.csv`` for the paper's compute table.
"""
from __future__ import annotations

import csv
import logging
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass

import psutil

from .paths import PATHS

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_CONFIGURED = False


def get_logger(name: str = "sentinel") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        try:
            from rich.logging import RichHandler

            handler: logging.Handler = RichHandler(
                rich_tracebacks=True, show_path=False, markup=False
            )
            fmt = "%(message)s"
        except Exception:  # pragma: no cover - rich always present here
            handler = logging.StreamHandler(sys.stderr)
            fmt = _LOG_FORMAT
        handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
        root = logging.getLogger("sentinel")
        root.setLevel(logging.INFO)
        root.handlers[:] = [handler]
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(name if name.startswith("sentinel") else f"sentinel.{name}")


class _PeakRSS(threading.Thread):
    """Background sampler tracking peak RSS of this process (MB)."""

    def __init__(self, interval: float = 0.25) -> None:
        super().__init__(daemon=True)
        self._proc = psutil.Process()
        self._interval = interval
        self._stop = threading.Event()
        self.peak_mb = self._proc.memory_info().rss / 1e6

    def run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                rss = self._proc.memory_info().rss / 1e6
                if rss > self.peak_mb:
                    self.peak_mb = rss
            except psutil.Error:  # pragma: no cover
                break

    def stop(self) -> float:
        self._stop.set()
        self.join(timeout=2.0)
        return self.peak_mb


def _peak_vram_mb() -> float | None:
    """Peak VRAM in MB since last reset, or None if torch/CUDA unused."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        return torch.cuda.max_memory_allocated() / 1e6
    except Exception:  # pragma: no cover
        return None


def _reset_vram_peak() -> None:
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:  # pragma: no cover
            pass


@dataclass
class StageResult:
    name: str
    seconds: float
    peak_rss_mb: float
    peak_vram_mb: float | None


@contextmanager
def stage(name: str, logger: logging.Logger | None = None):
    """Context manager timing a pipeline stage and recording peak memory."""
    log = logger or get_logger()
    log.info("[stage:%s] start", name)
    _reset_vram_peak()
    sampler = _PeakRSS()
    sampler.start()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        peak_rss = sampler.stop()
        peak_vram = _peak_vram_mb()
        vram_str = f"{peak_vram:.0f}MB" if peak_vram is not None else "n/a"
        log.info(
            "[stage:%s] done in %.1fs | peak RAM %.0fMB | peak VRAM %s",
            name, elapsed, peak_rss, vram_str,
        )
        _record_stage(StageResult(name, elapsed, peak_rss, peak_vram))


def _record_stage(res: StageResult) -> None:
    PATHS.logs_root.mkdir(parents=True, exist_ok=True)
    path = PATHS.logs_root / "stages.csv"
    new = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["timestamp", "stage", "seconds", "peak_rss_mb", "peak_vram_mb"])
        w.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            res.name,
            f"{res.seconds:.2f}",
            f"{res.peak_rss_mb:.1f}",
            "" if res.peak_vram_mb is None else f"{res.peak_vram_mb:.1f}",
        ])
