"""Compare two SampleStores side by side.

Used by ``polarisopt diff <a.yaml> <b.yaml>`` to compare runs of the
same study with different settings (e.g. baseline vs. tuned), or two
restarts of the same study at different snapshots.

Outputs a compact, human-readable summary plus a structured dict that
notebooks can consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from polarisopt.config.schema import StudyConfig
from polarisopt.samples.sample import Sample, SampleStatus
from polarisopt.samples.store import SampleStore
from polarisopt.studies.ops import open_store


@dataclass
class StudyDiff:
    """Side-by-side summary of two SampleStores.

    Each entry holds a per-store value (``a`` and ``b``) plus a delta
    where it makes sense. ``samples`` and ``finished`` are integers;
    ``best_metric`` is per-objective; ``pareto_size`` is the count of
    non-dominated finished samples (multi-objective only).
    """

    name_a: str = ""
    name_b: str = ""
    samples: tuple[int, int] = (0, 0)
    finished: tuple[int, int] = (0, 0)
    failed: tuple[int, int] = (0, 0)
    best_metric: tuple[list[float] | None, list[float] | None] = (None, None)
    pareto_size: tuple[int | None, int | None] = (None, None)
    n_objectives: int = 0
    extras: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        """Plain-text summary suitable for the CLI."""
        col_a, col_b = self.name_a or "A", self.name_b or "B"
        col_width = max(len(col_a), len(col_b), 12)
        lines: list[str] = [
            f"{'metric':<22} {col_a:>{col_width}}  {col_b:>{col_width}}",
            "-" * (24 + col_width * 2),
        ]

        def _row(label: str, a: Any, b: Any) -> str:
            return f"{label:<22} {_fmt(a):>{col_width}}  {_fmt(b):>{col_width}}"

        lines.append(_row("samples", self.samples[0], self.samples[1]))
        lines.append(_row("finished", self.finished[0], self.finished[1]))
        lines.append(_row("failed", self.failed[0], self.failed[1]))
        if self.n_objectives == 1:
            ba = self.best_metric[0][0] if self.best_metric[0] else None
            bb = self.best_metric[1][0] if self.best_metric[1] else None
            lines.append(_row("best metric", ba, bb))
        else:
            lines.append(_row("pareto front size", self.pareto_size[0], self.pareto_size[1]))
            ba = self.best_metric[0]
            bb = self.best_metric[1]
            lines.append(_row("best per obj", ba, bb))
        return "\n".join(lines)


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, list):
        return "[" + ", ".join(f"{x:.4g}" if isinstance(x, float) else str(x) for x in v) + "]"
    return str(v)


def _pareto_mask(Y: np.ndarray, *, minimize: bool = True) -> np.ndarray:
    Y = -Y if not minimize else Y
    n = Y.shape[0]
    is_dom = np.zeros(n, dtype=bool)
    for i in range(n):
        if is_dom[i]:
            continue
        diff = Y - Y[i]
        dominated = (diff <= 0).all(axis=1) & (diff < 0).any(axis=1)
        if dominated.any():
            is_dom[i] = True
    return ~is_dom


def _summarize(store: SampleStore) -> tuple[int, int, int, list[float] | None, int | None, int]:
    """Return (n_total, n_finished, n_failed, best_per_obj, pareto_size, n_obj)."""
    all_samples: list[Sample] = store.list()
    n_total = len(all_samples)
    finished = [s for s in all_samples if s.status is SampleStatus.FINISHED and s.metric is not None]
    n_failed = sum(1 for s in all_samples if s.status is SampleStatus.FAILED)
    if not finished:
        return n_total, len(finished), n_failed, None, None, 0
    Y = np.stack([s.metric for s in finished])  # type: ignore[arg-type]
    n_obj = Y.shape[1]
    if n_obj == 1:
        return n_total, len(finished), n_failed, [float(np.min(Y))], None, 1
    mask = _pareto_mask(Y, minimize=True)
    best = [float(np.min(Y[:, j])) for j in range(n_obj)]
    return n_total, len(finished), n_failed, best, int(mask.sum()), n_obj


def diff_studies(
    config_a: Path | str | StudyConfig,
    config_b: Path | str | StudyConfig,
) -> StudyDiff:
    """Compare two studies, identified by their YAML paths or configs.

    Both studies must have been started (their SampleStore exists); a
    fresh study with no samples yet still works — it shows up as zero
    counts on its side.

    Parameters
    ----------
    config_a, config_b : path or StudyConfig
        Either study YAML paths or pre-loaded :class:`StudyConfig`
        instances.

    Returns
    -------
    StudyDiff
    """
    cfg_a = _coerce_cfg(config_a)
    cfg_b = _coerce_cfg(config_b)
    store_a = open_store(cfg_a)
    store_b = open_store(cfg_b)
    (na, fa, ea, ba, pa, oa) = _summarize(store_a)
    (nb, fb, eb, bb, pb, ob) = _summarize(store_b)
    n_obj = max(oa, ob)
    return StudyDiff(
        name_a=cfg_a.name,
        name_b=cfg_b.name,
        samples=(na, nb),
        finished=(fa, fb),
        failed=(ea, eb),
        best_metric=(ba, bb),
        pareto_size=(pa, pb),
        n_objectives=n_obj,
    )


def _coerce_cfg(cfg: Path | str | StudyConfig) -> StudyConfig:
    if isinstance(cfg, StudyConfig):
        return cfg
    from polarisopt.config import load_study_config

    return load_study_config(cfg)
