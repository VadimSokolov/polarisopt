"""polarisopt metric plugin: registers the ``output_match`` metric type.

The toy analog of ``link_moe``: a calibration discrepancy between simulator
outputs and observed (target) values. Discovered through the
``polarisopt.metrics`` entry point declared in this package's
``pyproject.toml``.

Example study snippet::

    metric:
      type: output_match
      options:
        targets:                      # observed values to reproduce
          journeys_completed: 331.0
          pick_up_time: 95.4
          missed: 78.0
        # or, instead of inline values:
        # targets_file: ~/taxidemo-runs/calibration-targets.json

The metric is the mean over keys of the squared relative error
``((simulated - target) / scale)**2`` with ``scale = max(|target|, 1)``
unless overridden via ``scales``. Zero means a perfect match, so studies
minimize it (the sequential phase's default).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from polarisopt.metrics.base import Metric, MetricError, metric_registry


@metric_registry.register("output_match")
class OutputMatchMetric(Metric):
    """Mean squared relative error between simulator outputs and targets.

    Parameters
    ----------
    targets : dict[str, float], optional
        Observed values keyed by simulator output name.
    targets_file : str, optional
        Path to a JSON file holding the same ``{name: value}`` mapping.
        Exactly one of ``targets`` / ``targets_file`` must be given;
        ``~`` is expanded.
    scales : dict[str, float], optional
        Per-key normalization. Defaults to ``max(|target|, 1)`` so outputs
        of different magnitude (counts vs minutes) contribute comparably.
    """

    def __init__(
        self,
        targets: dict[str, float] | None = None,
        targets_file: str | None = None,
        scales: dict[str, float] | None = None,
    ) -> None:
        if (targets is None) == (targets_file is None):
            raise MetricError("output_match: give exactly one of 'targets' or 'targets_file'")
        if targets_file is not None:
            path = Path(targets_file).expanduser()
            if not path.exists():
                raise MetricError(f"output_match: targets_file not found: {path}")
            targets = json.loads(path.read_text())
        if not targets:
            raise MetricError("output_match: targets must be non-empty")
        self._targets = {str(k): float(v) for k, v in targets.items()}
        scales = scales or {}
        self._scales = {k: float(scales.get(k, max(abs(t), 1.0))) for k, t in self._targets.items()}
        bad = [k for k, s in self._scales.items() if s <= 0]
        if bad:
            raise MetricError(f"output_match: scales must be positive, got {bad}")

    @property
    def n_objectives(self) -> int:
        return 1

    @property
    def targets(self) -> dict[str, float]:
        return dict(self._targets)

    def compute(self, output: dict[str, Any]) -> np.ndarray:
        errors = []
        for key, target in self._targets.items():
            if key not in output:
                raise MetricError(f"output_match: key {key!r} not in simulator output (keys={list(output)})")
            value = float(output[key])
            if not np.isfinite(value):
                raise MetricError(f"output_match: output[{key!r}] is not finite: {value}")
            errors.append(((value - target) / self._scales[key]) ** 2)
        return np.asarray([float(np.mean(errors))])
