"""ConstantMetric — for studies that produce artifacts, not optimization objectives.

When you're running a Phase-1 baseline or producing inputs that
downstream studies will consume, there's no objective to optimize but
polarisopt still requires a Metric in the YAML. ``ConstantMetric``
documents that intent explicitly — every sample returns the same fixed
value, so any optimizer would be a no-op.

Aliased as ``null_metric`` for users who prefer that name.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from polarisopt.metrics.base import Metric, metric_registry


@metric_registry.register("constant")
@metric_registry.register("null_metric")
class ConstantMetric(Metric):
    """Return the same fixed value for every sample.

    Use this for artifact-producing studies that don't optimize anything
    (e.g. baseline runs, inputs for downstream studies). Documents the
    intent that this is *not* an optimization metric.

    Parameters
    ----------
    value : float or list of float, optional
        The constant to return. Length determines ``n_objectives``.
        Default ``0.0`` (single-objective).

    Raises
    ------
    ValueError
        If ``value`` is an empty list.

    Examples
    --------
    >>> m = ConstantMetric()
    >>> m.n_objectives
    1
    >>> m.compute({"anything": "ignored"}).tolist()
    [0.0]

    Multi-objective placeholder:

    >>> m = ConstantMetric(value=[0.0, 0.0])
    >>> m.n_objectives
    2
    """

    def __init__(self, value: float | list[float] = 0.0) -> None:
        if isinstance(value, list):
            if not value:
                raise ValueError("ConstantMetric: value list must be non-empty")
            self._values = np.asarray(value, dtype=float)
        else:
            self._values = np.asarray([float(value)], dtype=float)

    @property
    def n_objectives(self) -> int:
        return int(self._values.shape[0])

    def compute(self, output: dict[str, Any]) -> np.ndarray:  # noqa: ARG002
        # Output dict is intentionally unused — the metric is constant.
        return self._values.copy()
