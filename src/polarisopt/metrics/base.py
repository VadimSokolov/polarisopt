"""Metric ABC — turn raw simulator output into a numeric objective vector.

Metrics are vector-valued by design. Single-objective is just a length-1
vector. The orchestrator never cares whether you're doing single- or
multi-objective optimization at the Metric layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from polarisopt.utils.registry import Registry


class MetricError(RuntimeError):
    """Raised when a metric can't compute (e.g. missing key in output)."""


class Metric(ABC):
    """Compute an objective vector from a simulator output dict.

    Must return a 1-D :class:`numpy.ndarray` of floats with at least one
    entry. The length is fixed per Metric instance and equals the number
    of objectives (``m``).
    """

    @property
    @abstractmethod
    def n_objectives(self) -> int:
        """Number of objectives this metric returns."""

    @abstractmethod
    def compute(self, output: dict[str, Any]) -> np.ndarray:
        """Compute the metric vector. Shape ``(n_objectives,)``."""


metric_registry: Registry[Metric] = Registry("metric")


def make_metric(spec: dict[str, Any]) -> Metric:
    """Build a Metric from a YAML-style spec ``{"type": "...", "options": {...}}``."""
    if "type" not in spec:
        raise ValueError(f"metric spec missing 'type': {spec!r}")
    cls = metric_registry.get(spec["type"])
    options = spec.get("options", {}) or {}
    return cls(**options)
