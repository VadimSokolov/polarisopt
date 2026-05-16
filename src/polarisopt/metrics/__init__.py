"""Metric ABC and built-in metrics."""

from polarisopt.metrics.base import Metric, MetricError, make_metric, metric_registry
from polarisopt.metrics.identity import IdentityMetric

__all__ = [
    "IdentityMetric",
    "Metric",
    "MetricError",
    "make_metric",
    "metric_registry",
]
