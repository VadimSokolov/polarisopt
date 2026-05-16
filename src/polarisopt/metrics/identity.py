"""IdentityMetric — pluck one or more numeric values straight out of the output dict.

For mock studies and any setup where the simulator already produces the
objective in its output file.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from polarisopt.metrics.base import Metric, MetricError, metric_registry


@metric_registry.register("identity")
class IdentityMetric(Metric):
    """Read one or more scalar fields from the simulator output.

    Parameters
    ----------
    keys:
        Either a single key name (single-objective) or a list of keys
        (multi-objective). Each referenced value must be a finite scalar.
    """

    def __init__(self, keys: str | list[str] = "value") -> None:
        if isinstance(keys, str):
            self._keys: tuple[str, ...] = (keys,)
        else:
            if not keys:
                raise ValueError("IdentityMetric: keys must be non-empty")
            self._keys = tuple(str(k) for k in keys)

    @property
    def n_objectives(self) -> int:
        return len(self._keys)

    @property
    def keys(self) -> tuple[str, ...]:
        return self._keys

    def compute(self, output: dict[str, Any]) -> np.ndarray:
        values: list[float] = []
        for key in self._keys:
            if key not in output:
                raise MetricError(f"IdentityMetric: key {key!r} not in simulator output (keys={list(output)})")
            v = output[key]
            try:
                fv = float(v)
            except (TypeError, ValueError) as exc:
                raise MetricError(f"IdentityMetric: output[{key!r}] is not numeric: {v!r}") from exc
            if not np.isfinite(fv):
                raise MetricError(f"IdentityMetric: output[{key!r}] is not finite: {fv}")
            values.append(fv)
        return np.asarray(values, dtype=float)
