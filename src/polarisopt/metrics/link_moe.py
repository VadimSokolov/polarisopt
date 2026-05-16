"""Link Measures-of-Effectiveness metric.

Compares POLARIS link-level outputs (travel time × in-volume) to a target
HDF5 result file. Produces an aggregate scalar (default RMSE) which the
optimizer minimizes.

POLARIS result HDF5 layout assumed::

    link_moe/
        link_travel_time   shape (n_links, n_intervals)
        link_in_volume     shape (n_links, n_intervals)

The metric reads both arrays, multiplies them elementwise to form a
"vehicle-time" proxy, averages over intervals to a per-link vector, and
returns a scalar comparing simulated to target.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np

from polarisopt.metrics.base import Metric, MetricError, metric_registry

AggregationKind = Literal["rmse", "mse", "mae"]


def _vehicle_time_per_link(path: Path) -> np.ndarray:
    """Read link_moe from a POLARIS result and return a per-link vector."""
    if not path.exists():
        raise MetricError(f"result file not found: {path}")
    with h5py.File(path, "r") as f:
        if "link_moe" not in f:
            raise MetricError(f"{path}: missing 'link_moe' group")
        grp = f["link_moe"]
        for k in ("link_travel_time", "link_in_volume"):
            if k not in grp:
                raise MetricError(f"{path}: link_moe/{k} missing")
        tt = np.asarray(grp["link_travel_time"][...])
        vol = np.asarray(grp["link_in_volume"][...])
    if tt.shape != vol.shape:
        raise MetricError(
            f"link_travel_time {tt.shape} and link_in_volume {vol.shape} disagree"
        )
    # average over time intervals → per-link
    return np.mean(tt * vol, axis=1)


@metric_registry.register("link_moe")
class LinkMoeMetric(Metric):
    """RMSE / MSE / MAE of per-link vehicle-time vs a target HDF5.

    Parameters
    ----------
    target:
        Path to the target POLARIS result HDF5.
    aggregation:
        How to summarize the per-link error vector into a scalar
        (``rmse`` [default], ``mse``, or ``mae``).
    """

    def __init__(self, target: Path | str, *, aggregation: AggregationKind = "rmse") -> None:
        self.target_path = Path(target)
        if aggregation not in ("rmse", "mse", "mae"):
            raise ValueError(f"unknown aggregation: {aggregation!r}")
        self.aggregation: AggregationKind = aggregation
        self._target_cache: np.ndarray | None = None

    @property
    def n_objectives(self) -> int:
        return 1

    def _target(self) -> np.ndarray:
        if self._target_cache is None:
            self._target_cache = _vehicle_time_per_link(self.target_path)
        return self._target_cache

    def compute(self, output: dict[str, Any]) -> np.ndarray:
        if "result_path" not in output:
            raise MetricError("LinkMoeMetric: simulator output missing 'result_path'")
        sim_vec = _vehicle_time_per_link(Path(output["result_path"]))
        tgt_vec = self._target()
        if sim_vec.shape != tgt_vec.shape:
            raise MetricError(
                f"sim and target link vectors disagree: {sim_vec.shape} vs {tgt_vec.shape}"
            )
        err = sim_vec - tgt_vec
        if self.aggregation == "mae":
            value = float(np.mean(np.abs(err)))
        elif self.aggregation == "mse":
            value = float(np.mean(err**2))
        else:  # rmse
            value = float(np.sqrt(np.mean(err**2)))
        return np.array([value])
