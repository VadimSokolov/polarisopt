"""Hypervolume-based stopping for multi-objective problems.

Stops when the change in Pareto-front hypervolume between iterations is
below a tolerance for ``patience`` consecutive iterations.
"""

from __future__ import annotations

import numpy as np

from polarisopt.stop.base import StoppingCriterion, StoppingState, stop_registry


def _pareto_mask(Y: np.ndarray, *, minimize: bool = True) -> np.ndarray:
    """Boolean mask of non-dominated rows under elementwise <= / >= ordering."""
    Y = -Y if not minimize else Y
    n = Y.shape[0]
    is_dom = np.zeros(n, dtype=bool)
    for i in range(n):
        if is_dom[i]:
            continue
        # i is dominated by some j (j strictly better in at least one obj, no worse in others)
        diff = Y - Y[i]  # Y[j] - Y[i]
        better = (diff <= 0).all(axis=1) & (diff < 0).any(axis=1)
        if better.any():
            is_dom[i] = True
    return ~is_dom


def _hypervolume_2d(points: np.ndarray, ref: np.ndarray) -> float:
    """Exact 2-D HV (minimization). ``points`` are non-dominated, ``ref`` is a worst-case point."""
    if points.size == 0:
        return 0.0
    pts = points[np.argsort(points[:, 0])]
    hv = 0.0
    prev_x = ref[0]
    for p in pts[::-1]:
        if p[1] >= ref[1] or p[0] >= ref[0]:
            continue
        hv += (prev_x - p[0]) * (ref[1] - p[1])
        prev_x = p[0]
    return float(hv)


@stop_registry.register("hypervolume")
class HypervolumeStop(StoppingCriterion):
    """Stop when 2-D Pareto-front HV stops improving by more than ``tol``
    for ``patience`` consecutive iterations.

    Currently supports 2 objectives. For higher m, use BoTorch's
    :class:`Hypervolume` via a custom criterion (planned for v0.2).

    Parameters
    ----------
    tol:
        Minimum HV improvement per iteration to count as progress.
    patience:
        Number of consecutive non-improving iterations before stopping.
    ref_point:
        Reference point in user-space (worst-case for each objective).
        For minimization it should be larger than any expected value.
    """

    def __init__(self, ref_point: list[float], *, tol: float = 1e-3, patience: int = 3) -> None:
        if tol <= 0:
            raise ValueError(f"tol must be > 0, got {tol}")
        if patience <= 0:
            raise ValueError(f"patience must be > 0, got {patience}")
        self.ref_point = np.asarray(ref_point, dtype=float)
        if self.ref_point.shape != (2,):
            raise ValueError(f"HypervolumeStop currently only supports m=2, got ref_point shape {self.ref_point.shape}")
        self.tol = float(tol)
        self.patience = int(patience)
        self._prev_hv: float | None = None
        self._stagnant: int = 0

    def should_stop(self, state: StoppingState) -> bool:
        if state.Y.size == 0 or state.Y.shape[1] != 2:
            return False
        mask = _pareto_mask(state.Y, minimize=state.minimize)
        front = state.Y[mask]
        hv = _hypervolume_2d(front, self.ref_point if state.minimize else -self.ref_point)
        if self._prev_hv is None:
            self._prev_hv = hv
            return False
        if hv - self._prev_hv < self.tol:
            self._stagnant += 1
        else:
            self._stagnant = 0
        self._prev_hv = hv
        return self._stagnant >= self.patience
