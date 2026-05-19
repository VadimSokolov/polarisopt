"""Hypervolume-based stopping for multi-objective problems.

Stops when the Pareto-front hypervolume stagnates for ``patience``
consecutive iterations. Supports arbitrary ``m`` via BoTorch's
:class:`Hypervolume` when the ``[bo]`` extra is installed; falls back
to a hand-rolled 2-D implementation otherwise.
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
        diff = Y - Y[i]  # Y[j] - Y[i]
        better = (diff <= 0).all(axis=1) & (diff < 0).any(axis=1)
        if better.any():
            is_dom[i] = True
    return ~is_dom


def _hypervolume_2d(points: np.ndarray, ref: np.ndarray) -> float:
    """Exact 2-D hypervolume (minimization view).

    ``points`` must be non-dominated; ``ref`` is a worst-case point that
    every point in ``points`` strictly dominates.
    """
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


def _hypervolume_botorch(points: np.ndarray, ref: np.ndarray) -> float:
    """Arbitrary-dimensional hypervolume via BoTorch (requires [bo] extra).

    BoTorch's :class:`Hypervolume` maximizes, so we negate both
    points and ref to compute minimization HV.
    """
    import torch
    from botorch.utils.multi_objective.hypervolume import Hypervolume

    if points.size == 0:
        return 0.0
    neg_points = torch.as_tensor(-points, dtype=torch.double)
    neg_ref = torch.as_tensor(-ref, dtype=torch.double)
    hv = Hypervolume(ref_point=neg_ref)
    return float(hv.compute(neg_points))


def _compute_hv(points: np.ndarray, ref: np.ndarray) -> float:
    """Dispatch to the right HV implementation based on dimensionality."""
    if points.ndim != 2 or ref.ndim != 1:
        raise ValueError(
            f"shape mismatch: points {points.shape}, ref {ref.shape}"
        )
    m = ref.shape[0]
    if m == 2:
        return _hypervolume_2d(points, ref)
    # 3+ objectives: fall back to BoTorch
    try:
        return _hypervolume_botorch(points, ref)
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            f"HypervolumeStop with m={m} requires the [bo] extra "
            f"(pip install 'polarisopt[bo]')"
        ) from exc


@stop_registry.register("hypervolume")
class HypervolumeStop(StoppingCriterion):
    """Stop when Pareto-front hypervolume stops improving.

    Supports arbitrary number of objectives. For 2 objectives the
    hypervolume is computed exactly in pure NumPy. For 3+ objectives,
    requires the ``[bo]`` extra (uses
    :class:`botorch.utils.multi_objective.hypervolume.Hypervolume`).

    Parameters
    ----------
    ref_point : list of float
        Reference point in user-facing objective space (worst case for
        each objective). For minimization (default), it should be
        strictly larger than any expected objective value.
    tol : float, optional
        Minimum HV improvement per iteration that counts as progress.
        Default ``1e-3``.
    patience : int, optional
        Consecutive non-improving iterations before stopping. Default 3.

    Raises
    ------
    ValueError
        If ``tol <= 0`` or ``patience <= 0``.

    Examples
    --------
    >>> stop = HypervolumeStop(ref_point=[10.0, 10.0])
    >>> stop.patience
    3
    """

    def __init__(
        self,
        ref_point: list[float],
        *,
        tol: float = 1e-3,
        patience: int = 3,
    ) -> None:
        if tol <= 0:
            raise ValueError(f"tol must be > 0, got {tol}")
        if patience <= 0:
            raise ValueError(f"patience must be > 0, got {patience}")
        self.ref_point = np.asarray(ref_point, dtype=float)
        if self.ref_point.ndim != 1 or self.ref_point.shape[0] < 2:
            raise ValueError(
                f"ref_point must be a vector of length >= 2, got shape {self.ref_point.shape}"
            )
        self.tol = float(tol)
        self.patience = int(patience)
        self._prev_hv: float | None = None
        self._stagnant: int = 0

    @property
    def n_objectives(self) -> int:
        return int(self.ref_point.shape[0])

    def should_stop(self, state: StoppingState) -> bool:
        if state.Y.size == 0 or state.Y.shape[1] != self.n_objectives:
            return False
        mask = _pareto_mask(state.Y, minimize=state.minimize)
        front = state.Y[mask]
        # In user-facing coordinates ref is the worst-case. For
        # maximization, _pareto_mask negates Y so we negate the ref too.
        ref = self.ref_point if state.minimize else -self.ref_point
        # _compute_hv treats inputs as minimization; for maximize we
        # pass the negated front so the comparison directions match.
        front_view = front if state.minimize else -front
        hv = _compute_hv(front_view, ref)
        if self._prev_hv is None:
            self._prev_hv = hv
            return False
        if hv - self._prev_hv < self.tol:
            self._stagnant += 1
        else:
            self._stagnant = 0
        self._prev_hv = hv
        return self._stagnant >= self.patience
