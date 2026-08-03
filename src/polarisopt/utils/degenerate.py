"""Shared near-constant column detection (v0.37).

Prior to v0.37 every call site tested ``np.ptp(col) == 0.0`` exactly, so
a column that varies only by floating-point noise — e.g. a share
computed as a ratio of identical integer counts across designs, differing
in the last ulp — was treated as live. Fitting a GP to pure noise then
produces a meaningless posterior that still enters implausibility and
model-discrepancy estimates. Compare against a scale-relative tolerance
instead.
"""

from __future__ import annotations

import numpy as np

# Relative to the column's own magnitude: a column whose peak-to-peak
# spread is below this fraction of its typical absolute value carries no
# usable signal. 1e-9 is ~4 orders above float64 eps, so it catches
# accumulated round-off without discarding genuinely small variation.
DEFAULT_REL_TOL = 1e-9
DEFAULT_ABS_TOL = 1e-300


def is_near_constant(
    col: np.ndarray, *, rel_tol: float = DEFAULT_REL_TOL, abs_tol: float = DEFAULT_ABS_TOL,
) -> bool:
    """True when ``col`` carries no usable variation.

    A column counts as near-constant when its peak-to-peak range is at
    or below ``max(abs_tol, rel_tol * scale)`` where ``scale`` is the
    largest absolute value in the column (so an all-zero column, whose
    scale is 0, still falls back to ``abs_tol``).
    """
    arr = np.asarray(col, dtype=float).reshape(-1)
    if arr.size == 0:
        return True
    finite = arr[np.isfinite(arr)]
    if finite.size < 2:
        return True
    spread = float(np.ptp(finite))
    scale = float(np.max(np.abs(finite)))
    return spread <= max(abs_tol, rel_tol * scale)


def near_constant_mask(
    values: np.ndarray,
    *,
    rel_tol: float = DEFAULT_REL_TOL,
    abs_tol: float = DEFAULT_ABS_TOL,
) -> np.ndarray:
    """Boolean mask over the columns of an ``(n, m)`` array.

    See :func:`is_near_constant` for the per-column rule.
    """
    arr = np.atleast_2d(np.asarray(values, dtype=float))
    return np.array(
        [
            is_near_constant(arr[:, j], rel_tol=rel_tol, abs_tol=abs_tol)
            for j in range(arr.shape[1])
        ],
        dtype=bool,
    )
