"""Standard optimization benchmark functions (https://www.sfu.ca/~ssurjano/optimization.html).

Each function ``f(x: np.ndarray) -> float`` accepts a 1-D input vector and
returns a scalar. Used by :class:`~polarisopt.simulator.mock.MockSimulator`
to exercise the full master/slave pipeline without POLARIS.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

BenchmarkFn = Callable[[np.ndarray], float]


def branin(x: np.ndarray) -> float:
    """Branin function. Recommended input box: x1 in [-5, 10], x2 in [0, 15].
    Global minimum value ≈ 0.397887 at three locations."""
    if x.shape != (2,):
        raise ValueError(f"branin requires shape (2,), got {x.shape}")
    x1, x2 = x[0], x[1]
    a = 1.0
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6.0
    s = 10.0
    t = 1 / (8 * np.pi)
    return float(a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s)


def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock banana function, n-D. Global min 0 at x = (1, 1, ..., 1).
    Recommended box: x_i in [-5, 10]."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError(f"rosenbrock requires 1-D input with length >= 2, got {x.shape}")
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


_H6_A = np.array(
    [
        [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
        [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
        [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
        [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
    ]
)
_H6_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
_H6_P = 1e-4 * np.array(
    [
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381],
    ]
)


def hartmann6(x: np.ndarray) -> float:
    """Hartmann-6. Input box [0, 1]^6. Global min ≈ -3.32237 at
    (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)."""
    if x.shape != (6,):
        raise ValueError(f"hartmann6 requires shape (6,), got {x.shape}")
    inner = np.sum(_H6_A * (x - _H6_P) ** 2, axis=1)
    return float(-np.sum(_H6_ALPHA * np.exp(-inner)))


def quadratic(x: np.ndarray) -> float:
    """Simple sum-of-squares. Global min 0 at the origin. Useful as a smoke test."""
    return float(np.sum(np.asarray(x, dtype=float) ** 2))


BENCHMARKS: dict[str, BenchmarkFn] = {
    "branin": branin,
    "rosenbrock": rosenbrock,
    "hartmann6": hartmann6,
    "quadratic": quadratic,
}
