"""Sobol low-discrepancy sequence via ``scipy.stats.qmc.Sobol``."""

from __future__ import annotations

import numpy as np
from scipy.stats import qmc

from polarisopt.design.base import Design, design_registry
from polarisopt.parameters import ParameterSpace


@design_registry.register("sobol")
class SobolDesign(Design):
    """Sobol design. ``n`` should ideally be a power of 2; the underlying
    scipy generator emits a UserWarning otherwise.

    Parameters
    ----------
    n:
        Number of sample points.
    scramble:
        Owen scrambling (default True) — recommended for unbiased estimates.
    """

    def __init__(self, n: int, *, scramble: bool = True) -> None:
        if n <= 0:
            raise ValueError(f"SobolDesign: n must be positive, got {n}")
        self.n = int(n)
        self.scramble = bool(scramble)

    def generate(self, space: ParameterSpace, *, rng: np.random.Generator) -> np.ndarray:
        sampler = qmc.Sobol(d=space.ndim, scramble=self.scramble, rng=rng)
        unit = sampler.random(n=self.n)
        bounds = space.bounds
        scaled = unit * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        return space.clip(scaled)
