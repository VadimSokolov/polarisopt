"""Latin Hypercube Sampling via ``scipy.stats.qmc.LatinHypercube``."""

from __future__ import annotations

import numpy as np
from scipy.stats import qmc

from polarisopt.design.base import Design, design_registry
from polarisopt.parameters import ParameterSpace


@design_registry.register("lhs")
class LHSDesign(Design):
    """Latin Hypercube design.

    Parameters
    ----------
    n:
        Number of sample points.
    scramble:
        Pass through to scipy's ``LatinHypercube(scramble=...)``.
    """

    def __init__(self, n: int, *, scramble: bool = True) -> None:
        if n <= 0:
            raise ValueError(f"LHSDesign: n must be positive, got {n}")
        self.n = int(n)
        self.scramble = bool(scramble)

    def generate(self, space: ParameterSpace, *, rng: np.random.Generator) -> np.ndarray:
        sampler = qmc.LatinHypercube(d=space.ndim, scramble=self.scramble, rng=rng)
        unit = sampler.random(n=self.n)  # (n, ndim) in [0, 1)
        bounds = space.bounds  # (ndim, 2)
        scaled = unit * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        return space.clip(scaled)
