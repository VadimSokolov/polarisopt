"""Sobol low-discrepancy sequence via ``scipy.stats.qmc.Sobol``."""

from __future__ import annotations

import numpy as np
from scipy.stats import qmc

from polarisopt.design.base import Design, design_registry
from polarisopt.parameters import ParameterSpace


@design_registry.register("sobol")
class SobolDesign(Design):
    """Sobol low-discrepancy sequence via :class:`scipy.stats.qmc.Sobol`.

    Parameters
    ----------
    n : int
        Number of sample points. Should ideally be a power of 2 for
        balance; scipy emits a ``UserWarning`` otherwise.
    scramble : bool, optional
        Owen scrambling (default ``True``) — recommended for unbiased
        sample estimates.

    Raises
    ------
    ValueError
        If ``n <= 0``.

    Examples
    --------
    >>> import numpy as np
    >>> from polarisopt.parameters import Parameter, ParameterSpace
    >>> space = ParameterSpace.from_iterable([
    ...     Parameter("x", "a.json", 0.0, 1.0),
    ...     Parameter("y", "a.json", 0.0, 1.0),
    ... ])
    >>> pts = SobolDesign(n=8).generate(space, rng=np.random.default_rng(0))
    >>> pts.shape
    (8, 2)
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
