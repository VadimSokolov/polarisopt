"""Latin Hypercube Sampling via ``scipy.stats.qmc.LatinHypercube``."""

from __future__ import annotations

import numpy as np
from scipy.stats import qmc

from polarisopt.design.base import Design, design_registry
from polarisopt.parameters import ParameterSpace


@design_registry.register("lhs")
class LHSDesign(Design):
    """Latin Hypercube design via :class:`scipy.stats.qmc.LatinHypercube`.

    Parameters
    ----------
    n : int
        Number of sample points (must be positive).
    scramble : bool, optional
        Whether to scramble the LHS (default ``True``).
    include_prior_mean_anchor : bool, optional
        v0.27+ (P11). When True, the first row of the returned batch
        is replaced with the prior-mean anchor vector (per-parameter
        ``prior.mean`` where a prior is set; midpoint of the parameter
        box otherwise). Ensures wave-1 evaluates a defensible starting
        θ regardless of LHS randomness. Requires ``n >= 1``. Default
        ``False`` — existing behavior unchanged.

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
    ...     Parameter("y", "a.json", -1.0, 1.0),
    ... ])
    >>> design = LHSDesign(n=8)
    >>> pts = design.generate(space, rng=np.random.default_rng(0))
    >>> pts.shape
    (8, 2)
    """

    def __init__(
        self,
        n: int,
        *,
        scramble: bool = True,
        include_prior_mean_anchor: bool = False,
    ) -> None:
        if n <= 0:
            raise ValueError(f"LHSDesign: n must be positive, got {n}")
        self.n = int(n)
        self.scramble = bool(scramble)
        self.include_prior_mean_anchor = bool(include_prior_mean_anchor)

    def generate(self, space: ParameterSpace, *, rng: np.random.Generator) -> np.ndarray:
        sampler = qmc.LatinHypercube(d=space.ndim, scramble=self.scramble, rng=rng)
        unit = sampler.random(n=self.n)  # (n, ndim) in [0, 1)
        bounds = space.bounds  # (ndim, 2)
        scaled = unit * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        pts = space.clip(scaled)
        if self.include_prior_mean_anchor:
            # Replace the first row with the prior-mean anchor. Per-parameter:
            # use prior.mean if a prior is set, midpoint of the box otherwise.
            anchor = np.array([
                p.prior.mean if p.prior is not None else 0.5 * (p.low + p.high)
                for p in space.parameters
            ], dtype=float)
            pts[0] = space.clip(anchor)
        return pts
