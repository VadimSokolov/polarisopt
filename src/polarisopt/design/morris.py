"""Morris elementary-effects screening via SALib."""

from __future__ import annotations

import numpy as np
from SALib.sample import morris as salib_morris

from polarisopt.design.base import Design, design_registry
from polarisopt.parameters import ParameterSpace


@design_registry.register("morris")
class MorrisDesign(Design):
    """Morris elementary-effects screening design via SALib.

    Returns ``n_trajectories * (ndim + 1)`` sample rows — the standard Morris
    grid trajectory structure that SALib's ``morris.sample`` produces.

    Parameters
    ----------
    n_trajectories : int
        Number of trajectories (``N`` in the SALib API).
    num_levels : int, optional
        Number of grid levels per dimension (default 4).

    Raises
    ------
    ValueError
        If ``n_trajectories <= 0`` or ``num_levels < 2``.

    See Also
    --------
    SALib documentation : https://salib.readthedocs.io/

    Examples
    --------
    >>> import numpy as np
    >>> from polarisopt.parameters import Parameter, ParameterSpace
    >>> space = ParameterSpace.from_iterable([
    ...     Parameter("x", "a.json", 0.0, 1.0),
    ...     Parameter("y", "a.json", 0.0, 1.0),
    ... ])
    >>> design = MorrisDesign(n_trajectories=3, num_levels=4)
    >>> pts = design.generate(space, rng=np.random.default_rng(0))
    >>> # N=3 trajectories over d=2 dims => 3 * (2+1) = 9 points
    >>> pts.shape
    (9, 2)
    """

    def __init__(self, n_trajectories: int, *, num_levels: int = 4) -> None:
        if n_trajectories <= 0:
            raise ValueError(f"n_trajectories must be positive, got {n_trajectories}")
        if num_levels < 2:
            raise ValueError(f"num_levels must be >= 2, got {num_levels}")
        self.n_trajectories = int(n_trajectories)
        self.num_levels = int(num_levels)

    def generate(self, space: ParameterSpace, *, rng: np.random.Generator) -> np.ndarray:
        problem = {
            "num_vars": space.ndim,
            "names": list(space.names),
            "bounds": space.bounds.tolist(),
        }
        seed = int(rng.integers(0, 2**31 - 1))
        x = salib_morris.sample(
            problem,
            N=self.n_trajectories,
            num_levels=self.num_levels,
            seed=seed,
        )
        return space.clip(np.asarray(x, dtype=float))
