"""Morris elementary-effects screening via SALib."""

from __future__ import annotations

import numpy as np
from SALib.sample import morris as salib_morris

from polarisopt.design.base import Design, design_registry
from polarisopt.parameters import ParameterSpace


@design_registry.register("morris")
class MorrisDesign(Design):
    """Morris screening design.

    Returns ``n_trajectories * (ndim + 1)`` sample rows — the standard Morris
    grid trajectory structure that SALib's ``morris.sample`` produces.

    Parameters
    ----------
    n_trajectories:
        Number of trajectories (``N`` in the SALib API).
    num_levels:
        Number of grid levels per dimension (typically 4).
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
