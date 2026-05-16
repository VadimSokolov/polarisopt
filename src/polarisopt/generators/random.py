"""Random generator — uniform draws within the space. Baseline / debugging tool."""

from __future__ import annotations

import numpy as np

from polarisopt.generators.base import (
    GeneratorContext,
    SampleGenerator,
    generator_registry,
)


@generator_registry.register("random")
class RandomGenerator(SampleGenerator):
    """Pick the next batch uniformly at random within the space bounds."""

    def next(self, ctx: GeneratorContext, *, q: int) -> np.ndarray:
        if q < 1:
            raise ValueError(f"q must be >= 1, got {q}")
        bounds = ctx.space.bounds  # (d, 2)
        unit = ctx.rng.random(size=(q, ctx.space.ndim))
        scaled = unit * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        return ctx.space.clip(scaled)
