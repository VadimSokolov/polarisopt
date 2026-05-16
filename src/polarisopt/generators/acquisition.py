"""AcquisitionGenerator — fit Surrogate to history, optimize Acquisition for next q.

The standard Bayesian-optimization sample generator. Configurable via two
sub-specs:

    surrogate: { type: gp, options: { nu: 2.5 } }
    acquisition: { type: qei, options: { mc_samples: 256 } }

Both sub-specs flow through the registry factories so users can register
new surrogates/acquisitions externally and reference them by name.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from polarisopt.acquisition.base import make_acquisition
from polarisopt.generators.base import (
    GeneratorContext,
    SampleGenerator,
    SampleGeneratorError,
    generator_registry,
)
from polarisopt.surrogates.base import make_surrogate


@generator_registry.register("acquisition")
class AcquisitionGenerator(SampleGenerator):
    """Build a fresh Surrogate + Acquisition each call and return its candidates."""

    def __init__(
        self,
        surrogate: dict[str, Any],
        acquisition: dict[str, Any],
        *,
        minimize: bool = True,
    ) -> None:
        if not isinstance(surrogate, dict):
            raise TypeError("AcquisitionGenerator.surrogate must be a dict spec")
        if not isinstance(acquisition, dict):
            raise TypeError("AcquisitionGenerator.acquisition must be a dict spec")
        self.surrogate_spec = dict(surrogate)
        self.acquisition_spec = dict(acquisition)
        self.minimize = bool(minimize)

    def next(self, ctx: GeneratorContext, *, q: int) -> np.ndarray:
        if ctx.X.shape[0] < 2:
            raise SampleGeneratorError(
                "AcquisitionGenerator needs >=2 finished samples; "
                "configure a larger warm-up design"
            )
        surrogate = make_surrogate(self.surrogate_spec)
        surrogate.fit(ctx.X, ctx.Y)

        acq = make_acquisition(self.acquisition_spec, surrogate=surrogate, minimize=self.minimize)
        return acq.optimize(ctx.space, q=q, observed_Y=ctx.Y, rng=ctx.rng)
