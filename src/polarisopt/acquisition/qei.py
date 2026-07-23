"""Monte-Carlo batch Expected Improvement (q-EI) for single-objective."""

from __future__ import annotations

import numpy as np

try:
    import torch
    from botorch.acquisition.logei import qLogExpectedImprovement
    from botorch.acquisition.objective import GenericMCObjective
    from botorch.optim import optimize_acqf
    from botorch.sampling.normal import SobolQMCNormalSampler
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "polarisopt.acquisition.qei requires the [bo] extra: "
        "pip install 'polarisopt[bo]'"
    ) from exc

from polarisopt.acquisition.base import (
    AcquisitionError,
    AcquisitionFunction,
    acquisition_registry,
)
from polarisopt.parameters import ParameterSpace
from polarisopt.surrogates.gp import GPSurrogate


@acquisition_registry.register("qei")
class QEIAcquisition(AcquisitionFunction):
    """Batch q-EI for single-objective problems.

    Parameters
    ----------
    mc_samples:
        Number of Sobol QMC posterior samples (default 128).
    num_restarts, raw_samples:
        Passed to :func:`botorch.optim.optimize_acqf`.
    """

    def __init__(
        self,
        surrogate,
        *,
        minimize: bool = True,
        mc_samples: int = 128,
        num_restarts: int = 10,
        raw_samples: int = 256,
    ) -> None:
        super().__init__(surrogate=surrogate, minimize=minimize)
        if not isinstance(surrogate, GPSurrogate):
            raise AcquisitionError("QEIAcquisition requires a GPSurrogate")
        self.mc_samples = int(mc_samples)
        self.num_restarts = int(num_restarts)
        self.raw_samples = int(raw_samples)

    def optimize(
        self,
        space: ParameterSpace,
        *,
        q: int,
        observed_Y: np.ndarray,
        rng: np.random.Generator,
        observed_X: np.ndarray | None = None,  # v0.18: accepted for signature parity; unused
    ) -> np.ndarray:
        if q < 1:
            raise AcquisitionError(f"q must be >= 1, got {q}")
        gp = self.surrogate
        if gp.n_objectives != 1:
            raise AcquisitionError("qEI only supports single-objective surrogates")

        # For minimization: maximize -Y, so best_f = -min(Y) and we flip via a sign objective.
        if self.minimize:
            best_f = float(-np.min(observed_Y))
            objective = GenericMCObjective(lambda samples, X=None: -samples.squeeze(-1))
        else:
            best_f = float(np.max(observed_Y))
            objective = None

        seed = int(rng.integers(0, 2**31 - 1))
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.mc_samples]), seed=seed)
        acq = qLogExpectedImprovement(
            model=gp.model,
            best_f=best_f,
            sampler=sampler,
            objective=objective,
        )

        bounds = torch.as_tensor(space.bounds.T, dtype=torch.double)
        candidates, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=q,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            options={"seed": seed},
        )
        x = candidates.detach().cpu().numpy()
        return space.clip(x)
