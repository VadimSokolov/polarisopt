"""Single-point analytic Expected Improvement."""

from __future__ import annotations

import numpy as np

try:
    import torch
    from botorch.acquisition.analytic import LogExpectedImprovement
    from botorch.optim import optimize_acqf
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "polarisopt.acquisition.ei requires the [bo] extra: "
        "pip install 'polarisopt[bo]'"
    ) from exc

from polarisopt.acquisition.base import (
    AcquisitionError,
    AcquisitionFunction,
    acquisition_registry,
)
from polarisopt.parameters import ParameterSpace
from polarisopt.surrogates.gp import GPSurrogate


@acquisition_registry.register("ei")
class EIAcquisition(AcquisitionFunction):
    """Analytic single-point EI (q=1) for single-objective problems.

    Used for single-point sequential BO. For batch (q>1), prefer
    :class:`~polarisopt.acquisition.qei.QEIAcquisition`.
    """

    def __init__(
        self,
        surrogate,
        *,
        minimize: bool = True,
        num_restarts: int = 10,
        raw_samples: int = 256,
    ) -> None:
        super().__init__(surrogate=surrogate, minimize=minimize)
        if not isinstance(surrogate, GPSurrogate):
            raise AcquisitionError("EIAcquisition requires a GPSurrogate")
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
        if q != 1:
            raise AcquisitionError(f"EI only supports q=1; got q={q}. Use qei for batches.")
        gp = self.surrogate  # type: GPSurrogate
        if gp.n_objectives != 1:
            raise AcquisitionError("EI only supports single-objective surrogates")

        # BoTorch maximizes; for minimization we flip the sign of best_f and use
        # the model's negated posterior via maximize=False on the analytic class.
        best_f = float(np.min(observed_Y)) if self.minimize else float(np.max(observed_Y))
        acq = LogExpectedImprovement(model=gp.model, best_f=best_f, maximize=not self.minimize)

        bounds = torch.as_tensor(space.bounds.T, dtype=torch.double)
        seed = int(rng.integers(0, 2**31 - 1))
        candidates, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=1,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            sequential=False,
            options={"seed": seed},
        )
        x = candidates.detach().cpu().numpy()
        return space.clip(x)
