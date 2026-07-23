"""Noise-aware Monte-Carlo batch Expected Improvement (q-Log-NEI).

For studies where the objective is a noisy realisation of an unknown
mean (POLARIS calibration at low ``population_scale_factor``,
stochastic simulators generally). Standard ``qei`` treats the observed
value as ground truth, so a favorable-seed outlier gets picked as the
incumbent and BO chases noise. ``qlognei`` (BoTorch's
``qLogNoisyExpectedImprovement``) instead computes EI relative to the
posterior over the previously-evaluated points, treating each
observation as a draw from a noisy distribution.

When to use
-----------
Any study where the DFW-style diagnostics show:

- Repeat evaluations at the same input give visibly different metric values
- Best-so-far chase never converges: each iter picks a new "winner" that
  turns out to be a lucky seed
- Sobol / repeat-sampling analysis reports a measurable noise std

For deterministic simulators (POLARIS at high pop scale factor, or the
Branin benchmark), ``qei`` remains the right choice — no advantage
from noise modelling if there's no noise.

References
----------
- BoTorch docs: :class:`botorch.acquisition.logei.qLogNoisyExpectedImprovement`
- Balandat et al., *BoTorch: A Framework for Efficient Monte-Carlo
  Bayesian Optimization*, NeurIPS 2020.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    from botorch.acquisition.logei import qLogNoisyExpectedImprovement
    from botorch.acquisition.objective import GenericMCObjective
    from botorch.optim import optimize_acqf
    from botorch.sampling.normal import SobolQMCNormalSampler
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "polarisopt.acquisition.qlognei requires the [bo] extra: "
        "pip install 'polarisopt[bo]'"
    ) from exc

from polarisopt.acquisition.base import (
    AcquisitionError,
    AcquisitionFunction,
    acquisition_registry,
)
from polarisopt.parameters import ParameterSpace
from polarisopt.surrogates.gp import GPSurrogate


@acquisition_registry.register("qlognei")
class QLogNEIAcquisition(AcquisitionFunction):
    """Batch noise-aware Expected Improvement for single-objective problems.

    Parameters
    ----------
    mc_samples:
        Number of Sobol QMC posterior samples (default 128).
    num_restarts, raw_samples:
        Passed to :func:`botorch.optim.optimize_acqf`.
    prune_baseline:
        If True (BoTorch default), prune baseline points that are unlikely
        to be the incumbent to speed up optimization. Set False if you
        want to keep every observation in the baseline (e.g. for
        debugging or small-n studies where every point matters).
    cache_root:
        BoTorch's cache-root optimization for repeated NEI evaluations.
        Default True.
    """

    def __init__(
        self,
        surrogate,
        *,
        minimize: bool = True,
        mc_samples: int = 128,
        num_restarts: int = 10,
        raw_samples: int = 256,
        prune_baseline: bool = True,
        cache_root: bool = True,
    ) -> None:
        super().__init__(surrogate=surrogate, minimize=minimize)
        if not isinstance(surrogate, GPSurrogate):
            raise AcquisitionError("QLogNEIAcquisition requires a GPSurrogate")
        self.mc_samples = int(mc_samples)
        self.num_restarts = int(num_restarts)
        self.raw_samples = int(raw_samples)
        self.prune_baseline = bool(prune_baseline)
        self.cache_root = bool(cache_root)

    def optimize(
        self,
        space: ParameterSpace,
        *,
        q: int,
        observed_Y: np.ndarray,
        rng: np.random.Generator,
        observed_X: np.ndarray | None = None,
    ) -> np.ndarray:
        if q < 1:
            raise AcquisitionError(f"q must be >= 1, got {q}")
        if observed_X is None:
            raise AcquisitionError(
                "qlognei requires observed_X (the (n, ndim) matrix of "
                "previously-evaluated input vectors) as X_baseline. Pre-v0.18 "
                "callers of Acquisition.optimize() only passed observed_Y — "
                "make sure your AcquisitionGenerator (or custom driver) is "
                "forwarding observed_X too."
            )
        if observed_X.shape[0] != observed_Y.shape[0]:
            raise AcquisitionError(
                f"observed_X ({observed_X.shape}) and observed_Y "
                f"({observed_Y.shape}) row counts disagree",
            )
        gp = self.surrogate
        if gp.n_objectives != 1:
            raise AcquisitionError("qlognei only supports single-objective surrogates")

        # For minimization: flip signs via the GenericMCObjective so BoTorch's
        # (implicitly-maximizing) NEI chases the min. Same trick as qei.
        objective = (
            GenericMCObjective(lambda samples, X=None: -samples.squeeze(-1))
            if self.minimize
            else None
        )
        seed = int(rng.integers(0, 2**31 - 1))
        sampler = SobolQMCNormalSampler(
            sample_shape=torch.Size([self.mc_samples]), seed=seed,
        )
        X_baseline = torch.as_tensor(observed_X, dtype=torch.double)
        acq = qLogNoisyExpectedImprovement(
            model=gp.model,
            X_baseline=X_baseline,
            sampler=sampler,
            objective=objective,
            prune_baseline=self.prune_baseline,
            cache_root=self.cache_root,
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
