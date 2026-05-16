"""Batch q-Expected Hypervolume Improvement for multi-objective problems."""

from __future__ import annotations

import numpy as np

try:
    import torch
    from botorch.acquisition.multi_objective.logei import (
        qLogExpectedHypervolumeImprovement,
    )
    from botorch.acquisition.multi_objective.objective import WeightedMCMultiOutputObjective
    from botorch.optim import optimize_acqf
    from botorch.sampling.normal import SobolQMCNormalSampler
    from botorch.utils.multi_objective.box_decompositions.non_dominated import (
        FastNondominatedPartitioning,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "polarisopt.acquisition.qehvi requires the [bo] extra: "
        "pip install 'polarisopt[bo]'"
    ) from exc

from polarisopt.acquisition.base import (
    AcquisitionError,
    AcquisitionFunction,
    acquisition_registry,
)
from polarisopt.parameters import ParameterSpace
from polarisopt.surrogates.gp import GPSurrogate


@acquisition_registry.register("qehvi")
class QEHVIAcquisition(AcquisitionFunction):
    """Multi-objective batch acquisition via q-EHVI.

    Wraps :class:`botorch.acquisition.multi_objective.logei.qLogExpectedHypervolumeImprovement`.
    For minimization (the default), model outputs are negated via a
    :class:`WeightedMCMultiOutputObjective` so the underlying BoTorch
    routine can maximize hypervolume against a reference point that's
    safely worse than every observed point.

    Parameters
    ----------
    surrogate : GPSurrogate
        Fitted multi-output surrogate.
    minimize : bool, optional
        ``True`` to minimize each objective (default).
    ref_point : list of float or None
        Reference point in **user-facing** objective space (i.e. before
        the internal sign flip for minimization). If omitted, defaults
        to ``min(observed) - 1`` per objective in the negated space.
    mc_samples : int, optional
        Sobol QMC posterior samples (default 128).
    num_restarts, raw_samples : int, optional
        Forwarded to :func:`botorch.optim.optimize_acqf`.

    Raises
    ------
    AcquisitionError
        If the surrogate is single-output, the reference-point shape
        disagrees with the number of objectives, or ``q < 1``.

    Examples
    --------
    >>> # Two-output minimization on a 2D space
    >>> import numpy as np
    >>> from polarisopt.parameters import Parameter, ParameterSpace
    >>> from polarisopt.surrogates.gp import GPSurrogate
    >>> rng = np.random.default_rng(0)
    >>> X = rng.uniform(size=(12, 2))
    >>> Y = np.column_stack([((X - 0.3) ** 2).sum(1), ((X - 0.7) ** 2).sum(1)])
    >>> gp = GPSurrogate(); gp.fit(X, Y)
    >>> space = ParameterSpace.from_iterable([
    ...     Parameter("x1", "a.json", 0.0, 1.0),
    ...     Parameter("x2", "a.json", 0.0, 1.0),
    ... ])
    >>> acq = QEHVIAcquisition(gp, mc_samples=16, num_restarts=2, raw_samples=16)
    >>> next_pts = acq.optimize(space, q=2, observed_Y=Y, rng=rng)
    >>> next_pts.shape
    (2, 2)
    """

    def __init__(
        self,
        surrogate,
        *,
        minimize: bool = True,
        ref_point: list[float] | None = None,
        mc_samples: int = 128,
        num_restarts: int = 10,
        raw_samples: int = 256,
    ) -> None:
        super().__init__(surrogate=surrogate, minimize=minimize)
        if not isinstance(surrogate, GPSurrogate):
            raise AcquisitionError("QEHVIAcquisition requires a GPSurrogate")
        self.ref_point_user: list[float] | None = ref_point
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
    ) -> np.ndarray:
        if q < 1:
            raise AcquisitionError(f"q must be >= 1, got {q}")
        gp = self.surrogate
        m = gp.n_objectives
        if m < 2:
            raise AcquisitionError(f"qEHVI requires >=2 objectives; surrogate has {m}")
        if observed_Y.shape[1] != m:
            raise AcquisitionError(
                f"observed_Y has {observed_Y.shape[1]} cols but surrogate has {m} objectives"
            )

        # BoTorch maximizes; for minimization, work in negated objective space.
        Y = -observed_Y if self.minimize else observed_Y
        if self.ref_point_user is not None:
            ref = np.asarray(self.ref_point_user, dtype=float)
            if ref.shape != (m,):
                raise AcquisitionError(f"ref_point must have length {m}, got {ref.shape}")
            ref_internal = -ref if self.minimize else ref
        else:
            # Default: a hair worse than the worst observed value in the negated space
            ref_internal = Y.min(axis=0) - 1.0

        Y_t = torch.as_tensor(Y, dtype=torch.double)
        ref_t = torch.as_tensor(ref_internal, dtype=torch.double)
        partitioning = FastNondominatedPartitioning(ref_point=ref_t, Y=Y_t)

        seed = int(rng.integers(0, 2**31 - 1))
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.mc_samples]), seed=seed)

        # For minimization, negate model outputs via a weighted multi-output objective.
        # WeightedMCMultiOutputObjective preserves the multi-output shape qEHVI requires.
        objective = (
            WeightedMCMultiOutputObjective(weights=torch.full((m,), -1.0, dtype=torch.double))
            if self.minimize
            else None
        )

        acq = qLogExpectedHypervolumeImprovement(
            model=gp.model,
            ref_point=ref_t,
            partitioning=partitioning,
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
