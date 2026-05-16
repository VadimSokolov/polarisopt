"""Gaussian Process surrogate via BoTorch.

Single-objective uses :class:`botorch.models.SingleTaskGP` with Matern-5/2 ARD.
Multi-objective wraps one ``SingleTaskGP`` per output in
:class:`botorch.models.ModelListGP`. Inputs and outputs are normalized via
BoTorch's input/outcome transforms so the user doesn't have to standardize.

This module's import side effects are guarded — if torch/botorch aren't
installed, importing :mod:`polarisopt.surrogates` succeeds silently and
``surrogate_registry.get("gp")`` raises a clear error.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:  # heavy deps live in the [bo] extra
    import torch
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import ModelListGP, SingleTaskGP
    from botorch.models.transforms.input import Normalize
    from botorch.models.transforms.outcome import Standardize
    from gpytorch.kernels import MaternKernel, ScaleKernel
    from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
except ImportError as exc:  # pragma: no cover - import-guard branch
    raise ImportError(
        "polarisopt.surrogates.gp requires the [bo] extra: "
        "pip install 'polarisopt[bo]'"
    ) from exc

from polarisopt.surrogates.base import Surrogate, SurrogateError, surrogate_registry


@surrogate_registry.register("gp")
class GPSurrogate(Surrogate):
    """BoTorch GP, single- or multi-output.

    Parameters
    ----------
    nu:
        Matern smoothness (1.5 or 2.5 typical; default 2.5).
    bounds:
        Optional ``(d, 2)`` array of feature bounds for the Normalize
        transform. If omitted, BoTorch infers from training data.
    """

    def __init__(self, *, nu: float = 2.5, bounds: list[list[float]] | None = None) -> None:
        if nu not in (0.5, 1.5, 2.5):
            raise ValueError(f"Matern nu must be one of {{0.5, 1.5, 2.5}}, got {nu}")
        self._nu = float(nu)
        self._bounds_override: np.ndarray | None = (
            np.asarray(bounds, dtype=float) if bounds is not None else None
        )
        self._model: SingleTaskGP | ModelListGP | None = None
        self._n_obj: int | None = None
        self._x_dim: int | None = None

    @property
    def n_objectives(self) -> int:
        if self._n_obj is None:
            raise SurrogateError("GPSurrogate is not fitted yet")
        return self._n_obj

    def is_fitted(self) -> bool:
        return self._model is not None

    @property
    def model(self) -> SingleTaskGP | ModelListGP:
        """The underlying BoTorch model — for acquisition consumption.

        Intentionally a leaky abstraction: acquisitions need a BoTorch
        ``Model`` to construct themselves. Only acquisition implementations
        should call this.
        """
        if self._model is None:
            raise SurrogateError("GPSurrogate is not fitted yet")
        return self._model

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        if X.ndim != 2 or Y.ndim != 2:
            raise SurrogateError(
                f"GPSurrogate.fit: X must be (n,d), Y (n,m); got {X.shape}, {Y.shape}"
            )
        if X.shape[0] != Y.shape[0]:
            raise SurrogateError(f"X and Y row counts disagree: {X.shape[0]} vs {Y.shape[0]}")
        if X.shape[0] < 2:
            raise SurrogateError("GPSurrogate.fit requires at least 2 training points")
        if not np.isfinite(X).all() or not np.isfinite(Y).all():
            raise SurrogateError("GPSurrogate.fit: X or Y contains non-finite values")

        self._x_dim = X.shape[1]
        self._n_obj = Y.shape[1]

        X_t = torch.as_tensor(X, dtype=torch.double)
        Y_t = torch.as_tensor(Y, dtype=torch.double)

        bounds_t = (
            torch.as_tensor(self._bounds_override.T, dtype=torch.double)
            if self._bounds_override is not None
            else None
        )
        input_transform = (
            Normalize(d=self._x_dim, bounds=bounds_t) if bounds_t is not None else Normalize(d=self._x_dim)
        )

        if self._n_obj == 1:
            self._model = self._make_single_gp(X_t, Y_t, input_transform)
            mll = ExactMarginalLogLikelihood(self._model.likelihood, self._model)
            fit_gpytorch_mll(mll)
        else:
            sub_models = []
            for j in range(self._n_obj):
                # Each output needs its own Normalize/Standardize state, so
                # build a fresh input transform per sub-model.
                it = (
                    Normalize(d=self._x_dim, bounds=bounds_t)
                    if bounds_t is not None
                    else Normalize(d=self._x_dim)
                )
                sub = self._make_single_gp(X_t, Y_t[:, j : j + 1], it)
                sub_models.append(sub)
            self._model = ModelListGP(*sub_models)
            mll = SumMarginalLogLikelihood(self._model.likelihood, self._model)
            fit_gpytorch_mll(mll)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._model is None:
            raise SurrogateError("GPSurrogate.predict: not fitted")
        X_t = torch.as_tensor(X, dtype=torch.double)
        with torch.no_grad():
            posterior = self._model.posterior(X_t)
            mean = posterior.mean.detach().cpu().numpy()
            var = posterior.variance.detach().cpu().numpy()
        # botorch returns (n, m); ensure shape contract
        if mean.ndim == 1:
            mean = mean[:, None]
            var = var[:, None]
        return mean, var

    # ----- helpers -----

    def _make_single_gp(
        self, X_t: torch.Tensor, Y_t: torch.Tensor, input_transform: Any
    ) -> SingleTaskGP:
        d = X_t.shape[-1]
        covar = ScaleKernel(MaternKernel(nu=self._nu, ard_num_dims=d))
        model = SingleTaskGP(
            train_X=X_t,
            train_Y=Y_t,
            covar_module=covar,
            input_transform=input_transform,
            outcome_transform=Standardize(m=Y_t.shape[-1]),
        )
        model.double()
        return model
