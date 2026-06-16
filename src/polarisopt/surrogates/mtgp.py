"""Multi-task Gaussian-Process surrogate.

Uses BoTorch's :class:`botorch.models.KroneckerMultiTaskGP` to model
correlations *between* outputs. Use this instead of the default
``GPSurrogate`` when your objectives are correlated (e.g. link RMSE and
link MAE, or mode shares across multiple modes) — sharing information
between outputs typically improves prediction quality at small
training-set sizes.

Falls back to a plain :class:`botorch.models.SingleTaskGP` when only
one objective is present.
"""

from __future__ import annotations

import numpy as np

try:  # heavy deps live in the [bo] extra
    import torch
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import KroneckerMultiTaskGP, SingleTaskGP
    from botorch.models.transforms.input import Normalize
    from botorch.models.transforms.outcome import Standardize
    from gpytorch.kernels import MaternKernel, ScaleKernel
    from gpytorch.mlls import ExactMarginalLogLikelihood
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "polarisopt.surrogates.mtgp requires the [bo] extra: "
        "pip install 'polarisopt[bo]'"
    ) from exc

from polarisopt.surrogates.base import Surrogate, SurrogateError, surrogate_registry


@surrogate_registry.register("mtgp")
class MultiTaskGPSurrogate(Surrogate):
    """Multi-task GP sharing information across correlated outputs.

    Parameters
    ----------
    nu : {0.5, 1.5, 2.5}, optional
        Matern smoothness (default 2.5). Ignored for the multi-task
        case — :class:`KroneckerMultiTaskGP` uses its own default
        kernel internally.
    rank : int or None
        Task-covariance rank passed to ``KroneckerMultiTaskGP``. Lower
        rank → more regularized correlation structure; ``None`` means
        full rank.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.uniform(size=(10, 2))
    >>> Y = np.column_stack([(X ** 2).sum(axis=1), -(X ** 2).sum(axis=1)])
    >>> mt = MultiTaskGPSurrogate()
    >>> mt.fit(X, Y)
    >>> mean, var = mt.predict(X[:3])
    >>> mean.shape, var.shape
    ((3, 2), (3, 2))
    """

    def __init__(self, *, nu: float = 2.5, rank: int | None = None) -> None:
        if nu not in (0.5, 1.5, 2.5):
            raise ValueError(f"Matern nu must be one of {{0.5, 1.5, 2.5}}, got {nu}")
        self._nu = float(nu)
        self._rank = int(rank) if rank is not None else None
        self._model: SingleTaskGP | KroneckerMultiTaskGP | None = None
        self._n_obj: int | None = None

    @property
    def n_objectives(self) -> int:
        if self._n_obj is None:
            raise SurrogateError("MultiTaskGPSurrogate is not fitted yet")
        return self._n_obj

    def is_fitted(self) -> bool:
        return self._model is not None

    @property
    def model(self) -> SingleTaskGP | KroneckerMultiTaskGP:
        if self._model is None:
            raise SurrogateError("MultiTaskGPSurrogate is not fitted yet")
        return self._model

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        if X.ndim != 2 or Y.ndim != 2:
            raise SurrogateError(
                f"fit: X must be (n,d), Y (n,m); got {X.shape}, {Y.shape}"
            )
        if X.shape[0] != Y.shape[0]:
            raise SurrogateError(
                f"X and Y row counts disagree: {X.shape[0]} vs {Y.shape[0]}"
            )
        if X.shape[0] < 2:
            raise SurrogateError("fit requires at least 2 training points")
        if not np.isfinite(X).all() or not np.isfinite(Y).all():
            raise SurrogateError("fit: X or Y contains non-finite values")

        self._n_obj = Y.shape[1]
        d = X.shape[1]
        X_t = torch.as_tensor(X, dtype=torch.double)
        Y_t = torch.as_tensor(Y, dtype=torch.double)

        if self._n_obj == 1:
            # Degenerate case — KroneckerMultiTaskGP needs m>=2.
            covar = ScaleKernel(MaternKernel(nu=self._nu, ard_num_dims=d))
            self._model = SingleTaskGP(
                train_X=X_t,
                train_Y=Y_t,
                covar_module=covar,
                input_transform=Normalize(d=d),
                outcome_transform=Standardize(m=1),
            ).double()
            mll = ExactMarginalLogLikelihood(self._model.likelihood, self._model)
            fit_gpytorch_mll(mll)
            return

        mt = KroneckerMultiTaskGP(
            train_X=X_t,
            train_Y=Y_t,
            rank=self._rank,
            input_transform=Normalize(d=d),
            outcome_transform=Standardize(m=self._n_obj),
        ).double()
        mll = ExactMarginalLogLikelihood(mt.likelihood, mt)
        fit_gpytorch_mll(mll)
        self._model = mt

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._model is None:
            raise SurrogateError("predict: not fitted")
        X_t = torch.as_tensor(X, dtype=torch.double)
        with torch.no_grad():
            posterior = self._model.posterior(X_t)
            mean = posterior.mean.detach().cpu().numpy()
            var = posterior.variance.detach().cpu().numpy()
        if mean.ndim == 1:
            mean = mean[:, None]
            var = var[:, None]
        return mean, var
