from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from polarisopt.surrogates import make_surrogate
from polarisopt.surrogates.base import SurrogateError
from polarisopt.surrogates.gp import GPSurrogate


def _toy_data(d: int = 2, n: int = 12, m: int = 1, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(low=0.0, high=1.0, size=(n, d))
    if m == 1:
        # quadratic
        Y = np.sum((X - 0.3) ** 2, axis=1, keepdims=True)
    else:
        Y = np.stack([np.sum((X - 0.3) ** 2, axis=1), np.sum((X - 0.7) ** 2, axis=1)], axis=1)
    return X, Y


def test_gp_fit_predict_single_obj() -> None:
    X, Y = _toy_data(m=1)
    gp = GPSurrogate()
    gp.fit(X, Y)
    assert gp.is_fitted()
    assert gp.n_objectives == 1
    mean, var = gp.predict(X)
    assert mean.shape == (X.shape[0], 1)
    assert var.shape == (X.shape[0], 1)
    # GP should roughly match training points
    np.testing.assert_allclose(mean, Y, atol=0.15)


def test_gp_fit_predict_multi_obj() -> None:
    X, Y = _toy_data(m=2, n=14)
    gp = GPSurrogate()
    gp.fit(X, Y)
    assert gp.n_objectives == 2
    mean, var = gp.predict(X[:3])
    assert mean.shape == (3, 2)
    assert var.shape == (3, 2)
    assert (var > 0).all()


def test_gp_rejects_unfitted_predict() -> None:
    gp = GPSurrogate()
    with pytest.raises(SurrogateError):
        gp.predict(np.zeros((1, 2)))


def test_gp_rejects_too_few_points() -> None:
    gp = GPSurrogate()
    with pytest.raises(SurrogateError):
        gp.fit(np.zeros((1, 2)), np.zeros((1, 1)))


def test_gp_rejects_non_finite() -> None:
    gp = GPSurrogate()
    X = np.array([[0.1, 0.2], [0.3, np.nan]])
    Y = np.array([[1.0], [2.0]])
    with pytest.raises(SurrogateError):
        gp.fit(X, Y)


def test_gp_factory() -> None:
    s = make_surrogate({"type": "gp", "options": {"nu": 2.5}})
    assert isinstance(s, GPSurrogate)


def test_gp_rejects_bad_nu() -> None:
    with pytest.raises(ValueError):
        GPSurrogate(nu=1.0)


def test_gp_rejects_bad_observation_noise() -> None:
    """Scalar-path guardrails. Vector-path guardrails are covered by
    test_gp_observation_noise_rejects_bad_vector_values."""
    with pytest.raises(ValueError):
        GPSurrogate(observation_noise=0.0)
    with pytest.raises(ValueError):
        GPSurrogate(observation_noise=-1e-3)
    with pytest.raises(ValueError):
        GPSurrogate(observation_noise=float("inf"))
    # Since v0.25 a 1-D list is accepted as a per-point noise vector.
    # A 2-D array is still rejected (ambiguous shape).
    with pytest.raises(ValueError, match="scalar or 1-D array"):
        GPSurrogate(observation_noise=np.array([[1e-4]]))
    with pytest.raises(ValueError):
        GPSurrogate(observation_noise="not a number")


def test_gp_observation_noise_uses_fixed_likelihood() -> None:
    from gpytorch.likelihoods import FixedNoiseGaussianLikelihood

    X, Y = _toy_data(m=1)
    gp = GPSurrogate(observation_noise=1e-4)
    gp.fit(X, Y)
    assert isinstance(gp.model.likelihood, FixedNoiseGaussianLikelihood)


def test_gp_no_observation_noise_uses_learned_likelihood() -> None:
    from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood

    X, Y = _toy_data(m=1)
    gp = GPSurrogate()
    gp.fit(X, Y)
    # Default path stays on the learned homoskedastic likelihood, NOT the
    # fixed-noise sibling (they both inherit from _GaussianLikelihoodBase).
    assert isinstance(gp.model.likelihood, GaussianLikelihood)
    assert not isinstance(gp.model.likelihood, FixedNoiseGaussianLikelihood)


def test_gp_observation_noise_multi_obj() -> None:
    from gpytorch.likelihoods import FixedNoiseGaussianLikelihood

    X, Y = _toy_data(m=2, n=14)
    gp = GPSurrogate(observation_noise=1e-4)
    gp.fit(X, Y)
    # ModelListGP: each sub-model gets its own FixedNoiseGaussianLikelihood.
    for sub in gp.model.models:
        assert isinstance(sub.likelihood, FixedNoiseGaussianLikelihood)
    mean, var = gp.predict(X[:3])
    assert mean.shape == (3, 2)
    assert var.shape == (3, 2)


def test_gp_observation_noise_accepts_per_point_vector() -> None:
    """v0.25 heteroscedastic path: array-valued observation_noise applies
    each entry as the per-point noise variance. BoTorch still uses
    FixedNoiseGaussianLikelihood (the same class handles both scalar
    and vector Yvar)."""
    from gpytorch.likelihoods import FixedNoiseGaussianLikelihood

    X, Y = _toy_data(m=1, n=12)
    # Two noise regimes: half low-noise, half higher-noise.
    noise_vec = np.array([1e-4] * 6 + [4e-4] * 6)
    gp = GPSurrogate(observation_noise=noise_vec)
    gp.fit(X, Y)
    assert isinstance(gp.model.likelihood, FixedNoiseGaussianLikelihood)


def test_gp_observation_noise_vector_row_count_mismatch_at_fit() -> None:
    """A per-point vector of the wrong length surfaces a clear
    SurrogateError at fit time instead of a shape crash inside BoTorch."""
    X, Y = _toy_data(m=1, n=12)
    gp = GPSurrogate(observation_noise=np.array([1e-4] * 5))  # 5 != 12
    with pytest.raises(SurrogateError, match="does not match"):
        gp.fit(X, Y)


def test_gp_observation_noise_rejects_bad_vector_values() -> None:
    """Vector entries must be positive and finite — same guardrail as
    the scalar path."""
    for bad in (
        np.array([1e-4, 0.0, 1e-4]),
        np.array([1e-4, -1e-6, 1e-4]),
        np.array([1e-4, float("nan"), 1e-4]),
        np.array([1e-4, float("inf"), 1e-4]),
        np.array([]),
    ):
        with pytest.raises(ValueError, match="observation_noise"):
            GPSurrogate(observation_noise=bad)


def test_gp_observation_noise_rejects_higher_dim_array() -> None:
    with pytest.raises(ValueError, match="scalar or 1-D array"):
        GPSurrogate(observation_noise=np.array([[1e-4, 2e-4], [3e-4, 4e-4]]))


def test_gp_observation_noise_vector_multi_obj() -> None:
    """Multi-output studies wrap ModelListGP; the per-point noise vector
    should broadcast to every sub-model's Y column."""
    from gpytorch.likelihoods import FixedNoiseGaussianLikelihood

    X, Y = _toy_data(m=2, n=14)
    noise_vec = np.linspace(1e-4, 5e-4, 14)
    gp = GPSurrogate(observation_noise=noise_vec)
    gp.fit(X, Y)
    for sub in gp.model.models:
        assert isinstance(sub.likelihood, FixedNoiseGaussianLikelihood)


def test_gp_observation_noise_vector_not_mutated_after_fit() -> None:
    """The stored vector is defensively copied at construction, so a
    caller mutating the passed-in array cannot silently change the
    surrogate's noise after the fact."""
    X, Y = _toy_data(m=1, n=8)
    original = np.array([1e-4] * 8)
    gp = GPSurrogate(observation_noise=original)
    original[0] = 999.0  # try to poison the surrogate's stored copy
    gp.fit(X, Y)  # would raise if the mutation had gone through and
    # broken a positivity/isfinite invariant (999 is fine, but the point
    # is we shouldn't be using it — the surrogate's own copy stays 1e-4).
    stored = gp._observation_noise
    assert isinstance(stored, np.ndarray)
    assert stored[0] == pytest.approx(1e-4)


def test_gp_observation_noise_via_factory() -> None:
    from gpytorch.likelihoods import FixedNoiseGaussianLikelihood

    s = make_surrogate(
        {"type": "gp", "options": {"nu": 2.5, "observation_noise": 2.79e-6}}
    )
    X, Y = _toy_data(m=1)
    s.fit(X, Y)
    assert isinstance(s.model.likelihood, FixedNoiseGaussianLikelihood)
