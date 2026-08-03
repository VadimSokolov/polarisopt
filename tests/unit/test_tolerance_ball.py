"""Unit tests for v0.37 tolerance_ball / heaviside acquisitions.

These are deliberately *behavioural*: scores are checked against
brute-force Monte Carlo of the same posterior, and the search is checked
for actually recovering a known acceptable window. The v0.36 review
concluded that shape-and-signature tests were what let four fabricated
SQL helpers ship, so nothing here asserts only a shape.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from polarisopt.acquisition import make_acquisition
from polarisopt.acquisition.base import AcquisitionError
from polarisopt.acquisition.tolerance_ball import (
    HeavisideAcquisition,
    ToleranceBallAcquisition,
)
from polarisopt.parameters import Parameter, ParameterSpace


class _FakeSurrogate:
    """Posterior stub: mean/var are explicit functions of X."""

    def __init__(self, mean_fn, var_fn, *, fitted: bool = True) -> None:
        self._mean_fn, self._var_fn, self._fitted = mean_fn, var_fn, fitted

    def is_fitted(self) -> bool:
        return self._fitted

    def predict(self, X: np.ndarray):
        X = np.atleast_2d(X)
        return self._mean_fn(X), self._var_fn(X)


def _space(ndim: int = 2) -> ParameterSpace:
    return ParameterSpace.from_iterable(
        [Parameter(f"x{i}", "a.json", 0.0, 1.0) for i in range(ndim)]
    )


def _acq(cls=ToleranceBallAcquisition, *, tol=None, **kw):
    return cls(_FakeSurrogate(lambda X: X, lambda X: X), tolerance=tol, **kw)


# ----- score correctness vs Monte Carlo -----


@pytest.mark.parametrize("norm", ["linf", "l2"])
def test_score_matches_monte_carlo(norm: str) -> None:
    """The closed-form window-membership probability must match a
    brute-force MC estimate of the same Gaussian posterior."""
    rng = np.random.default_rng(0)
    K = 5
    mu = rng.normal(0.0, 1.5, K)
    # Modest sd spread so the l2 isotropic approximation is in its
    # accurate regime (its degradation is asserted separately below).
    sd = rng.uniform(0.7, 1.1, K)
    acq = _acq(tol=np.ones(K), norm=norm, cutoff_sigma=3.0)

    got = float(acq.score(mu[None, :], (sd**2)[None, :])[0])

    n = 400_000
    draws = rng.normal(mu, sd, size=(n, K))
    if norm == "linf":
        mc = float(np.mean(np.all(np.abs(draws) <= 3.0, axis=1)))
    else:
        delta2 = (3.0 * np.sqrt(K)) ** 2
        mc = float(np.mean((draws**2).sum(axis=1) <= delta2))
    assert got == pytest.approx(mc, abs=0.02), f"{norm}: closed={got} mc={mc}"


def test_linf_score_is_exact_under_heteroscedasticity() -> None:
    """The linf product form makes no isotropic assumption, so it stays
    exact even when per-output sds differ by 100x — the reason it is the
    default."""
    rng = np.random.default_rng(3)
    K = 6
    mu = rng.normal(0.0, 0.5, K)
    sd = np.geomspace(0.05, 5.0, K)
    acq = _acq(tol=np.ones(K), norm="linf", cutoff_sigma=3.0)
    got = float(acq.score(mu[None, :], (sd**2)[None, :])[0])
    n = 400_000
    mc = float(np.mean(np.all(np.abs(rng.normal(mu, sd, size=(n, K))) <= 3.0, axis=1)))
    assert got == pytest.approx(mc, abs=0.005)


def test_l2_isotropic_approximation_degrades_as_documented() -> None:
    """Guard on the documented caveat: the l2 noncentral-chi2 form
    assumes equal per-output variances, so it is materially wrong under a
    large sd spread. If this ever becomes accurate, the docstring's
    accuracy table must be updated."""
    rng = np.random.default_rng(1)
    K = 6
    mu = rng.normal(0.0, 0.5, K)
    sd_flat = np.full(K, 0.5)
    # 20x spread: the docstring table records abs err 0.1930 here under
    # the default delta = 3*sqrt(K). A 5x spread errs by only ~0.011 at
    # this delta because the probability saturates near 1, which is why
    # this test probes 20x rather than 5x.
    sd_spread = np.geomspace(0.5, 10.0, K)
    acq = _acq(tol=np.ones(K), norm="l2", cutoff_sigma=3.0)
    n = 300_000
    delta2 = (3.0 * np.sqrt(K)) ** 2

    flat = float(acq.score(mu[None, :], (sd_flat**2)[None, :])[0])
    mc_flat = float(np.mean((rng.normal(mu, sd_flat, size=(n, K))**2).sum(1) <= delta2))
    assert flat == pytest.approx(mc_flat, abs=0.01), "equal variances should be accurate"

    spread = float(acq.score(mu[None, :], (sd_spread**2)[None, :])[0])
    mc_spread = float(np.mean((rng.normal(mu, sd_spread, size=(n, K))**2).sum(1) <= delta2))
    assert abs(spread - mc_spread) > 0.05, (
        f"the isotropic approximation is documented as erring by ~0.19 at a "
        f"20x sd spread (closed={spread:.4f} mc={mc_spread:.4f}); it now "
        f"appears accurate — re-measure and update the docstring table"
    )


def test_score_is_monotone_in_distance_from_target() -> None:
    """A posterior mean further from target must never score higher."""
    K = 4
    acq = _acq(tol=np.ones(K), cutoff_sigma=3.0)
    var = np.full((1, K), 0.25)
    prev = None
    for offset in (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0):
        s = float(acq.score(np.full((1, K), offset), var)[0])
        if prev is not None:
            assert s <= prev + 1e-12, f"score rose at offset {offset}"
        prev = s
    assert prev < 1e-6, "a mean 10 sigma outside the window should score ~0"


def test_score_at_target_with_tiny_variance_is_certain() -> None:
    """Dead-on target with negligible posterior variance is a sure hit."""
    K = 3
    acq = _acq(tol=np.ones(K), cutoff_sigma=3.0)
    s = float(acq.score(np.zeros((1, K)), np.full((1, K), 1e-10))[0])
    assert s == pytest.approx(1.0, abs=1e-9)


def test_tolerance_scales_the_window() -> None:
    """A residual of 2.0 is inside a window of s=1,c=3 but outside s=0.1,c=3."""
    mean, var = np.array([[2.0]]), np.array([[1e-10]])
    wide = _acq(tol=np.array([1.0]), cutoff_sigma=3.0)
    narrow = _acq(tol=np.array([0.1]), cutoff_sigma=3.0)
    assert float(wide.score(mean, var)[0]) == pytest.approx(1.0, abs=1e-6)
    assert float(narrow.score(mean, var)[0]) == pytest.approx(0.0, abs=1e-6)


# ----- heaviside -----


def test_heaviside_saturates_inside_and_matches_tb_far_outside() -> None:
    K = 4
    tb = _acq(ToleranceBallAcquisition, tol=np.ones(K), cutoff_sigma=3.0)
    hv = _acq(
        HeavisideAcquisition, tol=np.ones(K), cutoff_sigma=3.0,
        boundary_softness=0.25,
    )
    mean_in, var_in = np.zeros((1, K)), np.full((1, K), 4.0)
    assert float(tb.score(mean_in, var_in)[0]) < 0.95
    assert float(hv.score(mean_in, var_in)[0]) == pytest.approx(1.0, abs=1e-6)
    mean_out, var_out = np.full((1, K), 20.0), np.full((1, K), 0.25)
    assert float(hv.score(mean_out, var_out)[0]) == pytest.approx(
        float(tb.score(mean_out, var_out)[0]), abs=1e-6
    )


# ----- search behaviour -----


def test_optimize_finds_the_acceptable_region() -> None:
    """End-to-end: with a posterior whose residual vanishes near x=0.7,
    the acquisition must propose points there — not merely return the
    right array shape."""
    K, target = 3, 0.7

    def mean_fn(X):
        return np.repeat((X[:, [0]] - target) * 10.0, K, axis=1)

    def var_fn(X):
        return np.full((X.shape[0], K), 0.01)

    acq = ToleranceBallAcquisition(
        _FakeSurrogate(mean_fn, var_fn),
        tolerance=np.ones(K), cutoff_sigma=3.0, n_candidates=2048,
    )
    pts = acq.optimize(
        _space(2), q=5, observed_Y=np.zeros((4, K)), rng=np.random.default_rng(0),
    )
    assert pts.shape == (5, 2)
    # Window is |10*(x0-0.7)| <= 3  ->  x0 within 0.3 of target.
    assert np.all(np.abs(pts[:, 0] - target) <= 0.31), pts[:, 0]


def test_optimize_batch_is_diverse_not_clustered() -> None:
    """The paper's objective is *diverse* valid designs. A top-q batch
    would collapse onto one point; maxi-min selection must spread the
    batch across the acceptable manifold."""
    K, target = 2, 0.5

    def mean_fn(X):
        # Only x0 matters -> x1 is a free (degenerate) direction, exactly
        # the ridge structure DFW Phase 6B found.
        return np.repeat((X[:, [0]] - target) * 20.0, K, axis=1)

    def var_fn(X):
        return np.full((X.shape[0], K), 0.01)

    acq = ToleranceBallAcquisition(
        _FakeSurrogate(mean_fn, var_fn),
        tolerance=np.ones(K), cutoff_sigma=3.0, n_candidates=4096,
    )
    pts = acq.optimize(
        _space(2), q=6, observed_Y=np.zeros((4, K)), rng=np.random.default_rng(0),
    )
    assert np.all(np.abs(pts[:, 0] - target) <= 0.16)
    assert pts[:, 1].max() - pts[:, 1].min() > 0.5, (
        f"batch clustered along the degenerate direction: {pts[:, 1]}"
    )


def test_optimize_q1_returns_argmax() -> None:
    K = 2

    def mean_fn(X):
        return np.repeat((X[:, [0]] - 0.25) * 50.0, K, axis=1)

    acq = ToleranceBallAcquisition(
        _FakeSurrogate(mean_fn, lambda X: np.full((X.shape[0], K), 0.01)),
        tolerance=np.ones(K), cutoff_sigma=3.0, n_candidates=1024,
    )
    pts = acq.optimize(
        _space(2), q=1, observed_Y=np.zeros((3, K)), rng=np.random.default_rng(0),
    )
    assert pts.shape == (1, 2)
    assert abs(pts[0, 0] - 0.25) < 0.07


# ----- guardrails -----


def test_missing_tolerance_is_refused_not_assumed() -> None:
    """Without a tolerance scale there is no defensible window; the
    acquisition must refuse rather than silently assume s=1."""
    acq = ToleranceBallAcquisition(_FakeSurrogate(lambda X: X, lambda X: X))
    with pytest.raises(AcquisitionError, match="tolerance scale"):
        acq.score(np.zeros((1, 3)), np.ones((1, 3)))


def test_bind_metric_derives_tolerance_from_moment_set() -> None:
    class _M:
        obs_noise_std_vector = np.array([0.005, 0.01])
        model_discrepancy_std_vector = np.array([0.02, 0.05])

    acq = ToleranceBallAcquisition(_FakeSurrogate(lambda X: X, lambda X: X))
    acq.bind_metric(_M())
    expected = np.sqrt(np.array([0.005, 0.01]) ** 2 + np.array([0.02, 0.05]) ** 2)
    np.testing.assert_allclose(acq._tolerance, expected)


def test_bind_metric_rejects_auto_md() -> None:
    """model_discrepancy_std='auto' is NaN — the window would be
    undefined, so this must error rather than produce NaN scores (the
    v0.36 class of silent failure)."""
    class _M:
        obs_noise_std_vector = np.array([0.005, 0.01])
        model_discrepancy_std_vector = np.array([0.02, np.nan])

    acq = ToleranceBallAcquisition(_FakeSurrogate(lambda X: X, lambda X: X))
    with pytest.raises(AcquisitionError, match="calibrate-md"):
        acq.bind_metric(_M())


def test_bind_metric_rejects_zero_width_window() -> None:
    class _M:
        obs_noise_std_vector = np.array([0.005, 0.0])
        model_discrepancy_std_vector = np.array([0.02, 0.0])

    acq = ToleranceBallAcquisition(_FakeSurrogate(lambda X: X, lambda X: X))
    with pytest.raises(AcquisitionError, match="zero"):
        acq.bind_metric(_M())


def test_explicit_tolerance_wins_over_metric() -> None:
    class _M:
        obs_noise_std_vector = np.array([1.0, 1.0])
        model_discrepancy_std_vector = np.array([1.0, 1.0])

    acq = ToleranceBallAcquisition(
        _FakeSurrogate(lambda X: X, lambda X: X), tolerance=[0.1, 0.2],
    )
    acq.bind_metric(_M())
    np.testing.assert_allclose(acq._tolerance, [0.1, 0.2])


def test_tolerance_length_must_match_output_dim() -> None:
    acq = _acq(tol=np.ones(3))
    with pytest.raises(AcquisitionError, match="has 3 entries"):
        acq.score(np.zeros((1, 5)), np.ones((1, 5)))


def test_unfitted_surrogate_is_refused() -> None:
    acq = ToleranceBallAcquisition(
        _FakeSurrogate(lambda X: X, lambda X: X, fitted=False),
        tolerance=np.ones(2),
    )
    with pytest.raises(AcquisitionError, match="not fitted"):
        acq.optimize(_space(2), q=1, observed_Y=np.zeros((2, 2)),
                     rng=np.random.default_rng(0))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"norm": "l1"}, "norm must be"),
        ({"cutoff_sigma": 0.0}, "cutoff_sigma"),
        ({"cutoff_sigma": float("nan")}, "cutoff_sigma"),
        ({"n_candidates": 0}, "n_candidates"),
        ({"pool_multiplier": 0}, "pool_multiplier"),
        ({"delta": -1.0}, "delta"),
        ({"tolerance": [[1.0, 2.0]]}, "1-D"),
        ({"tolerance": [1.0, 0.0]}, "positive"),
    ],
)
def test_construction_validation(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        ToleranceBallAcquisition(_FakeSurrogate(lambda X: X, lambda X: X), **kwargs)


def test_heaviside_rejects_bad_softness() -> None:
    with pytest.raises(ValueError, match="boundary_softness"):
        HeavisideAcquisition(
            _FakeSurrogate(lambda X: X, lambda X: X), boundary_softness=0.0,
        )


def test_registered_in_acquisition_registry() -> None:
    for name, cls in (
        ("tolerance_ball", ToleranceBallAcquisition),
        ("heaviside", HeavisideAcquisition),
    ):
        acq = make_acquisition(
            {"type": name, "options": {"tolerance": [1.0, 1.0]}},
            surrogate=_FakeSurrogate(lambda X: X, lambda X: X),
        )
        assert isinstance(acq, cls)


def test_generator_binds_metric_to_acquisition() -> None:
    """AcquisitionGenerator must hand the study's metric to acquisitions
    exposing bind_metric, so tolerance_ball works from YAML without the
    user restating obs/md by hand."""
    pytest.importorskip("torch")
    from polarisopt.generators.acquisition import AcquisitionGenerator
    from polarisopt.generators.base import GeneratorContext

    class _M:
        obs_noise_std_vector = np.array([0.05, 0.05])
        model_discrepancy_std_vector = np.array([0.1, 0.1])

    gen = AcquisitionGenerator(
        surrogate={"type": "gp", "options": {}},
        acquisition={"type": "tolerance_ball", "options": {"n_candidates": 256}},
    )
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(8, 2))
    Y = np.column_stack([X[:, 0] - 0.5, X[:, 0] - 0.5])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = gen.next(
            GeneratorContext(
                space=_space(2), X=X, Y=Y, iteration=1, rng=rng, metric=_M(),
            ),
            q=2,
        )
    # Without the metric binding this would have raised AcquisitionError.
    assert out.shape == (2, 2)


# ----- near-constant column detection (v0.37) -----


def test_near_constant_catches_floating_point_noise() -> None:
    """v0.37: every call site previously tested ``np.ptp(col) == 0.0``
    exactly, so a column varying only in the last ulp — e.g. a share
    computed from identical integer counts across designs — was treated
    as live and got a GP fitted to pure round-off."""
    from polarisopt.utils.degenerate import is_near_constant

    base = 0.37
    noisy = base + np.array([0.0, 1e-16, -2e-16, 3e-16, -1e-16])
    assert np.ptp(noisy) > 0.0, "fixture must not be exactly constant"
    assert is_near_constant(noisy)


def test_near_constant_keeps_genuinely_small_variation() -> None:
    """Relative tolerance: small absolute spread on a small-magnitude
    column is still real signal and must be kept."""
    from polarisopt.utils.degenerate import is_near_constant

    tiny = np.array([1e-8, 2e-8, 3e-8, 4e-8])   # 100% relative spread
    assert not is_near_constant(tiny)


def test_near_constant_handles_exact_and_all_zero() -> None:
    from polarisopt.utils.degenerate import is_near_constant

    assert is_near_constant(np.full(6, 2.5))
    assert is_near_constant(np.zeros(6))
    assert is_near_constant(np.array([1.0]))     # <2 points -> no variation
    assert is_near_constant(np.array([]))


def test_near_constant_mask_over_columns() -> None:
    from polarisopt.utils.degenerate import near_constant_mask

    Y = np.column_stack([
        np.linspace(0.0, 1.0, 8),                 # live
        np.full(8, 0.5),                          # exactly constant
        0.25 + np.arange(8) * 1e-17,              # ulp noise
    ])
    np.testing.assert_array_equal(near_constant_mask(Y), [False, True, True])
