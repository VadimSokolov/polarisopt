"""Unit tests for v0.27 Prior types (P2) and LHS prior-mean anchor (P11)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from polarisopt.design import LHSDesign
from polarisopt.parameters import (
    BetaPrior,
    GaussianPrior,
    LogNormalPrior,
    Parameter,
    ParameterSpace,
    TruncatedNormalPrior,
    UniformPrior,
    prior_from_dict,
)
from polarisopt.parameters.space import parameter_space_from_records

# ----- GaussianPrior -----


def test_gaussian_mean_and_logprob() -> None:
    p = GaussianPrior(mean=0.5, std=0.1)
    assert p.mean == pytest.approx(0.5)
    # Standard normal density peaks at μ: ln(1/(σ √(2π))) = −ln(σ √(2π))
    assert p.log_prob(0.5) == pytest.approx(-math.log(0.1 * math.sqrt(math.tau)))
    # Symmetric around the mean
    assert p.log_prob(0.4) == pytest.approx(p.log_prob(0.6))


def test_gaussian_rejects_bad_std() -> None:
    for bad in (0.0, -1e-6, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="std"):
            GaussianPrior(mean=0.0, std=bad)


# ----- LogNormalPrior -----


def test_lognormal_mean_matches_analytic() -> None:
    p = LogNormalPrior(log_mean=-0.7, log_std=0.3)
    # E[X] = exp(log_mean + log_std² / 2)
    assert p.mean == pytest.approx(math.exp(-0.7 + 0.5 * 0.3 * 0.3))


def test_lognormal_rejects_nonpositive_x() -> None:
    p = LogNormalPrior(log_mean=0.0, log_std=1.0)
    assert p.log_prob(0.0) == -math.inf
    assert p.log_prob(-0.5) == -math.inf
    assert math.isfinite(p.log_prob(1.0))


def test_lognormal_rejects_bad_scale() -> None:
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="log_std"):
            LogNormalPrior(log_mean=0.0, log_std=bad)


# ----- TruncatedNormalPrior -----


def test_truncated_normal_mean_stays_inside_support() -> None:
    """A truncated normal centered outside the interval should have a
    mean pulled back inside — never outside the [low, high] box."""
    for loc, low, high in ((0.0, 0.5, 1.0), (5.0, -1.0, 1.0), (-3.0, 0.0, 2.0)):
        p = TruncatedNormalPrior(loc=loc, scale=0.5, low=low, high=high)
        assert low <= p.mean <= high


def test_truncated_normal_matches_untruncated_when_wide() -> None:
    """When the interval covers ±10σ around loc, the truncated mean
    should be arbitrarily close to loc."""
    p = TruncatedNormalPrior(loc=0.3, scale=0.1, low=-100.0, high=100.0)
    assert p.mean == pytest.approx(0.3, abs=1e-6)


def test_truncated_normal_out_of_support_is_neg_inf() -> None:
    p = TruncatedNormalPrior(loc=0.0, scale=1.0, low=-1.0, high=1.0)
    assert p.log_prob(-1.5) == -math.inf
    assert p.log_prob(2.0) == -math.inf
    assert math.isfinite(p.log_prob(0.0))


def test_truncated_normal_rejects_bad_bounds() -> None:
    with pytest.raises(ValueError, match="must exceed low"):
        TruncatedNormalPrior(loc=0.0, scale=1.0, low=1.0, high=1.0)
    with pytest.raises(ValueError, match="must exceed low"):
        TruncatedNormalPrior(loc=0.0, scale=1.0, low=2.0, high=1.0)


# ----- UniformPrior -----


def test_uniform_mean_is_midpoint() -> None:
    assert UniformPrior(low=-2.0, high=6.0).mean == pytest.approx(2.0)


def test_uniform_log_prob_is_flat_on_support() -> None:
    p = UniformPrior(low=0.0, high=2.0)
    assert p.log_prob(0.5) == p.log_prob(1.9)
    assert p.log_prob(0.5) == pytest.approx(-math.log(2.0))
    assert p.log_prob(-0.5) == -math.inf
    assert p.log_prob(2.5) == -math.inf


# ----- BetaPrior -----


def test_beta_mean_matches_analytic() -> None:
    assert BetaPrior(alpha=2.0, beta=3.0).mean == pytest.approx(0.4)


def test_beta_log_prob_matches_scipy_when_available() -> None:
    scipy_stats = pytest.importorskip("scipy.stats")
    p = BetaPrior(alpha=2.0, beta=5.0)
    for x in (0.1, 0.3, 0.5, 0.9):
        assert p.log_prob(x) == pytest.approx(scipy_stats.beta.logpdf(x, 2.0, 5.0), abs=1e-9)


def test_beta_out_of_support_is_neg_inf() -> None:
    p = BetaPrior(alpha=2.0, beta=2.0)
    assert p.log_prob(0.0) == -math.inf
    assert p.log_prob(1.0) == -math.inf
    assert p.log_prob(-0.1) == -math.inf


# ----- prior_from_dict factory -----


def test_prior_from_dict_gaussian() -> None:
    p = prior_from_dict({"type": "gaussian", "mean": 0.5, "std": 0.1})
    assert isinstance(p, GaussianPrior)
    assert p.mean == 0.5


def test_prior_from_dict_all_types() -> None:
    factory_specs = [
        {"type": "gaussian", "mean": 0.0, "std": 1.0},
        {"type": "log_normal", "log_mean": 0.0, "log_std": 1.0},
        {"type": "truncated_normal", "loc": 0.0, "scale": 1.0, "low": -2, "high": 2},
        {"type": "uniform", "low": 0.0, "high": 1.0},
        {"type": "beta", "alpha": 1.0, "beta": 1.0},
    ]
    for spec in factory_specs:
        assert isinstance(prior_from_dict(spec), (
            GaussianPrior, LogNormalPrior, TruncatedNormalPrior, UniformPrior, BetaPrior,
        ))


def test_prior_from_dict_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="unknown prior type"):
        prior_from_dict({"type": "cauchy", "loc": 0.0, "scale": 1.0})


def test_prior_from_dict_rejects_missing_type() -> None:
    with pytest.raises(ValueError, match="missing 'type'"):
        prior_from_dict({"mean": 0.0, "std": 1.0})


# ----- Parameter.prior wiring -----


def test_parameter_accepts_prior_field() -> None:
    p = Parameter(
        name="b_IVTT_Auto_Mean",
        file="ModeChoice.json",
        low=-0.30, high=-0.001,
        prior=GaussianPrior(mean=-0.10, std=0.035),
    )
    assert p.prior is not None
    assert p.prior.mean == pytest.approx(-0.10)


def test_parameter_defaults_prior_to_none() -> None:
    p = Parameter("x", "a.json", 0.0, 1.0)
    assert p.prior is None
    assert p.hold_at_prior_mean_if_unidentified is False


def test_parameter_rejects_prior_mean_outside_box() -> None:
    """A prior whose mean is outside the parameter's box is almost
    always a config bug — reject at construction."""
    with pytest.raises(ValueError, match="outside the parameter box"):
        Parameter(
            "x", "a.json", 0.0, 1.0,
            prior=GaussianPrior(mean=2.0, std=0.1),
        )


def test_parameter_space_from_records_wires_prior() -> None:
    space = parameter_space_from_records([
        {
            "name": "b_IVTT_Auto_Mean",
            "file": "ModeChoice.json",
            "min": -0.30, "max": -0.001,
            "prior": {"type": "gaussian", "mean": -0.10, "std": 0.035},
            "hold_at_prior_mean_if_unidentified": True,
        },
        {"name": "b_no_prior", "file": "ModeChoice.json", "min": -1.0, "max": 1.0},
    ])
    assert isinstance(space.parameters[0].prior, GaussianPrior)
    assert space.parameters[0].hold_at_prior_mean_if_unidentified is True
    assert space.parameters[1].prior is None


# ----- LHSDesign include_prior_mean_anchor (P11) -----


def _space_with_prior() -> ParameterSpace:
    return ParameterSpace.from_iterable([
        Parameter("with_prior", "a.json", 0.0, 1.0,
                  prior=GaussianPrior(mean=0.3, std=0.1)),
        Parameter("no_prior", "a.json", -5.0, 5.0),  # midpoint anchor
    ])


def test_lhs_anchor_off_by_default() -> None:
    space = _space_with_prior()
    pts = LHSDesign(n=8).generate(space, rng=np.random.default_rng(0))
    assert pts.shape == (8, 2)
    # No guarantee the anchor row appears when disabled.


def test_lhs_anchor_replaces_first_row_with_prior_mean() -> None:
    space = _space_with_prior()
    design = LHSDesign(n=8, include_prior_mean_anchor=True)
    pts = design.generate(space, rng=np.random.default_rng(0))
    # First row exactly matches (prior.mean, box midpoint).
    np.testing.assert_allclose(pts[0], [0.3, 0.0])
    # Remaining rows still fill the space (LHS not degenerate).
    assert pts.shape == (8, 2)


def test_lhs_anchor_falls_back_to_midpoint_without_priors() -> None:
    """When no parameter has a prior, the anchor is the per-parameter
    midpoint of the box — still a defensible starting point."""
    space = ParameterSpace.from_iterable([
        Parameter("x", "a.json", 0.0, 4.0),
        Parameter("y", "a.json", -10.0, 10.0),
    ])
    design = LHSDesign(n=4, include_prior_mean_anchor=True)
    pts = design.generate(space, rng=np.random.default_rng(0))
    np.testing.assert_allclose(pts[0], [2.0, 0.0])


def test_lhs_anchor_respects_int_param_type() -> None:
    """Int-typed parameters clip the anchor to the nearest integer."""
    from polarisopt.parameters import ParameterType

    space = ParameterSpace.from_iterable([
        Parameter("k", "a.json", 0, 10, ParameterType.INT,
                  prior=GaussianPrior(mean=3.7, std=1.0)),
    ])
    design = LHSDesign(n=3, include_prior_mean_anchor=True)
    pts = design.generate(space, rng=np.random.default_rng(0))
    assert pts[0, 0] == 4  # round(3.7)
