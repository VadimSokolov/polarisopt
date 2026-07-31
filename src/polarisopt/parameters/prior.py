"""Optional prior distributions on individual :class:`Parameter` values.

Motivated by the DFW β-calibration design (P2). POLARIS mode-choice
utility has ~85 free parameters but aggregate mode shares provide only
~6 independent moments — the calibration problem is rank-deficient by
tens of dimensions. Meta-analytic literature priors (Small-Verhoef 2007
value-of-time; Wardman et al. 2016 TRA meta of 3,109 valuations) are
the empirical basis for shrinkage regularization that resolves the
ridge in a defensible way.

Priors participate:

- In MAP-style scalar objectives as `-log p(θ)` penalties (v0.31+
  history matching + BO wrappers will add these).
- In history matching as a virtual moment: implausibility
  `((θ − prior_mean) / prior_std)²` added to the max-implausibility
  computation (v0.31).
- In identifiability pre-flight (v0.29) as the "fall-back value" for
  parameters flagged as un-identified; the parameter is pinned at
  `prior.mean` and dropped from the search.
- In LHS designs as the anchor point (v0.27 / P11 — see
  ``include_prior_mean_anchor`` on :class:`LHSDesign`).

The base class exposes ``mean`` (used by all three consumers) and
``log_prob(x)`` (used by MAP / implausibility). Concrete types:
:class:`GaussianPrior`, :class:`LogNormalPrior`,
:class:`TruncatedNormalPrior`, :class:`UniformPrior`, :class:`BetaPrior`.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

_TAU = math.tau  # 2π


class Prior(ABC):
    """Abstract prior on a single scalar parameter.

    Concrete subclasses must expose:

    - ``mean`` — attribute or property, used as the anchor point in
      LHS designs (P11) and the pin value for un-identified
      parameters (P5). Not enforced by the ABC because subclasses
      that inherit from :class:`dataclasses.dataclass` collide with
      an abstract-property declaration.
    - ``std`` — property giving the distribution's standard deviation,
      or ``None`` when the prior is uninformative (flat). Used by
      history matching's ``include_prior_terms`` to build the virtual
      prior moment ``((theta - mean) / std)**2`` (v0.36+).
    - :meth:`log_prob` — density at a single value; return ``-inf``
      on out-of-support values.
    """

    @abstractmethod
    def log_prob(self, x: float) -> float:
        """Log-density at ``x``. Returns ``-inf`` on out-of-support
        values; consumers should treat that as infinite penalty."""


@dataclass(frozen=True)
class GaussianPrior(Prior):
    """N(mean, std²) — unrestricted real-valued parameters.

    Use for utility coefficients where the sign is known (bounds enforce
    that) and the meta-analytic literature provides a central estimate.

    Parameters
    ----------
    mean : float
        Prior mean (also the anchor / pin value).
    std : float
        Prior standard deviation. Must be positive and finite.
    """

    mean: float
    std: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.mean):
            raise ValueError(f"GaussianPrior.mean must be finite, got {self.mean!r}")
        if not math.isfinite(self.std) or self.std <= 0:
            raise ValueError(f"GaussianPrior.std must be positive finite, got {self.std!r}")

    def log_prob(self, x: float) -> float:
        z = (x - self.mean) / self.std
        return -0.5 * z * z - math.log(self.std * math.sqrt(_TAU))


@dataclass(frozen=True)
class LogNormalPrior(Prior):
    """Log-normal — parameters constrained to positive-valued domain.

    ``x`` must be positive; the density is defined on ``x > 0`` with
    ``log x ∼ N(log_mean, log_std²)``. Useful for cost or scale
    parameters where the sign is fixed.

    Parameters
    ----------
    log_mean : float
        Mean of log(x).
    log_std : float
        Standard deviation of log(x). Must be positive finite.
    """

    log_mean: float
    log_std: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.log_mean):
            raise ValueError(f"LogNormalPrior.log_mean must be finite, got {self.log_mean!r}")
        if not math.isfinite(self.log_std) or self.log_std <= 0:
            raise ValueError(
                f"LogNormalPrior.log_std must be positive finite, got {self.log_std!r}"
            )

    @property
    def mean(self) -> float:
        """Mean of the log-normal random variable itself
        (``exp(log_mean + 0.5·log_std²)``)."""
        return math.exp(self.log_mean + 0.5 * self.log_std * self.log_std)

    @property
    def std(self) -> float:
        """Standard deviation of the log-normal variable itself:
        ``sqrt((exp(s^2) - 1) * exp(2*m + s^2))``."""
        s2 = self.log_std * self.log_std
        return math.sqrt((math.exp(s2) - 1.0) * math.exp(2.0 * self.log_mean + s2))

    def log_prob(self, x: float) -> float:
        if x <= 0:
            return -math.inf
        z = (math.log(x) - self.log_mean) / self.log_std
        return -0.5 * z * z - math.log(x * self.log_std * math.sqrt(_TAU))


@dataclass(frozen=True)
class TruncatedNormalPrior(Prior):
    """N(loc, scale²) restricted to ``[low, high]``.

    Density is the Gaussian density divided by the mass inside the
    interval, computed via the standard-normal CDF. ``x`` outside
    ``[low, high]`` gets ``-inf``. The ``mean`` property returns the
    truncated distribution's mean (not the underlying ``loc``), so
    LHS anchoring lands inside the support.

    Naming convention follows torch/scipy: ``loc`` and ``scale`` are
    the underlying Gaussian parameters; ``mean`` is the property of
    the truncated distribution.

    Parameters
    ----------
    loc : float
        Location parameter of the underlying (untruncated) Gaussian.
    scale : float
        Scale parameter. Must be positive finite.
    low, high : float
        Support endpoints. ``high > low`` required.
    """

    loc: float
    scale: float
    low: float
    high: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.loc):
            raise ValueError(f"TruncatedNormalPrior.loc must be finite, got {self.loc!r}")
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError(
                f"TruncatedNormalPrior.scale must be positive finite, got {self.scale!r}"
            )
        if not (math.isfinite(self.low) and math.isfinite(self.high)):
            raise ValueError(
                f"TruncatedNormalPrior.low/high must be finite, got low={self.low!r} "
                f"high={self.high!r}"
            )
        if self.high <= self.low:
            raise ValueError(
                f"TruncatedNormalPrior: high ({self.high}) must exceed low ({self.low})"
            )

    @staticmethod
    def _phi(z: float) -> float:
        return math.exp(-0.5 * z * z) / math.sqrt(_TAU)

    @staticmethod
    def _cdf_standard(z: float) -> float:
        # Standard-normal CDF via erf. No scipy dependency needed here.
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))

    def _bounds_mass(self) -> float:
        a = (self.low - self.loc) / self.scale
        b = (self.high - self.loc) / self.scale
        return self._cdf_standard(b) - self._cdf_standard(a)

    @property
    def mean(self) -> float:
        # Truncated Gaussian mean: loc + scale · (φ(a) − φ(b)) / (Φ(b) − Φ(a)).
        # Falls back to (low+high)/2 when the untruncated location is far
        # outside the interval and the mass is numerically zero.
        a = (self.low - self.loc) / self.scale
        b = (self.high - self.loc) / self.scale
        mass = self._cdf_standard(b) - self._cdf_standard(a)
        if mass <= 0:
            return 0.5 * (self.low + self.high)
        return self.loc + self.scale * (self._phi(a) - self._phi(b)) / mass

    @property
    def std(self) -> float:
        """Standard deviation of the truncated distribution.

        ``scale^2 * [1 + (a*phi(a) - b*phi(b))/Z - ((phi(a)-phi(b))/Z)^2]``
        with ``Z`` the retained mass. Falls back to the underlying
        ``scale`` when the mass underflows.
        """
        a = (self.low - self.loc) / self.scale
        b = (self.high - self.loc) / self.scale
        z = self._bounds_mass()
        if z <= 0:
            return self.scale
        pa, pb = self._phi(a), self._phi(b)
        var = 1.0 + (a * pa - b * pb) / z - ((pa - pb) / z) ** 2
        return self.scale * math.sqrt(max(var, 0.0))

    def log_prob(self, x: float) -> float:
        if x < self.low or x > self.high:
            return -math.inf
        z = (x - self.loc) / self.scale
        log_phi = -0.5 * z * z - math.log(self.scale * math.sqrt(_TAU))
        mass = self._bounds_mass()
        if mass <= 0:
            return -math.inf
        return log_phi - math.log(mass)


@dataclass(frozen=True)
class UniformPrior(Prior):
    """Flat prior on ``[low, high]`` — the "no informative prior" default.

    Provided so a parameter's ``prior`` field can always be non-None
    when the caller wants uniform semantics explicitly. Semantically
    equivalent to no prior at all (its ``log_prob`` contribution is a
    constant that doesn't affect the argmin).
    """

    low: float
    high: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.low) and math.isfinite(self.high)):
            raise ValueError(
                f"UniformPrior.low/high must be finite, got low={self.low!r} high={self.high!r}"
            )
        if self.high <= self.low:
            raise ValueError(
                f"UniformPrior: high ({self.high}) must exceed low ({self.low})"
            )

    @property
    def mean(self) -> float:
        return 0.5 * (self.low + self.high)

    @property
    def std(self) -> None:
        """``None`` — a flat prior carries no information, so it must not
        contribute a virtual prior moment to history-matching
        implausibility. (The variance of a uniform is finite, but using
        it would penalise the box edges purely for being edges.)"""
        return None

    def log_prob(self, x: float) -> float:
        if x < self.low or x > self.high:
            return -math.inf
        return -math.log(self.high - self.low)


@dataclass(frozen=True)
class BetaPrior(Prior):
    """Beta(α, β) on ``[0, 1]`` — parameters that are ratios.

    Parameters
    ----------
    alpha, beta : float
        Shape parameters, both > 0.
    """

    alpha: float
    beta: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError(f"BetaPrior.alpha must be positive finite, got {self.alpha!r}")
        if not math.isfinite(self.beta) or self.beta <= 0:
            raise ValueError(f"BetaPrior.beta must be positive finite, got {self.beta!r}")

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def std(self) -> float:
        """``sqrt(a*b / ((a+b)^2 * (a+b+1)))``."""
        s = self.alpha + self.beta
        return math.sqrt(self.alpha * self.beta / (s * s * (s + 1.0)))

    def log_prob(self, x: float) -> float:
        if x <= 0 or x >= 1:
            return -math.inf
        # log-Beta density: (α-1) log x + (β-1) log(1-x) - log B(α, β)
        # where B(α, β) = Γ(α)Γ(β)/Γ(α+β) → log B = lgamma(α) + lgamma(β) - lgamma(α+β)
        log_beta_fn = math.lgamma(self.alpha) + math.lgamma(self.beta) - math.lgamma(
            self.alpha + self.beta
        )
        return (
            (self.alpha - 1) * math.log(x)
            + (self.beta - 1) * math.log(1 - x)
            - log_beta_fn
        )


_PRIOR_TYPES: dict[str, type[Prior]] = {
    "gaussian": GaussianPrior,
    "log_normal": LogNormalPrior,
    "truncated_normal": TruncatedNormalPrior,
    "uniform": UniformPrior,
    "beta": BetaPrior,
}


def prior_from_dict(spec: dict[str, Any]) -> Prior:
    """Build a :class:`Prior` from a YAML-style dict.

    Every prior spec must include ``type`` (one of ``gaussian``,
    ``log_normal``, ``truncated_normal``, ``uniform``, ``beta``).
    Remaining keys are the constructor kwargs of the matching class.
    """
    if "type" not in spec:
        raise ValueError(f"prior spec missing 'type': {spec!r}")
    kind = str(spec["type"]).strip().lower()
    if kind not in _PRIOR_TYPES:
        raise ValueError(
            f"unknown prior type {kind!r} (expected one of {sorted(_PRIOR_TYPES)})"
        )
    cls = _PRIOR_TYPES[kind]
    kwargs = {k: v for k, v in spec.items() if k != "type"}
    return cls(**kwargs)
