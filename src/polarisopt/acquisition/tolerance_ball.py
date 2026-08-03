"""Range-aware acquisition — target-window membership probability.

v0.37. Implements the ``tolerance_ball`` and ``heaviside`` acquisitions
of Jiang, Wu, Schroeder & Webb (2026), "Range-aware Bayesian
optimization for discovering diverse designs within target property
windows", *Digital Discovery* (DOI 10.1039/d6dd00358c; preprint
arXiv:2606.11574).

Motivation
----------
polarisopt's other acquisitions optimize a scalar toward a *minimum*.
POLARIS choice-model calibration is not an optimization problem — it is
**specification satisfaction**: a parameter vector is acceptable when
every moment residual lands inside its tolerance window, and because the
objective is rank-deficient the deliverable is a *set* of acceptable θ,
not a point. :mod:`polarisopt.studies.history_matching` produces that
set but searches passively (space-filling design, filter afterwards).
This module scores candidates by the posterior probability that they
land **inside** the window, so the search targets the acceptable region
directly.

Two window geometries
---------------------
The acceptance criterion polarisopt already implements for history
matching is Vernon implausibility: ``max_j |r_j| / s_j < c``, an
**L-infinity box** in standardized residual units ``z_j = r_j / s_j``
where ``s_j = sqrt(obs_j^2 + md_j^2)``. The reference paper's tolerance
ball is an **L2 ball**, ``sum_j z_j^2 <= delta^2``. These are different
regions, so both are provided:

``norm="linf"`` (default)
    ``P[all_j |z_j| <= c] = prod_j [Phi((c - mu_j)/sigma_j)
    - Phi((-c - mu_j)/sigma_j)]``.
    Exact for independent Gaussian posteriors — no approximation — and
    matches the history-matching cutoff *exactly*, so the two are
    directly comparable on one study. This is the default because it is
    the criterion the DFW calibration actually states.

``norm="l2"``
    The paper's formulation:
    ``alpha_TB(x) = F_{K,lambda(x)}(delta^2 / h^2(x))`` with ``F`` the
    noncentral chi-square CDF, ``K`` the number of moment elements,
    ``h^2(x)`` an isotropic summary of the per-output predictive
    variance, and ``lambda(x) = sum_j mu_j^2 / h^2(x)``.

    **Accuracy caveat.** The noncentral chi-square form requires all
    per-output posterior variances to be equal; polarisopt substitutes
    their mean. Measured against 400k-draw Monte Carlo at ``K=6``,
    ``cutoff_sigma=3``, with the per-output posterior standard
    deviations spanning a ratio ``R`` (absolute error in probability):

    =====  ======================  ===============
    R      delta = 3*sqrt(K)       delta = 3
           (the default)           (tight ball)
    =====  ======================  ===============
    1      0.0000                  0.0000
    2      0.0000                  0.0097
    5      0.0112                  0.0805
    20     0.1930                  0.0181
    =====  ======================  ===============

    The error depends on both the spread and where the ball sits
    relative to the posterior — it is worst when the probability is not
    saturated near 0 or 1. A 5-20x spread across a heterogeneous moment
    set is entirely plausible, so ``l2`` is offered for fidelity to the
    reference but is **not** the default. Prefer ``linf`` unless you
    specifically want the Euclidean-ball geometry.

Batch selection
---------------
The paper's objective is *diverse* valid designs, and the DFW problem is
a degenerate manifold (Phase 6B found a 27-dimensional ridge). Taking
the top-``q`` scoring candidates would cluster them. Instead the batch
is chosen by greedy maxi-min: score a Sobol candidate pool, keep the
top ``pool_multiplier * q``, then greedily pick the point furthest (in
the unit cube) from those already selected. For ``q = 1`` this reduces
to the argmax.
"""

from __future__ import annotations

import numpy as np

from polarisopt.acquisition.base import (
    AcquisitionError,
    AcquisitionFunction,
    acquisition_registry,
)
from polarisopt.parameters import ParameterSpace
from polarisopt.surrogates.base import Surrogate
from polarisopt.utils.logging import get_logger

log = get_logger(__name__)

_ALLOWED_NORMS = ("linf", "l2")
DEFAULT_CANDIDATE_POOL = 4096


class _WindowAcquisitionBase(AcquisitionFunction):
    """Shared machinery for window-membership acquisitions."""

    def __init__(
        self,
        surrogate: Surrogate,
        *,
        minimize: bool = True,
        cutoff_sigma: float = 3.0,
        norm: str = "linf",
        tolerance: list[float] | np.ndarray | None = None,
        delta: float | None = None,
        n_candidates: int = DEFAULT_CANDIDATE_POOL,
        pool_multiplier: int = 10,
    ) -> None:
        super().__init__(surrogate, minimize=minimize)
        if norm not in _ALLOWED_NORMS:
            raise ValueError(
                f"{type(self).__name__}: norm must be one of {_ALLOWED_NORMS}, "
                f"got {norm!r}"
            )
        if not np.isfinite(cutoff_sigma) or cutoff_sigma <= 0:
            raise ValueError(
                f"cutoff_sigma must be positive finite, got {cutoff_sigma!r}"
            )
        if int(n_candidates) < 1:
            raise ValueError(f"n_candidates must be >= 1, got {n_candidates!r}")
        if int(pool_multiplier) < 1:
            raise ValueError(f"pool_multiplier must be >= 1, got {pool_multiplier!r}")
        if delta is not None and (not np.isfinite(delta) or delta <= 0):
            raise ValueError(f"delta must be positive finite or None, got {delta!r}")
        self.cutoff_sigma = float(cutoff_sigma)
        self.norm = norm
        self.delta = float(delta) if delta is not None else None
        self.n_candidates = int(n_candidates)
        self.pool_multiplier = int(pool_multiplier)
        self._tolerance: np.ndarray | None = (
            np.asarray(tolerance, dtype=float) if tolerance is not None else None
        )
        if self._tolerance is not None:
            if self._tolerance.ndim != 1:
                raise ValueError(
                    f"tolerance must be a 1-D per-moment vector, got shape "
                    f"{self._tolerance.shape}"
                )
            if not np.all(np.isfinite(self._tolerance)) or np.any(self._tolerance <= 0):
                raise ValueError(
                    "tolerance entries must all be positive and finite"
                )

    # ----- window scale -----

    def _tolerance_vector(self, m: int) -> np.ndarray:
        """Per-moment ``s_j`` used to standardize residuals.

        Explicit ``tolerance`` wins. Otherwise polarisopt reads
        ``sqrt(obs^2 + md^2)`` off the study's ``moment_set`` metric via
        :meth:`bind_metric`; if that was never called there is no
        defensible scale and we refuse rather than silently assuming 1.
        """
        if self._tolerance is not None:
            if self._tolerance.shape[0] != m:
                raise AcquisitionError(
                    f"tolerance has {self._tolerance.shape[0]} entries but the "
                    f"surrogate has {m} outputs"
                )
            return self._tolerance
        raise AcquisitionError(
            f"{type(self).__name__} needs a per-moment tolerance scale. Either "
            f"pass `tolerance: [...]` explicitly, or use a `moment_set` metric "
            f"so polarisopt can derive sqrt(obs_noise_std^2 + "
            f"model_discrepancy_std^2) from it."
        )

    def bind_metric(self, metric: object) -> None:
        """Adopt the tolerance scale from a ``moment_set`` metric.

        Called by :class:`~polarisopt.generators.acquisition.AcquisitionGenerator`
        when the study's metric exposes per-moment obs/md vectors. An
        explicit ``tolerance`` option always wins over this.
        """
        if self._tolerance is not None:
            return
        obs = getattr(metric, "obs_noise_std_vector", None)
        md = getattr(metric, "model_discrepancy_std_vector", None)
        if obs is None or md is None:
            return
        obs = np.asarray(obs, dtype=float)
        md = np.asarray(md, dtype=float)
        if not np.all(np.isfinite(md)):
            raise AcquisitionError(
                f"{type(self).__name__}: the metric has moment(s) with "
                f"model_discrepancy_std='auto' (stored as NaN), so the tolerance "
                f"window is undefined. Run `polarisopt calibrate-md` and put the "
                f"calibrated numbers in the YAML first."
            )
        scale = np.sqrt(obs**2 + md**2)
        if np.any(scale <= 0):
            bad = np.flatnonzero(scale <= 0).tolist()
            raise AcquisitionError(
                f"{type(self).__name__}: moment element(s) {bad} have zero "
                f"obs_noise_std AND model_discrepancy_std, so their tolerance "
                f"window has zero width and membership probability is degenerate."
            )
        self._tolerance = scale

    # ----- scoring -----

    def score(self, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
        """Window-membership probability for each row of ``mean``/``var``.

        Parameters
        ----------
        mean, var
            ``(n, m)`` posterior mean and variance of the moment
            residual vector. The target is the zero vector — a
            ``moment_set`` metric returns ``sim - target`` directly.

        Returns
        -------
        numpy.ndarray
            ``(n,)`` probabilities in ``[0, 1]``.
        """
        mean = np.atleast_2d(np.asarray(mean, dtype=float))
        var = np.atleast_2d(np.asarray(var, dtype=float))
        if mean.shape != var.shape:
            raise AcquisitionError(
                f"mean {mean.shape} and var {var.shape} must have the same shape"
            )
        m = mean.shape[1]
        s = self._tolerance_vector(m)
        # Standardize into tolerance units: the window is |z_j| <= cutoff_sigma.
        z_mu = mean / s
        z_sd = np.sqrt(np.maximum(var, 0.0)) / s
        # A zero-variance output is a point mass; nudge so the CDFs stay finite.
        z_sd = np.maximum(z_sd, 1e-12)
        c = self.cutoff_sigma
        if self.norm == "linf":
            from scipy.stats import norm as _norm

            # Exact for independent Gaussians: product of per-moment
            # interval probabilities. No isotropic approximation.
            per = _norm.cdf((c - z_mu) / z_sd) - _norm.cdf((-c - z_mu) / z_sd)
            return np.clip(np.prod(per, axis=1), 0.0, 1.0)
        # l2: noncentral chi-square under the isotropic-variance approximation.
        from scipy.stats import ncx2

        h2 = np.mean(z_sd**2, axis=1)
        h2 = np.maximum(h2, 1e-300)
        lam = np.sum(z_mu**2, axis=1) / h2
        delta2 = (self.delta if self.delta is not None else c * np.sqrt(m)) ** 2
        return np.clip(ncx2.cdf(delta2 / h2, df=m, nc=lam), 0.0, 1.0)

    # ----- optimization -----

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
        if not self.surrogate.is_fitted():
            raise AcquisitionError(
                f"{type(self).__name__}.optimize: surrogate is not fitted"
            )
        from scipy.stats import qmc

        sampler = qmc.Sobol(d=space.ndim, scramble=True, rng=rng)
        unit = sampler.random(n=self.n_candidates)
        bounds = space.bounds
        cand = space.clip(unit * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0])

        mean, var = self.surrogate.predict(cand)
        scores = self.score(mean, var)
        log.info(
            "%s: scored %d candidates (norm=%s, cutoff=%.2f) — "
            "best P(in window)=%.4f, %d above 0.5",
            type(self).__name__, cand.shape[0], self.norm, self.cutoff_sigma,
            float(scores.max()), int((scores > 0.5).sum()),
        )
        if q == 1:
            return cand[np.argmax(scores)][None, :]
        # Diversity: keep a high-scoring pool, then greedy maxi-min within it
        # so a batch spans the acceptable manifold instead of clustering.
        pool_size = min(cand.shape[0], max(q, self.pool_multiplier * q))
        pool_idx = np.argsort(-scores)[:pool_size]
        return _maximin_subset(cand[pool_idx], q, space)


def _maximin_subset(rows: np.ndarray, q: int, space: ParameterSpace) -> np.ndarray:
    """Greedy maxi-min subset of ``rows``, seeded at the first row.

    Distances are taken in the unit cube so parameters with different
    ranges contribute comparably. ``rows`` is assumed score-ordered, so
    seeding at index 0 anchors the batch on the best-scoring candidate.
    """
    if rows.shape[0] <= q:
        return rows
    bounds = space.bounds
    span = bounds[:, 1] - bounds[:, 0]
    span = np.where(span > 0, span, 1.0)
    unit = (rows - bounds[:, 0]) / span
    chosen = [0]
    min_d = np.linalg.norm(unit - unit[0], axis=1)
    for _ in range(q - 1):
        nxt = int(np.argmax(min_d))
        chosen.append(nxt)
        min_d = np.minimum(min_d, np.linalg.norm(unit - unit[nxt], axis=1))
    return rows[np.array(chosen)]


@acquisition_registry.register("tolerance_ball")
class ToleranceBallAcquisition(_WindowAcquisitionBase):
    """Probability that a candidate lands inside the tolerance window.

    Parameters
    ----------
    cutoff_sigma : float, optional
        Window half-width in standardized units — a candidate is
        acceptable when ``|r_j| <= cutoff_sigma * sqrt(obs_j^2 + md_j^2)``.
        Default ``3.0``, matching Pukelsheim's 3-sigma rule and
        history matching's default implausibility cutoff.
    norm : {"linf", "l2"}, optional
        Window geometry. ``"linf"`` (default) is the exact box that
        matches history matching; ``"l2"`` is the reference paper's
        Euclidean ball via a noncentral chi-square with an isotropic
        variance approximation. See the module docstring for the
        measured accuracy of that approximation.
    tolerance : list of float, optional
        Explicit per-moment ``s_j``. Overrides the value derived from a
        ``moment_set`` metric. Length must equal the surrogate's output
        dimension.
    delta : float, optional
        ``l2`` only — explicit ball radius in standardized units.
        Defaults to ``cutoff_sigma * sqrt(K)``, the radius at which a
        point with every ``|z_j| = cutoff_sigma`` sits on the sphere.
    n_candidates : int, optional
        Sobol candidate-pool size scored per call. Default 4096. The
        score is closed-form, so this is cheap relative to one POLARIS
        run.
    pool_multiplier : int, optional
        For ``q > 1``, the top ``pool_multiplier * q`` candidates form
        the pool that the diversity selection draws from. Default 10.

    Notes
    -----
    Consumes a ``moment_set`` metric with ``scalarize: none`` — the same
    input :class:`~polarisopt.studies.history_matching.HistoryMatchingStudy`
    takes — so the two searches are directly comparable on one study.

    ``minimize`` is accepted for interface conformance but ignored: the
    score is a probability that is always maximized.

    Examples
    --------
    .. code-block:: yaml

        generator:
          type: acquisition
          options:
            surrogate: { type: gp, options: {} }
            acquisition:
              type: tolerance_ball
              options:
                cutoff_sigma: 3.0
                norm: linf
    """


@acquisition_registry.register("heaviside")
class HeavisideAcquisition(_WindowAcquisitionBase):
    """Boundary-sharpened variant of :class:`ToleranceBallAcquisition`.

    ``alpha_HV(x) = (1 - w(x)) + w(x) * alpha_TB(x)`` with
    ``w(x) = 0.5 * [1 + tanh((D^2(x) - delta^2) / s)]`` where ``D^2`` is
    the squared standardized distance of the posterior mean from target.

    Candidates whose posterior mean is comfortably *inside* the window
    saturate at 1 regardless of their variance, so the search stops
    paying for further variance reduction in a region already known to
    be acceptable and pushes outward to find *more* acceptable designs.
    Candidates outside fall back to the plain membership probability.

    Parameters
    ----------
    boundary_softness : float, optional
        ``s`` in the weight above, in units of squared standardized
        distance. Smaller is a sharper transition. Default ``0.25``.
    Other parameters as :class:`ToleranceBallAcquisition`.
    """

    def __init__(self, *args, boundary_softness: float = 0.25, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not np.isfinite(boundary_softness) or boundary_softness <= 0:
            raise ValueError(
                f"boundary_softness must be positive finite, got {boundary_softness!r}"
            )
        self.boundary_softness = float(boundary_softness)

    def score(self, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
        base = super().score(mean, var)
        mean = np.atleast_2d(np.asarray(mean, dtype=float))
        s = self._tolerance_vector(mean.shape[1])
        z_mu = mean / s
        d2 = np.sum(z_mu**2, axis=1)
        m = mean.shape[1]
        if self.norm == "l2":
            delta2 = (
                self.delta if self.delta is not None else self.cutoff_sigma * np.sqrt(m)
            ) ** 2
        else:
            # The linf box's natural squared-distance scale: a point on the
            # box corner has d2 = K * cutoff^2.
            delta2 = (self.cutoff_sigma**2) * m
        w = 0.5 * (1.0 + np.tanh((d2 - delta2) / self.boundary_softness))
        return np.clip((1.0 - w) + w * base, 0.0, 1.0)
