"""Latin Hypercube Sampling via ``scipy.stats.qmc.LatinHypercube``."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import qmc

from polarisopt.design.base import Design, design_registry
from polarisopt.parameters import ParameterSpace
from polarisopt.utils.logging import get_logger

log = get_logger(__name__)


@design_registry.register("lhs")
class LHSDesign(Design):
    """Latin Hypercube design via :class:`scipy.stats.qmc.LatinHypercube`.

    Parameters
    ----------
    n : int
        Number of sample points (must be positive).
    scramble : bool, optional
        Whether to scramble the LHS (default ``True``).
    include_prior_mean_anchor : bool, optional
        v0.27+ (P11). When True, the first row of the returned batch
        is replaced with the prior-mean anchor vector (per-parameter
        ``prior.mean`` where a prior is set; midpoint of the parameter
        box otherwise). Ensures wave-1 evaluates a defensible starting
        θ regardless of LHS randomness. Requires ``n >= 1``. Default
        ``False`` — existing behavior unchanged.

    Raises
    ------
    ValueError
        If ``n <= 0``.

    Examples
    --------
    >>> import numpy as np
    >>> from polarisopt.parameters import Parameter, ParameterSpace
    >>> space = ParameterSpace.from_iterable([
    ...     Parameter("x", "a.json", 0.0, 1.0),
    ...     Parameter("y", "a.json", -1.0, 1.0),
    ... ])
    >>> design = LHSDesign(n=8)
    >>> pts = design.generate(space, rng=np.random.default_rng(0))
    >>> pts.shape
    (8, 2)
    """

    def __init__(
        self,
        n: int,
        *,
        scramble: bool = True,
        include_prior_mean_anchor: bool = False,
        analytical_prefilter: dict[str, Any] | None = None,
    ) -> None:
        if n <= 0:
            raise ValueError(f"LHSDesign: n must be positive, got {n}")
        self.n = int(n)
        self.scramble = bool(scramble)
        self.include_prior_mean_anchor = bool(include_prior_mean_anchor)
        self.analytical_prefilter: dict[str, Any] | None = (
            _validate_prefilter(analytical_prefilter)
            if analytical_prefilter is not None
            else None
        )

    def generate(self, space: ParameterSpace, *, rng: np.random.Generator) -> np.ndarray:
        if self.analytical_prefilter is not None:
            pts = self._generate_with_prefilter(space, rng=rng)
        else:
            pts = self._generate_raw_lhs(space, rng=rng, n=self.n)
        if self.include_prior_mean_anchor:
            # Replace the first row with the prior-mean anchor. Per-parameter:
            # use prior.mean if a prior is set, midpoint of the box otherwise.
            anchor = space.clip(np.array([
                p.prior.mean if p.prior is not None else 0.5 * (p.low + p.high)
                for p in space.parameters
            ], dtype=float))
            # v0.36: score the anchor through the prefilter before
            # substituting it. Previously the anchor was written in
            # unconditionally AFTER filtering, so (a) a θ the analytical
            # screen had already rejected entered the design and burned a
            # full POLARIS run, and (b) it overwrote row 0 — which the old
            # score-sort made the single best-scoring survivor.
            if self.analytical_prefilter is not None and not self._anchor_passes_prefilter(
                anchor, space,
            ):
                log.warning(
                    "analytical_prefilter rejected the prior-mean anchor %s — "
                    "keeping the filtered design instead of substituting it. "
                    "Either the priors disagree with the analytical model or "
                    "the reject threshold is too tight.",
                    dict(zip(space.names, anchor.tolist(), strict=True)),
                )
                return pts
            pts[0] = anchor
        return pts

    def _anchor_passes_prefilter(
        self, anchor: np.ndarray, space: ParameterSpace,
    ) -> bool:
        """Run the prior-mean anchor through the configured prefilter."""
        cfg = self.analytical_prefilter
        assert cfg is not None
        callable_fn = _load_callable(cfg["module"], cfg["function"])
        threshold = cfg.get("reject_if_max_share_dev_gt")
        theta = {p.name: float(v) for p, v in zip(space.parameters, anchor, strict=True)}
        try:
            result = callable_fn(theta)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "analytical_prefilter: callable raised while scoring the "
                "prior-mean anchor: %s — treating anchor as rejected", exc,
            )
            return False
        if not isinstance(result, dict) or result.get("reject"):
            return False
        score = result.get("max_share_dev")
        return not (
            threshold is not None
            and score is not None
            and float(score) > float(threshold)
        )

    def _generate_raw_lhs(
        self, space: ParameterSpace, *, rng: np.random.Generator, n: int,
    ) -> np.ndarray:
        sampler = qmc.LatinHypercube(d=space.ndim, scramble=self.scramble, rng=rng)
        unit = sampler.random(n=n)
        bounds = space.bounds
        scaled = unit * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        return space.clip(scaled)

    def _generate_with_prefilter(
        self, space: ParameterSpace, *, rng: np.random.Generator,
    ) -> np.ndarray:
        """v0.33 P1: draw ``oversample_factor × n`` candidates, score
        each with the user callable, keep the best ``n`` survivors.

        The callable is loaded from ``{module, function}`` in the
        prefilter config. It receives one theta dict per call and
        returns either:

        - ``{"reject": True, "reason": "..."}`` — hard reject; or
        - ``{"max_share_dev": float, ...}`` — soft score. When the
          config sets ``reject_if_max_share_dev_gt``, points above
          the threshold drop. Any additional keys are logged and
          ignored.

        The score is used **only as a filter**, never as a ranking.
        Survivors are thinned back to ``n`` by re-stratifying over the
        parameter box (see :func:`_stratified_thin`), which preserves
        the space-filling property the LHS exists to provide.

        Changed in v0.36: this previously sorted survivors by score and
        took the best ``n``, which collapsed the design onto the
        feasibility boundary — a measured run put all 10 returned points
        inside 5% of the x-range of a unit box. For history matching
        that is a methodological failure: the emulator only ever sees
        the region the analytical proxy likes best, so the proxy
        silently determines the NROY instead of pre-screening it.

        If survivors < n, warn and return them anyway — a wave with too
        few points is better than a crash.
        """
        cfg = self.analytical_prefilter
        assert cfg is not None
        callable_fn = _load_callable(cfg["module"], cfg["function"])
        oversample = int(cfg.get("oversample_factor", 20))
        threshold = cfg.get("reject_if_max_share_dev_gt")
        n_draw = self.n * max(oversample, 1)
        candidates = self._generate_raw_lhs(space, rng=rng, n=n_draw)

        surviving: list[tuple[np.ndarray, float | None]] = []
        n_hard_rejected = 0
        n_soft_rejected = 0
        for row in candidates:
            theta = {p.name: float(v) for p, v in zip(space.parameters, row, strict=True)}
            try:
                result = callable_fn(theta)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "analytical_prefilter: callable raised on theta=%s: %s — treating as rejected",
                    theta, exc,
                )
                n_hard_rejected += 1
                continue
            if not isinstance(result, dict):
                log.warning(
                    "analytical_prefilter: callable returned %s (expected dict); "
                    "treating as rejected", type(result).__name__,
                )
                n_hard_rejected += 1
                continue
            if result.get("reject"):
                n_hard_rejected += 1
                continue
            score = result.get("max_share_dev")
            if threshold is not None and score is not None and float(score) > float(threshold):
                n_soft_rejected += 1
                continue
            surviving.append((row, float(score) if score is not None else None))

        log.info(
            "analytical_prefilter: drew %d, kept %d (%d hard rejects, %d over threshold)",
            n_draw, len(surviving), n_hard_rejected, n_soft_rejected,
        )
        if not surviving:
            raise ValueError(
                f"analytical_prefilter rejected ALL {n_draw} candidates. "
                f"Widen the parameter box, raise reject_if_max_share_dev_gt "
                f"(currently {threshold!r}), or check the callable."
            )
        if len(surviving) < self.n:
            log.warning(
                "analytical_prefilter: only %d survivors < requested n=%d. "
                "Returning survivors as-is — consider raising oversample_factor "
                "(currently %d) or widening the reject threshold.",
                len(surviving), self.n, oversample,
            )
            return np.stack([row for row, _ in surviving])
        survivor_rows = np.stack([row for row, _ in surviving])
        return _stratified_thin(survivor_rows, self.n, space, rng)


def _stratified_thin(
    rows: np.ndarray, n: int, space: ParameterSpace, rng: np.random.Generator,
) -> np.ndarray:
    """Thin ``rows`` down to ``n`` while preserving space-filling coverage.

    Greedy maxi-min: seed with a random survivor, then repeatedly take
    the candidate whose minimum distance to the already-chosen set is
    largest. Distances are computed in the unit cube (each dimension
    normalized by its parameter range) so dimensions with wildly
    different scales contribute equally.

    This is the correct way to reduce a filtered candidate pool to a
    design: it keeps the survivors spread across the feasible region
    instead of clustering them wherever an auxiliary score happened to
    be lowest.
    """
    if rows.shape[0] <= n:
        return rows
    bounds = space.bounds
    span = bounds[:, 1] - bounds[:, 0]
    span = np.where(span > 0, span, 1.0)
    unit = (rows - bounds[:, 0]) / span

    chosen = [int(rng.integers(rows.shape[0]))]
    # Running min-distance from every candidate to the chosen set.
    min_d = np.linalg.norm(unit - unit[chosen[0]], axis=1)
    for _ in range(n - 1):
        nxt = int(np.argmax(min_d))
        chosen.append(nxt)
        min_d = np.minimum(min_d, np.linalg.norm(unit - unit[nxt], axis=1))
    return rows[np.array(chosen)]


def _validate_prefilter(cfg: dict[str, Any]) -> dict[str, Any]:
    """Enforce the analytical_prefilter YAML shape at construction."""
    if not isinstance(cfg, dict):
        raise ValueError(
            f"analytical_prefilter must be a dict, got {type(cfg).__name__}"
        )
    for required in ("module", "function"):
        if required not in cfg:
            raise ValueError(
                f"analytical_prefilter: missing required key {required!r}"
            )
    module_path = Path(cfg["module"])
    if not module_path.exists():
        raise ValueError(
            f"analytical_prefilter: module file does not exist: {module_path}"
        )
    oversample = int(cfg.get("oversample_factor", 20))
    if oversample < 1:
        raise ValueError(
            f"analytical_prefilter.oversample_factor must be >= 1, got {oversample}"
        )
    threshold = cfg.get("reject_if_max_share_dev_gt")
    if threshold is not None and (
        not isinstance(threshold, (int, float)) or float(threshold) <= 0
    ):
        raise ValueError(
            f"analytical_prefilter.reject_if_max_share_dev_gt must be positive "
            f"or absent, got {threshold!r}"
        )
    # Normalize to a shallow copy so the caller's dict can't be mutated.
    return dict(cfg)


def _load_callable(module_path: str, function_name: str) -> Callable[[dict[str, float]], Any]:
    """Import ``function_name`` from the Python file at ``module_path``."""
    spec = importlib.util.spec_from_file_location(
        f"_polarisopt_prefilter_{Path(module_path).stem}", module_path,
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"analytical_prefilter: cannot load spec for {module_path!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, function_name, None)
    if fn is None or not callable(fn):
        raise ValueError(
            f"analytical_prefilter: {module_path}::{function_name} is not a callable"
        )
    return fn
