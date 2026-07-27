"""Choice-share metric — compares categorical shares (e.g. mode share, activity types).

Reads a POLARIS demand SQLite database and computes shares from a SQL query
against a configured target. Useful as a single scalar (KS / sum-abs-diff)
or as a vector of per-category errors (multi-objective).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from polarisopt.metrics.base import Metric, MetricError, metric_registry


def _query_counts(
    db_path: Path, sql: str, *, category_col: str, count_col: str
) -> tuple[dict[str, float], float]:
    """Run ``sql`` against the SQLite ``db_path`` and return ``({category: count}, total)``.

    Kept separate from :func:`_query_shares` so callers that need Laplace
    smoothing (e.g. ``cross_entropy`` in v0.21+) don't lose the raw counts
    to premature normalization.
    """
    if not db_path.exists():
        raise MetricError(f"SQLite DB not found: {db_path}")
    try:
        conn = sqlite3.connect(str(db_path), timeout=120)
        try:
            cur = conn.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            cols = [c[0] for c in cur.description]
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise MetricError(f"SQLite query failed on {db_path}: {exc}") from exc

    if category_col not in cols or count_col not in cols:
        raise MetricError(
            f"query must return columns {category_col!r} and {count_col!r}; got {cols}"
        )
    ci = cols.index(category_col)
    ni = cols.index(count_col)
    counts = {str(row[ci]): float(row[ni]) for row in rows}
    total = sum(counts.values())
    if total <= 0:
        raise MetricError(f"query returned zero total count: {db_path}")
    return counts, total


def _query_shares(db_path: Path, sql: str, *, category_col: str, count_col: str) -> dict[str, float]:
    """Run ``sql`` against the SQLite ``db_path`` and return ``{category: share}``."""
    counts, total = _query_counts(db_path, sql, category_col=category_col, count_col=count_col)
    return {k: v / total for k, v in counts.items()}


@metric_registry.register("choice_share")
class ChoiceShareMetric(Metric):
    """Compare simulated and target categorical shares via SQL.

    The simulator output must include a key (default ``demand_db``) pointing
    to the POLARIS demand SQLite. The configured ``sql`` is run against
    both that DB and the configured ``target_db``. The resulting per-category
    shares are then compared.

    Parameters
    ----------
    target_db:
        Path to the target SQLite database.
    sql:
        Query returning two columns (category + count). The default uses
        the conventional names; override ``category_col``/``count_col`` if
        your query produces different names.
    aggregation:
        - ``sum_abs``: scalar = sum(|sim_share - tgt_share|) per category
        - ``rmse``: scalar = sqrt(mean((sim - tgt)^2))
        - ``cross_entropy``: scalar = -sum(tgt_share * log(sim_share)); the
          standard loss for probability-vector matching. Zero-target categories
          drop out. See ``laplace_smoothing_alpha`` for how zero-sim on a
          positive-target category is handled.
        - ``kl_divergence``: scalar = sum(tgt_share * log(tgt_share / sim_share));
          cross-entropy minus the (constant) entropy of the target. Zero at
          perfect match instead of ``H(tgt)``. Same smoothing behavior as
          ``cross_entropy``.
        - ``jensen_shannon``: scalar = 0.5·KL(p || m) + 0.5·KL(q || m) where
          m = 0.5·(p + q). Bounded in ``[0, ln 2]``, symmetric, and needs no
          eps or smoothing — zero-mass categories drop out of the sum
          because m is zero exactly when both p and q are zero.
        - ``vector``: per-category absolute error vector (multi-objective)
    source_key:
        Key in the simulator output dict naming the SQLite path
        (default ``"demand_db"``).
    laplace_smoothing_alpha:
        Add-α smoothing applied to simulated shares for ``cross_entropy``
        and ``kl_divergence`` (v0.21+). When ``α > 0``, the simulated
        shares become ``(count_k + α) / (N_sim + K · α)`` — this is the
        posterior mean under a Dirichlet(α) prior on sim shares and
        removes the ``−target_k · log(eps)`` blow-up that the fixed-eps
        floor produced in v0.20 when the sim happened to report zero
        count on a target-positive category. Default ``1.0`` (add-one
        smoothing). Set to ``0`` to disable and fall back to eps-clipping
        for backwards compatibility with v0.20. Ignored by aggregations
        other than ``cross_entropy`` / ``kl_divergence``.
    eps:
        Numerical floor used only for ``cross_entropy`` / ``kl_divergence``
        **and only when** ``laplace_smoothing_alpha == 0``. Default
        ``1e-12`` for compatibility with v0.20 semantics on the opt-out
        path. Ignored by other aggregations and by the smoothed CE / KL
        path.
    """

    _SCALAR_AGGREGATIONS = (
        "sum_abs",
        "rmse",
        "cross_entropy",
        "kl_divergence",
        "jensen_shannon",
    )
    _VALID_AGGREGATIONS = (*_SCALAR_AGGREGATIONS, "vector")

    def __init__(
        self,
        target_db: Path | str,
        sql: str,
        *,
        category_col: str = "category",
        count_col: str = "count",
        aggregation: str = "sum_abs",
        source_key: str = "demand_db",
        eps: float = 1e-12,
        laplace_smoothing_alpha: float = 1.0,
    ) -> None:
        self.target_db = Path(target_db)
        self.sql = sql
        self.category_col = category_col
        self.count_col = count_col
        if aggregation not in self._VALID_AGGREGATIONS:
            raise ValueError(
                f"unknown aggregation: {aggregation!r} "
                f"(expected one of {self._VALID_AGGREGATIONS})"
            )
        self.aggregation = aggregation
        self.source_key = source_key
        if not (isinstance(eps, (int, float)) and np.isfinite(eps) and eps > 0):
            raise ValueError(f"eps must be a positive finite scalar, got {eps!r}")
        self.eps = float(eps)
        # 0.0 disables smoothing (fall back to eps-clipping). Any positive
        # finite value smooths; negatives / NaN / Inf are rejected.
        if isinstance(laplace_smoothing_alpha, bool) or not isinstance(
            laplace_smoothing_alpha, (int, float)
        ):
            raise ValueError(
                f"laplace_smoothing_alpha must be a non-negative finite scalar, "
                f"got {laplace_smoothing_alpha!r}"
            )
        if not np.isfinite(laplace_smoothing_alpha) or laplace_smoothing_alpha < 0:
            raise ValueError(
                f"laplace_smoothing_alpha must be a non-negative finite scalar, "
                f"got {laplace_smoothing_alpha!r}"
            )
        self.laplace_smoothing_alpha = float(laplace_smoothing_alpha)
        self._target_cache: dict[str, float] | None = None

    @property
    def n_objectives(self) -> int:
        if self.aggregation == "vector":
            # Number of categories known only after first query against target
            if self._target_cache is None:
                self._target_cache = _query_shares(
                    self.target_db, self.sql,
                    category_col=self.category_col, count_col=self.count_col,
                )
            return len(self._target_cache)
        return 1

    def _target(self) -> dict[str, float]:
        if self._target_cache is None:
            self._target_cache = _query_shares(
                self.target_db, self.sql,
                category_col=self.category_col, count_col=self.count_col,
            )
        return self._target_cache

    def compute(self, output: dict[str, Any]) -> np.ndarray:
        if self.source_key not in output:
            raise MetricError(
                f"ChoiceShareMetric: simulator output missing {self.source_key!r}"
            )
        # CE / KL with Laplace smoothing need raw counts; every other path
        # only needs shares. Query counts once and normalize on demand.
        sim_counts, sim_total = _query_counts(
            Path(output[self.source_key]),
            self.sql,
            category_col=self.category_col,
            count_col=self.count_col,
        )
        sim = {k: v / sim_total for k, v in sim_counts.items()}
        tgt = self._target()

        keys = sorted(set(tgt) | set(sim))
        errs = np.array([sim.get(k, 0.0) - tgt.get(k, 0.0) for k in keys], dtype=float)

        if self.aggregation == "vector":
            # Order matches sorted target keys; new sim categories appear at end.
            target_keys = sorted(tgt)
            return np.array([abs(sim.get(k, 0.0) - tgt[k]) for k in target_keys])
        if self.aggregation == "rmse":
            return np.array([float(np.sqrt(np.mean(errs**2)))])
        if self.aggregation == "jensen_shannon":
            # Bounded [0, ln 2], symmetric, no eps needed. When p_k = q_k = 0
            # the term is naturally 0 (0·log 0 = 0); when only one side is 0
            # the other contributes 0·log(m) = 0, and the positive side sees
            # m = p/2 > 0, so no divergence.
            p = np.array([tgt.get(k, 0.0) for k in keys], dtype=float)
            q = np.array([sim.get(k, 0.0) for k in keys], dtype=float)
            m = 0.5 * (p + q)
            with np.errstate(divide="ignore", invalid="ignore"):
                term_p = np.where(p > 0, p * np.log(p / m), 0.0)
                term_q = np.where(q > 0, q * np.log(q / m), 0.0)
            return np.array([float(0.5 * (term_p.sum() + term_q.sum()))])
        if self.aggregation in ("cross_entropy", "kl_divergence"):
            # Only categories with positive target contribute (0 · log x = 0).
            active_keys = [k for k in keys if tgt.get(k, 0.0) > 0.0]
            p_tgt = np.array([tgt[k] for k in active_keys], dtype=float)
            if self.laplace_smoothing_alpha > 0:
                # Dirichlet(α) posterior mean: (n_k + α) / (N + K · α).
                # K is the full number of categories the union of sim/tgt sees —
                # not just active ones — because the smoothing prior applies to
                # every mode the metric could have observed.
                alpha = self.laplace_smoothing_alpha
                k_all = len(keys)
                denom = sim_total + k_all * alpha
                p_sim = np.array(
                    [(sim_counts.get(k, 0.0) + alpha) / denom for k in active_keys],
                    dtype=float,
                )
            else:
                # v0.20 semantics: floor at eps. Kept as opt-out for
                # reproducibility of pre-v0.21 studies.
                p_sim = np.array(
                    [max(sim.get(k, 0.0), self.eps) for k in active_keys],
                    dtype=float,
                )
            if self.aggregation == "cross_entropy":
                return np.array([float(-np.sum(p_tgt * np.log(p_sim)))])
            # kl_divergence — compute as log(p) − log(q) rather than log(p/q)
            # for numerical stability when p and q span many decades.
            return np.array(
                [float(np.sum(p_tgt * (np.log(p_tgt) - np.log(p_sim))))]
            )
        return np.array([float(np.sum(np.abs(errs)))])
