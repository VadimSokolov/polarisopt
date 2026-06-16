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


def _query_shares(db_path: Path, sql: str, *, category_col: str, count_col: str) -> dict[str, float]:
    """Run ``sql`` against the SQLite ``db_path`` and return ``{category: share}``."""
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
        - ``vector``: per-category absolute error vector (multi-objective)
    source_key:
        Key in the simulator output dict naming the SQLite path
        (default ``"demand_db"``).
    """

    def __init__(
        self,
        target_db: Path | str,
        sql: str,
        *,
        category_col: str = "category",
        count_col: str = "count",
        aggregation: str = "sum_abs",
        source_key: str = "demand_db",
    ) -> None:
        self.target_db = Path(target_db)
        self.sql = sql
        self.category_col = category_col
        self.count_col = count_col
        if aggregation not in ("sum_abs", "rmse", "vector"):
            raise ValueError(f"unknown aggregation: {aggregation!r}")
        self.aggregation = aggregation
        self.source_key = source_key
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
        sim = _query_shares(
            Path(output[self.source_key]),
            self.sql,
            category_col=self.category_col,
            count_col=self.count_col,
        )
        tgt = self._target()

        keys = sorted(set(tgt) | set(sim))
        errs = np.array([sim.get(k, 0.0) - tgt.get(k, 0.0) for k in keys], dtype=float)

        if self.aggregation == "vector":
            # Order matches sorted target keys; new sim categories appear at end.
            target_keys = sorted(tgt)
            return np.array([abs(sim.get(k, 0.0) - tgt[k]) for k in target_keys])
        if self.aggregation == "rmse":
            return np.array([float(np.sqrt(np.mean(errs**2)))])
        return np.array([float(np.sum(np.abs(errs)))])
