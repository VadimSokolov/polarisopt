"""Multi-moment metric — concatenated residual vector across N named moments.

Motivated by the DFW β-calibration design (Phase 7a). A single scalar
loss on aggregate mode shares is rank-deficient by tens of dimensions
on real POLARIS choice-model calibration problems (Phase 6 multistart:
27D ridge on 33-parameter MLE). Escaping the ridge requires the
objective to be the raw vector of moment residuals — each moment acts
as a filter on a subspace of the parameter space (POM design principle,
Grimm et al. 2005; history matching consumers use per-moment
implausibility per Vernon-Goldstein-Bower 2010).

``moment_set``:

- Reads N named moments from configured CSV targets and per-moment SQL
  against a SQLite output (default ``demand_db``).
- Concatenates per-moment residuals into a single vector.
- Optionally scalarizes the vector for downstream BO consumers that
  need a scalar objective (``scalarize`` YAML option).
- Exposes per-element ``obs_noise_std``, ``model_discrepancy_std``,
  and ``weight`` arrays as instance attributes for history-matching
  consumers (v0.31+).

Two per-moment aggregations shipped in v0.26:

- ``elementwise_residual`` (default) — ``sim_k − target_k`` per element.
- ``log_ratio_residual`` — ``log((sim_k + ε) / (target_k + ε))`` per
  element. Better for count-based moments spanning multiple decades
  (transit boarding, VMT) where absolute residuals over-weight the
  large-magnitude buckets.

Advanced aggregations (``sum_squared_residual_by_functional_class``,
etc.) are reserved for follow-ups.
"""

from __future__ import annotations

import csv
import sqlite3
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from polarisopt.metrics.base import Metric, MetricError, metric_registry

_ALLOWED_AGGREGATIONS = ("elementwise_residual", "log_ratio_residual")
_ALLOWED_SCALARIZE = ("none", "sum_squared_weighted", "max_implausibility", "mean_implausibility")

# Small floor for log_ratio_residual so log() stays finite when sim or
# target happens to hit exactly zero. Chosen conservatively small
# relative to observed POLARIS mode shares (~0.01–0.9); users can
# override per-moment via ``log_epsilon``.
_DEFAULT_LOG_EPS = 1e-9


@dataclass(slots=True)
class MomentSpec:
    """Per-moment configuration inside a ``moment_set`` metric.

    Attributes
    ----------
    name
        Human-readable identifier used in error messages and by history-
        matching consumers to slice the vector back into named moments.
    source_sql
        SELECT returning key columns + one value column. Rows are joined
        to the target CSV by matching every key column exactly.
    target
        Path to the target CSV. Must contain the same key columns as the
        SQL, plus one value column.
    target_key_cols
        Names of the key columns (order-sensitive, must match both SQL
        output and CSV headers).
    target_value_col
        Name of the value column in both SQL output and CSV.
    obs_noise_std
        Observation-noise standard deviation for this moment (applied
        elementwise). Consumed by history-matching's implausibility
        calc; does not affect the raw residual output.
    model_discrepancy_std
        Vernon-Goldstein-Bower model-discrepancy standard deviation for
        this moment. Vernon 2010 §3.5 warns that omitting this produces
        empty NROY; the elicited value goes here. Zero or absent is
        allowed but emits a UserWarning at construction (P10 partial).
    weight_per_element
        Positive scalar weight multiplied into each residual before it
        enters the output vector. Default 1.0.
    aggregation
        ``elementwise_residual`` (default) or ``log_ratio_residual``.
    log_epsilon
        Floor for ``log_ratio_residual`` when either side hits zero.
        Default 1e-9. Ignored by other aggregations.
    """

    name: str
    source_sql: str
    target: Path
    target_key_cols: tuple[str, ...]
    target_value_col: str
    obs_noise_std: float = 0.0
    model_discrepancy_std: float = 0.0
    weight_per_element: float = 1.0
    aggregation: str = "elementwise_residual"
    log_epsilon: float = _DEFAULT_LOG_EPS
    # Populated by _load_target(); ordered dict from key-tuple → target value.
    # Order determines the vector layout for this moment.
    _target_by_key: dict[tuple[str, ...], float] = field(default_factory=dict, repr=False)


@metric_registry.register("moment_set")
class MomentSetMetric(Metric):
    """Concatenate residuals across N configured moments into one vector.

    Each moment matches simulated rows to target rows by an exact key
    tuple; unmatched target keys default their simulated value to 0
    (a mode/purpose that POLARIS never sampled has 0 count, hence 0
    share), unmatched simulated keys are dropped (extra categories the
    target didn't specify aren't part of the calibration objective).

    Parameters
    ----------
    moments
        List of moment dicts. Each entry becomes a :class:`MomentSpec`.
        Required keys: ``name``, ``source_sql``, ``target``,
        ``target_key_cols``, ``target_value_col``. Optional:
        ``obs_noise_std``, ``model_discrepancy_std``,
        ``weight_per_element``, ``aggregation``, ``log_epsilon``.
    source_key
        Key in the simulator output dict naming the SQLite database
        path (default ``"demand_db"``, matching
        :class:`ChoiceShareMetric`).
    scalarize
        How to reduce the concatenated residual vector to a scalar
        for BO consumers. One of:

        - ``"none"`` (default) — return the raw vector. Use this for
          history matching or multi-objective studies.
        - ``"sum_squared_weighted"`` — return ``sum(w · r²)`` as a
          length-1 vector. Standard weighted-least-squares scalar loss.
        - ``"max_implausibility"`` — return ``max_k I_k`` where
          ``I_k = |r_k| / sqrt(σ²_obs,k + σ²_md,k)``. Requires the
          moment's obs/md std to be positive.
        - ``"mean_implausibility"`` — return ``mean_k I_k`` (same
          requirement).

    Examples
    --------
    .. code-block:: yaml

        metric:
          type: moment_set
          options:
            source_key: demand_db
            scalarize: none      # or sum_squared_weighted for BO
            moments:
              - name: mode_shares_by_purpose
                source_sql: |
                  SELECT purpose, mode,
                    COUNT(*) * 1.0 / (SELECT COUNT(*) FROM Trip WHERE purpose IS NOT NULL)
                    AS share
                  FROM Trip GROUP BY purpose, mode
                target: calibration_targets/mode_choice_targets.csv
                target_key_cols: [purpose, mode]
                target_value_col: share
                obs_noise_std: 0.005
                model_discrepancy_std: 0.02

    Attributes exposed on the fitted instance (for history-matching):
        - ``moment_names`` — tuple of moment names in vector order.
        - ``moment_slices`` — dict[str, slice] mapping name → slice of
          the raw residual vector (BEFORE scalarize).
        - ``obs_noise_std_vector`` — length-``n_raw`` array.
        - ``model_discrepancy_std_vector`` — length-``n_raw`` array.
        - ``weight_vector`` — length-``n_raw`` array.

        History-matching phases (v0.31+) consume these to compute
        per-moment implausibility regardless of scalarize.

    Raises
    ------
    ValueError
        On construction: unknown aggregation or scalarize, empty
        moments list, missing required moment field, or an
        obs_noise/discrepancy that is not a non-negative finite scalar.
    MetricError
        At :meth:`compute` time: missing source_key, unreadable SQLite,
        missing target CSV, or SQL that doesn't produce the declared
        key + value columns.
    """

    def __init__(
        self,
        *,
        moments: list[dict[str, Any]],
        source_key: str = "demand_db",
        scalarize: str = "none",
    ) -> None:
        if not moments:
            raise ValueError("moment_set: at least one moment is required")
        if scalarize not in _ALLOWED_SCALARIZE:
            raise ValueError(
                f"moment_set: unknown scalarize={scalarize!r} "
                f"(expected one of {_ALLOWED_SCALARIZE})"
            )
        self.source_key = str(source_key)
        self.scalarize = scalarize
        self.moments: list[MomentSpec] = [_build_moment_spec(m) for m in moments]
        # Force target load at construction so a bad path fails-fast at
        # study setup instead of on the first sample's compute().
        for spec in self.moments:
            _load_target(spec)
        # Raw vector length = sum of per-moment target sizes.
        n_raw = sum(len(spec._target_by_key) for spec in self.moments)
        if n_raw == 0:
            raise ValueError(
                "moment_set: every moment's target CSV loaded zero rows; "
                "check target paths and CSV contents"
            )
        self._n_raw = n_raw
        # n_objectives is what the Metric ABC returns; scalarize=none
        # returns the full vector, everything else returns length-1.
        self._n_obj = n_raw if scalarize == "none" else 1
        # Precompute the per-name output-vector slices and the metadata
        # vectors — these describe the RAW vector (before scalarize) so
        # history matching can always slice back to per-moment.
        self.moment_names: tuple[str, ...] = tuple(spec.name for spec in self.moments)
        self.moment_slices: dict[str, slice] = {}
        obs_noise = np.empty(n_raw, dtype=float)
        discrepancy = np.empty(n_raw, dtype=float)
        weights = np.empty(n_raw, dtype=float)
        cursor = 0
        for spec in self.moments:
            width = len(spec._target_by_key)
            self.moment_slices[spec.name] = slice(cursor, cursor + width)
            obs_noise[cursor : cursor + width] = spec.obs_noise_std
            discrepancy[cursor : cursor + width] = spec.model_discrepancy_std
            weights[cursor : cursor + width] = spec.weight_per_element
            cursor += width
        self.obs_noise_std_vector = obs_noise
        self.model_discrepancy_std_vector = discrepancy
        self.weight_vector = weights
        # P10 partial: warn (don't raise) when any moment omits or sets
        # model_discrepancy_std=0. Vernon 2010 §3.5: this systematically
        # produces empty NROY in history matching, so users need to know.
        zero_md = [spec.name for spec in self.moments if spec.model_discrepancy_std <= 0]
        if zero_md:
            warnings.warn(
                f"moment_set: model_discrepancy_std is 0 (or absent) for moments "
                f"{zero_md}. Vernon 2010 §3.5 warns this produces empty NROY in "
                f"history matching — set model_discrepancy_std explicitly per moment.",
                UserWarning,
                stacklevel=2,
            )
        # Implausibility scalarizations need denom > 0.
        if scalarize in ("max_implausibility", "mean_implausibility"):
            denom_sq = obs_noise**2 + discrepancy**2
            if np.any(denom_sq <= 0):
                raise ValueError(
                    f"moment_set: scalarize={scalarize!r} requires every moment to "
                    f"have positive obs_noise_std or model_discrepancy_std; got zero "
                    f"denominator for moments {zero_md}"
                )

    @property
    def n_objectives(self) -> int:
        return self._n_obj

    def compute(self, output: dict[str, Any]) -> np.ndarray:
        if self.source_key not in output:
            raise MetricError(
                f"MomentSetMetric: simulator output missing {self.source_key!r}"
            )
        db_path = Path(output[self.source_key])
        if not db_path.exists():
            raise MetricError(f"MomentSetMetric: SQLite DB not found: {db_path}")

        residuals = np.empty(self._n_raw, dtype=float)
        for spec in self.moments:
            sim_by_key = _query_moment(db_path, spec)
            sl = self.moment_slices[spec.name]
            residuals[sl] = spec.weight_per_element * _residuals_for_moment(spec, sim_by_key)
        # Note: weight is multiplied INSIDE the moment slice; the raw
        # vector already carries the weight. Scalarizations below use
        # the weighted residuals directly.
        if self.scalarize == "none":
            return residuals
        if self.scalarize == "sum_squared_weighted":
            # weight is already folded in via the per-moment multiply above;
            # r_k here is w_k * raw_residual_k, so r_k^2 = w_k^2 * raw^2.
            # This matches "sum of squared weighted residuals" which is the
            # standard WLS scalar.
            return np.array([float(np.sum(residuals**2))])
        # implausibility variants — divide by the noise+md denom.
        denom = np.sqrt(self.obs_noise_std_vector**2 + self.model_discrepancy_std_vector**2)
        # residuals already carry weight; implausibility is |r| / denom.
        impl = np.abs(residuals) / denom
        if self.scalarize == "max_implausibility":
            return np.array([float(np.max(impl))])
        # mean_implausibility
        return np.array([float(np.mean(impl))])


# ----- helpers -----


def _residuals_for_moment(
    spec: MomentSpec, sim_by_key: dict[tuple[str, ...], float]
) -> np.ndarray:
    """Compute the per-element residual vector for one moment (pre-weight)."""
    target_items = list(spec._target_by_key.items())
    if spec.aggregation == "elementwise_residual":
        return np.fromiter(
            (
                sim_by_key.get(key, 0.0) - target_val
                for key, target_val in target_items
            ),
            dtype=float,
            count=len(target_items),
        )
    # log_ratio_residual
    eps = spec.log_epsilon
    return np.fromiter(
        (
            float(np.log((sim_by_key.get(key, 0.0) + eps) / (target_val + eps)))
            for key, target_val in target_items
        ),
        dtype=float,
        count=len(target_items),
    )


def _build_moment_spec(m: dict[str, Any]) -> MomentSpec:
    for required in ("name", "source_sql", "target", "target_key_cols", "target_value_col"):
        if required not in m:
            raise ValueError(f"moment_set: moment missing required field {required!r}: {m!r}")
    keys = tuple(str(k) for k in m["target_key_cols"])
    if not keys:
        raise ValueError(f"moment_set: moment {m['name']!r} must have >=1 target_key_col")
    aggregation = str(m.get("aggregation", "elementwise_residual"))
    if aggregation not in _ALLOWED_AGGREGATIONS:
        raise ValueError(
            f"moment_set: unknown aggregation {aggregation!r} for moment "
            f"{m['name']!r} (expected one of {_ALLOWED_AGGREGATIONS})"
        )
    obs_noise = float(m.get("obs_noise_std", 0.0))
    discrepancy = float(m.get("model_discrepancy_std", 0.0))
    weight = float(m.get("weight_per_element", 1.0))
    log_eps = float(m.get("log_epsilon", _DEFAULT_LOG_EPS))
    for label, value, must_be_positive in (
        ("obs_noise_std", obs_noise, False),
        ("model_discrepancy_std", discrepancy, False),
        ("weight_per_element", weight, True),
        ("log_epsilon", log_eps, True),
    ):
        if not np.isfinite(value) or value < 0 or (must_be_positive and value <= 0):
            expected = "positive" if must_be_positive else "non-negative"
            raise ValueError(
                f"moment_set: {label} for moment {m['name']!r} must be "
                f"a {expected} finite scalar, got {value!r}"
            )
    return MomentSpec(
        name=str(m["name"]),
        source_sql=str(m["source_sql"]),
        target=Path(m["target"]),
        target_key_cols=keys,
        target_value_col=str(m["target_value_col"]),
        obs_noise_std=obs_noise,
        model_discrepancy_std=discrepancy,
        weight_per_element=weight,
        aggregation=aggregation,
        log_epsilon=log_eps,
    )


def _load_target(spec: MomentSpec) -> None:
    """Read spec.target CSV into spec._target_by_key. Fails fast on
    missing file, missing columns, non-numeric value, or duplicate keys."""
    if not spec.target.exists():
        raise MetricError(
            f"moment_set: target CSV not found for moment {spec.name!r}: "
            f"{spec.target}"
        )
    try:
        with spec.target.open(newline="") as fh:
            reader = csv.DictReader(fh)
            fieldnames = list(reader.fieldnames or [])
            for col in (*spec.target_key_cols, spec.target_value_col):
                if col not in fieldnames:
                    raise MetricError(
                        f"moment_set: target CSV for moment {spec.name!r} "
                        f"missing column {col!r}; got {fieldnames}"
                    )
            for row_idx, row in enumerate(reader, start=2):  # header is line 1
                key = tuple(str(row[k]) for k in spec.target_key_cols)
                try:
                    value = float(row[spec.target_value_col])
                except (TypeError, ValueError) as exc:
                    raise MetricError(
                        f"moment_set: moment {spec.name!r} row {row_idx} "
                        f"has non-numeric {spec.target_value_col}="
                        f"{row[spec.target_value_col]!r}"
                    ) from exc
                if key in spec._target_by_key:
                    raise MetricError(
                        f"moment_set: moment {spec.name!r} has duplicate "
                        f"key {key!r} at row {row_idx}"
                    )
                spec._target_by_key[key] = value
    except OSError as exc:
        raise MetricError(
            f"moment_set: reading target CSV for moment {spec.name!r} "
            f"failed: {exc}"
        ) from exc


def _query_moment(db_path: Path, spec: MomentSpec) -> dict[tuple[str, ...], float]:
    """Run spec.source_sql and return {key_tuple: value}."""
    try:
        conn = sqlite3.connect(str(db_path), timeout=120)
        try:
            cur = conn.cursor()
            cur.execute(spec.source_sql)
            rows = cur.fetchall()
            cols = [c[0] for c in cur.description]
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise MetricError(
            f"moment_set: SQL for moment {spec.name!r} failed on {db_path}: {exc}"
        ) from exc
    for col in (*spec.target_key_cols, spec.target_value_col):
        if col not in cols:
            raise MetricError(
                f"moment_set: moment {spec.name!r} SQL must return columns "
                f"{list((*spec.target_key_cols, spec.target_value_col))}; got {cols}"
            )
    key_indices = [cols.index(k) for k in spec.target_key_cols]
    value_index = cols.index(spec.target_value_col)
    out: dict[tuple[str, ...], float] = {}
    for row in rows:
        key = tuple(str(row[i]) for i in key_indices)
        try:
            value = float(row[value_index])
        except (TypeError, ValueError):
            # NULL AGG(...) or similar → 0.0 (no observations in this bucket).
            value = 0.0
        out[key] = value
    return out
