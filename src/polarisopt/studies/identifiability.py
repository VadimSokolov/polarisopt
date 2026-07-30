"""Identifiability pre-flight — per-moment Sobol on a moment_set study.

v0.29 P5. Extends v0.24 :func:`polarisopt.studies.sensitivity.run_sensitivity_analysis`
to moment-vector metrics: for each moment column in the metric output,
fit a GP and compute Sobol first-order + total indices, then classify
each parameter as *identified* or *unidentified* based on its maximum
first-order Sobol index across all moments (Vernon 2010 §3.5
identification rule).

The output shape is:
- Per parameter: max_first_order_across_moments, max_total_across_moments,
  is_identified (bool), and (if unidentified and prior is set) the
  suggested pin value = prior.mean.
- Per (parameter, moment): the raw first-order and total indices.

Callable from the CLI (``polarisopt identifiability <config.yaml>``)
and from Python (:func:`run_identifiability_analysis`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from polarisopt.metrics.moment_set import MomentSetMetric
from polarisopt.parameters import ParameterSpace
from polarisopt.samples.sample import SampleStatus
from polarisopt.samples.store import SampleStore

DEFAULT_N_SOBOL = 8192
DEFAULT_IDENTIFICATION_THRESHOLD = 0.05


class IdentifiabilityError(RuntimeError):
    """Raised when the store / space / metric can't support the analysis."""


@dataclass(slots=True)
class ParameterIdentifiability:
    name: str
    max_first_order: float
    max_total: float
    is_identified: bool
    suggested_pin: float | None  # prior.mean if unidentified AND prior set
    per_moment_first_order: dict[str, float] = field(default_factory=dict)
    per_moment_total: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class IdentifiabilityReport:
    parameters: list[ParameterIdentifiability]
    moment_names: tuple[str, ...]
    n_train: int
    n_sobol: int
    threshold: float

    def identified(self) -> list[str]:
        return [p.name for p in self.parameters if p.is_identified]

    def unidentified(self) -> list[str]:
        return [p.name for p in self.parameters if not p.is_identified]


def _extract_x_and_moment_matrix(
    store: SampleStore, space: ParameterSpace, *, phase: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pull FINISHED (X, Y_vector) pairs where Y is the raw moment
    residual vector (may be length > 1)."""
    samples = store.list(phase=phase, status=SampleStatus.FINISHED)
    finished = [s for s in samples if s.metric is not None and s.inputs is not None]
    if not finished:
        raise IdentifiabilityError(
            "no FINISHED samples with metric available"
            + (f" for phase={phase!r}" if phase else "")
        )
    # All rows must have the same-length metric vector; otherwise the
    # metric changed between samples and no per-moment fit is possible.
    widths = {np.asarray(s.metric).size for s in finished}
    if len(widths) != 1:
        raise IdentifiabilityError(
            f"stored samples have inconsistent metric widths {sorted(widths)}; "
            "identifiability requires a fixed moment_set output length"
        )
    X = np.stack([s.inputs for s in finished])
    Y = np.stack([np.asarray(s.metric).reshape(-1) for s in finished])
    if X.shape[1] != space.ndim:
        raise IdentifiabilityError(
            f"stored inputs have ndim={X.shape[1]} but ParameterSpace has "
            f"ndim={space.ndim}"
        )
    if Y.shape[1] < 2:
        raise IdentifiabilityError(
            "identifiability requires a multi-moment metric (Y width >= 2). "
            f"Got width={Y.shape[1]}. Use `polarisopt sensitivity` instead "
            "for single-output metrics."
        )
    return X, Y


def _sobol_per_column(
    X: np.ndarray, Y: np.ndarray, space: ParameterSpace, n_sobol: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one GP per Y column and compute (S1, ST) as ``(ndim, m)`` arrays."""
    from SALib.analyze import sobol as sobol_analyze
    from SALib.sample import sobol as sobol_sample

    from polarisopt.surrogates.gp import GPSurrogate

    problem = {
        "num_vars": space.ndim,
        "names": list(space.names),
        "bounds": space.bounds.tolist(),
    }
    param_values = sobol_sample.sample(problem, n_sobol)
    m = Y.shape[1]
    s1 = np.zeros((space.ndim, m), dtype=float)
    st = np.zeros((space.ndim, m), dtype=float)
    for j in range(m):
        y_col = Y[:, j : j + 1]
        # Skip degenerate columns (identical values across samples) — no
        # signal, Sobol undefined; the parameter contributes 0 here.
        if float(np.ptp(y_col)) == 0.0:
            continue
        gp = GPSurrogate()
        gp.fit(X, y_col)
        y_pred, _ = gp.predict(param_values)
        Si = sobol_analyze.analyze(problem, np.asarray(y_pred).reshape(-1), print_to_console=False)
        s1[:, j] = np.asarray(Si["S1"], dtype=float)
        st[:, j] = np.asarray(Si["ST"], dtype=float)
    return s1, st


def run_identifiability_analysis(
    store: SampleStore,
    space: ParameterSpace,
    *,
    metric: MomentSetMetric,
    phase: str | None = None,
    n_sobol: int = DEFAULT_N_SOBOL,
    threshold: float = DEFAULT_IDENTIFICATION_THRESHOLD,
) -> IdentifiabilityReport:
    """Fit per-moment GPs on the store's FINISHED samples, Sobol each,
    and classify parameters.

    A parameter is *identified* iff its maximum first-order Sobol index
    across all moments is at or above ``threshold`` (Vernon 2010's rule).

    Parameters
    ----------
    store
        Opened SampleStore for the study.
    space
        The study's ParameterSpace. Column order of stored X must match.
    metric
        The moment_set metric — used to slice Y into named moments so the
        report can attribute Sobol indices per (parameter, moment name).
    phase
        Optional phase filter.
    n_sobol
        Number of Sobol samples for SALib (default 8192).
    threshold
        First-order Sobol cutoff for "identified". Default 0.05 (Vernon).

    Returns
    -------
    IdentifiabilityReport

    Raises
    ------
    IdentifiabilityError
        Empty store, inconsistent metric widths, shape mismatch against
        space, or a scalar metric passed in.
    """
    X, Y = _extract_x_and_moment_matrix(store, space, phase=phase)
    # Sanity: metric.n_objectives (after scalarize) may be 1 even when
    # the raw moment vector is wider. Identifiability needs the RAW
    # vector; caller must have constructed the metric with scalarize=none.
    if Y.shape[1] != sum(len(spec._target_by_key) for spec in metric.moments):
        raise IdentifiabilityError(
            "stored metric width does not match moment_set raw width; "
            "identifiability requires scalarize='none' at metric construction"
        )
    s1, st = _sobol_per_column(X, Y, space, n_sobol)
    # Map raw column indices → moment name (using metric.moment_slices).
    col_to_moment: dict[int, str] = {}
    for name, sl in metric.moment_slices.items():
        for c in range(sl.start, sl.stop):
            col_to_moment[c] = name

    params: list[ParameterIdentifiability] = []
    for i, name in enumerate(space.names):
        first_row = s1[i]
        total_row = st[i]
        max_first = float(np.max(first_row))
        max_total = float(np.max(total_row))
        is_id = max_first >= threshold
        pin = None
        if not is_id:
            p = space.parameters[i]
            if p.prior is not None:
                pin = float(p.prior.mean)
        per_first: dict[str, float] = {}
        per_total: dict[str, float] = {}
        for j in range(s1.shape[1]):
            mname = col_to_moment[j]
            # If a parameter contributes to multiple moment columns
            # sharing a name, keep the maximum contribution (a single
            # "identifiable via this moment" answer).
            per_first[mname] = max(per_first.get(mname, 0.0), float(first_row[j]))
            per_total[mname] = max(per_total.get(mname, 0.0), float(total_row[j]))
        params.append(ParameterIdentifiability(
            name=name,
            max_first_order=max_first,
            max_total=max_total,
            is_identified=is_id,
            suggested_pin=pin,
            per_moment_first_order=per_first,
            per_moment_total=per_total,
        ))

    return IdentifiabilityReport(
        parameters=params,
        moment_names=metric.moment_names,
        n_train=int(X.shape[0]),
        n_sobol=int(n_sobol),
        threshold=float(threshold),
    )


def format_identifiability(report: IdentifiabilityReport) -> str:
    """Human-readable table for CLI output."""
    lines = [
        f"Identifiability report  (n_train={report.n_train}, "
        f"n_sobol={report.n_sobol}, threshold={report.threshold})",
        "",
        f"  {'parameter':<40} {'max_S1':>8} {'max_ST':>8} {'status':>12} {'suggested_pin':>14}",
        "  " + "-" * 88,
    ]
    for p in sorted(report.parameters, key=lambda x: -x.max_first_order):
        status = "IDENTIFIED" if p.is_identified else "unidentified"
        pin = f"{p.suggested_pin:.4g}" if p.suggested_pin is not None else "-"
        lines.append(
            f"  {p.name:<40} {p.max_first_order:>8.4f} {p.max_total:>8.4f} "
            f"{status:>12} {pin:>14}"
        )
    lines.append("")
    lines.append(
        f"  Identified:   {len(report.identified())} / {len(report.parameters)}"
    )
    lines.append(f"  Unidentified: {len(report.unidentified())}")
    unpinnable = [
        p.name for p in report.parameters
        if not p.is_identified and p.suggested_pin is None
    ]
    if unpinnable:
        lines.append(
            f"  NOTE: {len(unpinnable)} unidentified parameter(s) have no prior; "
            f"they cannot be pinned without one (add a `prior:` to the YAML)."
        )
    return "\n".join(lines)


def identifiability_as_dict(report: IdentifiabilityReport) -> dict[str, Any]:
    """JSON-serializable form."""
    return {
        "n_train": report.n_train,
        "n_sobol": report.n_sobol,
        "threshold": report.threshold,
        "moment_names": list(report.moment_names),
        "parameters": [
            {
                "name": p.name,
                "max_first_order": p.max_first_order,
                "max_total": p.max_total,
                "is_identified": p.is_identified,
                "suggested_pin": p.suggested_pin,
                "per_moment_first_order": dict(p.per_moment_first_order),
                "per_moment_total": dict(p.per_moment_total),
            }
            for p in report.parameters
        ],
    }


def auto_drop_to_prior_mean(
    config_yaml: Path, report: IdentifiabilityReport, *, dry_run: bool = False,
) -> tuple[Path, list[str]]:
    """Rewrite ``config_yaml`` to pin unidentified-with-prior parameters
    at their prior mean. Returns ``(output_path, list_of_pinned_names)``.

    Only parameters flagged with ``hold_at_prior_mean_if_unidentified: true``
    AND a set prior are actually pinned; the rest are reported as
    "would be pinned if you flag them".

    Non-destructive: writes to ``<stem>.pinned.yaml`` alongside the
    input by default. Pass ``dry_run=True`` to return the diff without
    writing.
    """
    import yaml

    with config_yaml.open() as fh:
        cfg = yaml.safe_load(fh)
    params = cfg.get("parameters", {}).get("inline", [])
    to_pin: list[str] = []
    for rec in params:
        name = rec.get("name")
        if not name:
            continue
        entry = next((p for p in report.parameters if p.name == name), None)
        if entry is None or entry.is_identified:
            continue
        if not rec.get("hold_at_prior_mean_if_unidentified", False):
            continue
        if entry.suggested_pin is None:
            continue
        # Pin: replace min/max with a tiny box around the prior mean.
        # A zero-width box would break most designs, so we use a
        # 0.1% relative half-width bounded below by a small absolute
        # floor. LHS-in-a-tiny-box degenerates to "always the pin".
        pin = entry.suggested_pin
        halfwidth = max(abs(pin) * 1e-3, 1e-6)
        rec["min"] = pin - halfwidth
        rec["max"] = pin + halfwidth
        rec["_pinned_at"] = pin  # audit trail
        to_pin.append(name)
    if dry_run:
        return config_yaml, to_pin
    out_path = config_yaml.with_suffix(f".pinned{config_yaml.suffix}")
    with out_path.open("w") as fh:
        yaml.safe_dump(cfg, fh, default_flow_style=False, sort_keys=False)
    return out_path, to_pin
