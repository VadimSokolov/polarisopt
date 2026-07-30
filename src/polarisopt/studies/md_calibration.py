"""Empirical model-discrepancy calibration (v0.33 P2).

Vernon 2010 §3.5 explicit failure mode: understated
``model_discrepancy_std`` produces empty NROY in history matching. This
module estimates the discrepancy empirically from a completed wave's
per-moment residuals via leave-one-out cross-validation on a GP
emulator fit to each moment column.

The math: for each moment column j, the CV residual
``r_i = y_i - E[y|X_{-i}]`` has variance
``σ²_r,j = σ²_obs,j + σ²_md,j + σ²_emu,j``
where the emulator variance is known from the GP posterior. Subtract
the known terms to isolate ``σ²_md,j``:

    md²_j ≈ max(0, mean(r²) - σ²_obs,j - mean(gp_var_at_holdout))

Consumers:

- ``polarisopt calibrate-md <yaml>`` — CLI that runs LOO CV on all
  ``model_discrepancy_std: auto`` moments and writes a
  ``md_std_calibrated.yaml`` snippet the user copies back into the
  study YAML.
- Programmatic use from notebooks — call
  :func:`calibrate_md_from_store` directly.

Guardrails from the DFW report:

- If empirical md > 3× user's initial guess → warn "wave 1 md was
  understated, expect NROY expansion".
- If empirical md < 0.3× user's guess → warn "over-permissive md, NROY
  may admit implausibles".
- If empirical md ≈ 0 → warn "residuals dominated by obs+emu, moment
  contributes nothing to implausibility".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from polarisopt.metrics.moment_set import MomentSetMetric, is_md_auto
from polarisopt.parameters import ParameterSpace
from polarisopt.samples.sample import SampleStatus
from polarisopt.samples.store import SampleStore


class MdCalibrationError(RuntimeError):
    """Raised when the store / metric / space can't support MD calibration."""


@dataclass(slots=True)
class MdEstimate:
    """Per-moment MD calibration result."""

    moment_name: str
    empirical_md_std: float
    user_md_std: float | None  # None when the moment was 'auto'
    residual_var: float
    obs_var: float
    emulator_var_mean: float
    n_samples: int


def calibrate_md_from_store(
    store: SampleStore,
    space: ParameterSpace,
    metric: MomentSetMetric,
    *,
    phase: str | None = None,
    only_auto: bool = True,
) -> list[MdEstimate]:
    """Estimate empirical ``model_discrepancy_std`` per moment via LOO CV.

    Parameters
    ----------
    store
        Opened SampleStore with FINISHED samples for the phase.
    space
        The study's ParameterSpace (column order of stored X must match).
    metric
        The ``moment_set`` metric — used to slice residuals back into
        named moments and to read the user's obs / initial md values.
    phase
        Optional phase filter (default: all phases).
    only_auto
        When True (default), only moments whose ``model_discrepancy_std``
        was set to ``'auto'`` in the YAML get calibrated — moments the
        user pinned to a numeric value are left alone. Set to False to
        recalibrate every moment (useful for auditing).

    Returns
    -------
    list of MdEstimate

    Raises
    ------
    MdCalibrationError
        If there are no FINISHED samples with residual vectors matching
        the metric width, or if X and stored metric-widths are
        inconsistent across samples.
    """
    from polarisopt.surrogates.gp import GPSurrogate

    samples = store.list(phase=phase, status=SampleStatus.FINISHED)
    finished = [s for s in samples if s.metric is not None and s.inputs is not None]
    if not finished:
        raise MdCalibrationError(
            "no FINISHED samples with metric available"
            + (f" for phase={phase!r}" if phase else "")
        )
    widths = {np.asarray(s.metric).size for s in finished}
    if len(widths) != 1:
        raise MdCalibrationError(
            f"inconsistent stored metric widths {sorted(widths)}; "
            "MD calibration requires a fixed moment_set output length"
        )
    X = np.stack([s.inputs for s in finished])
    Y = np.stack([np.asarray(s.metric).reshape(-1) for s in finished])
    if X.shape[1] != space.ndim:
        raise MdCalibrationError(
            f"stored inputs have ndim={X.shape[1]} but ParameterSpace has "
            f"ndim={space.ndim}"
        )
    if Y.shape[1] != sum(len(spec._target_by_key) for spec in metric.moments):
        raise MdCalibrationError(
            "stored metric width does not match moment_set raw width; "
            "MD calibration requires scalarize='none' at metric construction"
        )
    if X.shape[0] < 3:
        raise MdCalibrationError(
            f"MD calibration needs at least 3 samples for LOO CV; got {X.shape[0]}"
        )

    # For each moment name, pick its representative column index (the
    # first one; if a moment spans multiple targets the same md applies).
    per_moment_col: dict[str, int] = {}
    for name, sl in metric.moment_slices.items():
        per_moment_col[name] = sl.start

    results: list[MdEstimate] = []
    for spec in metric.moments:
        if only_auto and not is_md_auto(spec):
            continue
        col = per_moment_col[spec.name]
        y_col = Y[:, col : col + 1]
        if float(np.ptp(y_col)) == 0.0:
            # Degenerate column — no variance to attribute. md is 0 here.
            results.append(MdEstimate(
                moment_name=spec.name, empirical_md_std=0.0,
                user_md_std=None if is_md_auto(spec) else float(spec.model_discrepancy_std),
                residual_var=0.0, obs_var=spec.obs_noise_std**2,
                emulator_var_mean=0.0, n_samples=int(X.shape[0]),
            ))
            continue

        # Leave-one-out CV on a GP per moment.
        loo_residuals = np.zeros(X.shape[0], dtype=float)
        loo_gp_vars = np.zeros(X.shape[0], dtype=float)
        for i in range(X.shape[0]):
            mask = np.arange(X.shape[0]) != i
            gp = GPSurrogate()
            gp.fit(X[mask], y_col[mask])
            mean_i, var_i = gp.predict(X[i : i + 1])
            loo_residuals[i] = float(y_col[i, 0]) - float(mean_i[0, 0])
            loo_gp_vars[i] = float(var_i[0, 0])
        residual_var = float(np.mean(loo_residuals**2))
        emu_var_mean = float(np.mean(loo_gp_vars))
        obs_var = float(spec.obs_noise_std**2)
        md_var = max(0.0, residual_var - obs_var - emu_var_mean)
        results.append(MdEstimate(
            moment_name=spec.name,
            empirical_md_std=float(np.sqrt(md_var)),
            user_md_std=None if is_md_auto(spec) else float(spec.model_discrepancy_std),
            residual_var=residual_var,
            obs_var=obs_var,
            emulator_var_mean=emu_var_mean,
            n_samples=int(X.shape[0]),
        ))
    return results


def format_md_report(estimates: list[MdEstimate]) -> str:
    """Human-readable table for CLI output."""
    lines = [
        f"Model-discrepancy calibration  ({len(estimates)} moment(s))",
        "",
        f"  {'moment':<30} {'user_md':>10} {'empirical_md':>14} "
        f"{'residual_std':>14} {'obs_std':>10} {'emu_std':>10} {'guidance':<40}",
        "  " + "-" * 130,
    ]
    for e in estimates:
        user_cell = f"{e.user_md_std:.4g}" if e.user_md_std is not None else "auto"
        guidance = _guidance(e)
        lines.append(
            f"  {e.moment_name[:30]:<30} {user_cell:>10} "
            f"{e.empirical_md_std:>14.4g} {np.sqrt(e.residual_var):>14.4g} "
            f"{np.sqrt(e.obs_var):>10.4g} {np.sqrt(e.emulator_var_mean):>10.4g} "
            f"{guidance:<40}"
        )
    return "\n".join(lines)


def _guidance(e: MdEstimate) -> str:
    """Vernon 2010 §3.5 style warnings on the empirical/user md ratio."""
    if e.empirical_md_std <= 0:
        return "residuals dominated by obs+emu; moment inert"
    if e.user_md_std is None:
        return "OK (write into yaml via calibrated snippet)"
    if e.user_md_std <= 0:
        return "user was 0 — set to empirical"
    ratio = e.empirical_md_std / e.user_md_std
    if ratio > 3:
        return f"empirical > 3× user ({ratio:.1f}×) — NROY may expand"
    if ratio < 0.3:
        return f"empirical < 0.3× user ({ratio:.2f}×) — NROY may admit implausibles"
    return "OK"


def calibrated_yaml_snippet(estimates: list[MdEstimate]) -> str:
    """Emit the metric.options.moments overrides as a copy-pasteable YAML block."""
    lines = ["# polarisopt calibrate-md — paste into metric.options.moments overrides"]
    lines.append("metric:")
    lines.append("  options:")
    lines.append("    moments:")
    for e in estimates:
        lines.append(f"      - name: {e.moment_name}")
        lines.append(f"        model_discrepancy_std: {e.empirical_md_std:.6g}")
    return "\n".join(lines) + "\n"


def estimates_as_dict(estimates: list[MdEstimate]) -> dict[str, Any]:
    """JSON-serializable form."""
    return {
        "moments": [
            {
                "name": e.moment_name,
                "empirical_md_std": e.empirical_md_std,
                "user_md_std": e.user_md_std,
                "residual_std": float(np.sqrt(e.residual_var)),
                "obs_std": float(np.sqrt(e.obs_var)),
                "emulator_std": float(np.sqrt(e.emulator_var_mean)),
                "n_samples": e.n_samples,
                "guidance": _guidance(e),
            }
            for e in estimates
        ],
    }
