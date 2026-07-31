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
from polarisopt.utils.logging import get_logger

log = get_logger(__name__)


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

    targets = [s for s in metric.moments if not (only_auto and not is_md_auto(s))]
    # v0.36: LOO refits a full GP (fit_gpytorch_mll hyperparameter
    # optimization, not a cheap update) per held-out sample per column.
    # That is O(N * total_columns) fits and can run for hours on a real
    # wave; previously this module logged nothing at all and presented as
    # a hung terminal. Announce the cost up front and log per moment.
    n_train = int(X.shape[0])
    total_cols = sum(
        metric.moment_slices[s.name].stop - metric.moment_slices[s.name].start
        for s in targets
    )
    log.info(
        "calibrate-md: leave-one-out CV over %d moment(s) / %d column(s) x %d "
        "samples = %d GP fits. This is minutes-to-hours on a large wave; "
        "progress is logged per moment.",
        len(targets), total_cols, n_train, total_cols * n_train,
    )

    results: list[MdEstimate] = []
    for spec in targets:
        sl = metric.moment_slices[spec.name]
        # v0.36: pool residuals across EVERY column of the moment. The
        # previous implementation used only sl.start — so a 27-element
        # mode-share moment was calibrated from 1 element, the answer
        # depended on target-CSV row order, and if that first row
        # happened to be an all-zero bucket the whole moment reported
        # md=0 and the snippet told the user to write 0 into their YAML,
        # producing exactly the Vernon empty-NROY failure this command
        # exists to prevent.
        col_indices = list(range(sl.start, sl.stop))
        live_cols = [c for c in col_indices if float(np.ptp(Y[:, c])) > 0.0]
        n_dead = len(col_indices) - len(live_cols)
        if not live_cols:
            log.warning(
                "calibrate-md: moment %r has zero variance in all %d column(s) "
                "across %d samples — cannot separate discrepancy from noise. "
                "Reporting md=0; check the moment SQL actually varies with the "
                "parameters before pasting this into your YAML.",
                spec.name, len(col_indices), n_train,
            )
            results.append(MdEstimate(
                moment_name=spec.name, empirical_md_std=0.0,
                user_md_std=None if is_md_auto(spec) else float(spec.model_discrepancy_std),
                residual_var=0.0, obs_var=spec.obs_noise_std**2,
                emulator_var_mean=0.0, n_samples=n_train,
            ))
            continue
        if n_dead:
            log.info(
                "calibrate-md: moment %r — %d of %d columns are constant and "
                "excluded from the estimate", spec.name, n_dead, len(col_indices),
            )
        log.info(
            "calibrate-md: moment %r — %d GP fits (%d live column(s) x %d samples)",
            spec.name, len(live_cols) * n_train, len(live_cols), n_train,
        )

        # Leave-one-out CV, pooled over all live columns of this moment.
        pooled_sq_residuals: list[float] = []
        pooled_gp_vars: list[float] = []
        for col in live_cols:
            y_col = Y[:, col : col + 1]
            for i in range(n_train):
                mask = np.arange(n_train) != i
                gp = GPSurrogate()
                gp.fit(X[mask], y_col[mask])
                mean_i, var_i = gp.predict(X[i : i + 1])
                pooled_sq_residuals.append(
                    (float(y_col[i, 0]) - float(mean_i[0, 0])) ** 2
                )
                pooled_gp_vars.append(float(var_i[0, 0]))
        residual_var = float(np.mean(pooled_sq_residuals))
        emu_var_mean = float(np.mean(pooled_gp_vars))
        obs_var = float(spec.obs_noise_std**2)
        md_var = max(0.0, residual_var - obs_var - emu_var_mean)
        results.append(MdEstimate(
            moment_name=spec.name,
            empirical_md_std=float(np.sqrt(md_var)),
            user_md_std=None if is_md_auto(spec) else float(spec.model_discrepancy_std),
            residual_var=residual_var,
            obs_var=obs_var,
            emulator_var_mean=emu_var_mean,
            n_samples=n_train,
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
