"""Post-hoc GP-Sobol sensitivity analysis on a completed SampleStore.

Fits a Matern-ARD GP to the (X, y) pairs from FINISHED samples of a study,
then runs SALib's Sobol analysis on GP-predicted values across a Sobol
low-discrepancy sequence to estimate first-order (S1) and total-effect
(ST) indices. Also reports the GP's fitted length-scales as a
"poor-person's variable-importance report" — parameters whose length-
scale saturates at the upper prior bound have effectively been dropped
by the surrogate.

Filed by the DFW calibration study (Phase 3B.1 pattern): a post-BO
diagnostic that costs a few surrogate fits and can shrink the effective
dimensionality of the next study by 60% or more.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from polarisopt.parameters import ParameterSpace
from polarisopt.samples.sample import SampleStatus
from polarisopt.samples.store import SampleStore

# Default number of Sobol samples for the SALib analysis. 2**13 = 8192
# gives good stability for ndim ~ 10 without dominating the analysis
# wall-time on a 200-point warmup (GP predict is cheap).
DEFAULT_N_SOBOL = 8192


@dataclass(slots=True)
class SensitivityReport:
    """Result of :func:`run_sensitivity_analysis`.

    Attributes
    ----------
    parameter_names
        Column-aligned names of the parameters (in ParameterSpace order).
    s1, s1_conf
        First-order Sobol indices and their 95% CIs.
    st, st_conf
        Total-effect Sobol indices and their 95% CIs.
    length_scales
        Fitted Matern-ARD length-scale per input (shape ``(ndim,)``).
        Length-scales that pegged near the upper prior bound flag
        dimensions the GP has effectively dropped.
    n_train
        Number of FINISHED samples the GP was fit on.
    n_sobol
        Number of surrogate evaluations SALib ran.
    """

    parameter_names: tuple[str, ...]
    s1: np.ndarray
    s1_conf: np.ndarray
    st: np.ndarray
    st_conf: np.ndarray
    length_scales: np.ndarray
    n_train: int
    n_sobol: int

    def ranked_by_st(self) -> list[tuple[str, float, float, float, float, float]]:
        """Return ``(name, s1, s1_conf, st, st_conf, length_scale)`` sorted by ST desc."""
        order = np.argsort(-self.st)
        return [
            (
                self.parameter_names[i],
                float(self.s1[i]), float(self.s1_conf[i]),
                float(self.st[i]), float(self.st_conf[i]),
                float(self.length_scales[i]),
            )
            for i in order
        ]


class SensitivityError(RuntimeError):
    """Raised when the store, samples, or fitted GP can't support the analysis."""


def _extract_xy(
    store: SampleStore, space: ParameterSpace, *, phase: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pull FINISHED (X, y) pairs from the store, filtered to a phase if given."""
    samples = store.list(phase=phase, status=SampleStatus.FINISHED)
    finished = [s for s in samples if s.metric is not None and s.inputs is not None]
    if not finished:
        raise SensitivityError(
            "no FINISHED samples with metric available"
            + (f" for phase={phase!r}" if phase else "")
        )
    X = np.stack([s.inputs for s in finished])
    metrics = [np.asarray(s.metric).reshape(-1) for s in finished]
    if any(len(m) != 1 for m in metrics):
        raise SensitivityError(
            "sensitivity only supports single-objective studies; "
            f"got multi-output metric of shape {metrics[0].shape} on sample {finished[0].id}"
        )
    Y = np.array([float(m[0]) for m in metrics]).reshape(-1, 1)
    if X.shape[1] != space.ndim:
        raise SensitivityError(
            f"stored inputs have ndim={X.shape[1]} but ParameterSpace has "
            f"ndim={space.ndim} — was the study run against a different space?"
        )
    return X, Y


def _fitted_length_scales(gp: Any) -> np.ndarray:
    """Extract the GP's Matern-ARD length-scale tensor as a numpy array.

    Robust to both SingleTaskGP and FixedNoiseGP paths and to whether
    the covar_module is a ScaleKernel wrapping MaternKernel (our GP
    plugin) or bare MaternKernel (a caller may override).
    """
    model = gp.model
    covar = getattr(model, "covar_module", None)
    base = getattr(covar, "base_kernel", covar)
    ls = base.lengthscale.detach().cpu().numpy().reshape(-1)
    return ls


def run_sensitivity_analysis(
    store: SampleStore,
    space: ParameterSpace,
    *,
    phase: str | None = None,
    n_sobol: int = DEFAULT_N_SOBOL,
    gp_kwargs: dict[str, Any] | None = None,
) -> SensitivityReport:
    """Fit a GP to the store's FINISHED samples and run Sobol on it.

    Parameters
    ----------
    store
        Opened :class:`SampleStore` bound to the study of interest.
    space
        The study's ParameterSpace. Column order of ``X`` must match.
    phase
        Optional phase filter — restrict to samples from one phase
        (e.g. only the LHS warmup). Default ``None`` (all phases).
    n_sobol
        Number of Sobol low-discrepancy points to evaluate the GP at.
        SALib's analyzer will consume ``n_sobol * (ndim + 2)`` GP
        predictions total. Default ``8192``.
    gp_kwargs
        Extra keyword arguments passed to :class:`GPSurrogate`
        (e.g. ``{"observation_noise": 2.79e-6}`` when you've measured
        the noise via a Phase 3B.0-style study).

    Returns
    -------
    SensitivityReport
        Ranked S1 / ST plus GP length-scales. See dataclass docstring.

    Raises
    ------
    SensitivityError
        If there are no FINISHED samples with metrics, the metric is
        multi-objective, or the stored input width doesn't match the
        ParameterSpace.
    """
    # Torch/BoTorch are in the [bo] extra; lazy-import so users
    # without them can still import the module (e.g. for typing).
    from SALib.analyze import sobol as sobol_analyze
    from SALib.sample import sobol as sobol_sample

    from polarisopt.surrogates.gp import GPSurrogate

    X, Y = _extract_xy(store, space, phase=phase)
    gp = GPSurrogate(**(gp_kwargs or {}))
    gp.fit(X, Y)

    bounds = space.bounds  # (ndim, 2)
    problem = {
        "num_vars": space.ndim,
        "names": list(space.names),
        "bounds": bounds.tolist(),
    }
    # SALib emits UserWarning when N is not a power of 2; caller can pick.
    param_values = sobol_sample.sample(problem, n_sobol)
    y_pred, _ = gp.predict(param_values)
    y_flat = np.asarray(y_pred).reshape(-1)
    Si = sobol_analyze.analyze(problem, y_flat, print_to_console=False)
    return SensitivityReport(
        parameter_names=tuple(space.names),
        s1=np.asarray(Si["S1"], dtype=float),
        s1_conf=np.asarray(Si["S1_conf"], dtype=float),
        st=np.asarray(Si["ST"], dtype=float),
        st_conf=np.asarray(Si["ST_conf"], dtype=float),
        length_scales=_fitted_length_scales(gp),
        n_train=int(X.shape[0]),
        n_sobol=int(param_values.shape[0]),
    )


def format_report(report: SensitivityReport) -> str:
    """Human-readable text table for CLI output."""
    lines = [
        f"GP-Sobol sensitivity analysis  (n_train={report.n_train}, "
        f"n_sobol_evals={report.n_sobol})",
        "",
        f"{'#':>2}  {'parameter':<40} {'S1':>8} {'ST':>8} {'length_scale':>14}",
        "  " + "-" * 76,
    ]
    for rank, (name, s1, _s1c, st, _stc, ls) in enumerate(report.ranked_by_st(), start=1):
        lines.append(f"{rank:>2}  {name:<40} {s1:>8.4f} {st:>8.4f} {ls:>14.4g}")
    lines.append("")
    lines.append(
        "  ST = total-effect index (includes interactions). "
        "S1 = first-order (main effect only)."
    )
    lines.append(
        "  Length-scale near the upper prior bound = dimension the GP "
        "cannot distinguish from noise; a candidate for removal."
    )
    return "\n".join(lines)


def report_as_dict(report: SensitivityReport) -> dict[str, Any]:
    """Structured form for machine consumption (JSON dump, notebooks)."""
    return {
        "n_train": report.n_train,
        "n_sobol": report.n_sobol,
        "parameters": [
            {
                "name": name,
                "s1": s1, "s1_conf": s1c,
                "st": st, "st_conf": stc,
                "length_scale": ls,
            }
            for name, s1, s1c, st, stc, ls in report.ranked_by_st()
        ],
    }


def _load_space_from_config(config: Path) -> tuple[ParameterSpace, Path]:
    """Resolve the study's ParameterSpace and workspace from a YAML config."""
    from polarisopt.config import load_study_config
    from polarisopt.studies.runner import _build_space

    study = load_study_config(config)
    workspace = Path(study.workspace).resolve()
    space = _build_space(study.parameters)
    return space, workspace
