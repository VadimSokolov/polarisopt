"""History-matching phase — Vernon-Goldstein-Bower 2010 wave protocol.

v0.31 P3+P9. Consumes a :class:`polarisopt.metrics.MomentSetMetric`
(v0.26) with `scalarize: none` so per-moment residuals are available.
Fits one GP per moment column, samples a dense grid, computes the
univariate implausibility per moment, aggregates via max (or
second_max for robustness), and retains the θ region where
``I_max(θ) < cutoff`` as the wave's Not Ruled Out Yet (NROY) set.

Emits ``nroy_wave{N}.parquet`` (P9) containing every dense-grid θ
labeled with its per-moment residuals, max implausibility, and a
retained flag.

Not iterated across multiple waves in this release — v0.31 ships a
single wave. Multi-wave chaining is a follow-up (v0.32+) that runs
consecutive wave phases with the retained region as the wave-2
LHS box.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from polarisopt.design.base import Design
from polarisopt.metrics.moment_set import MomentSetMetric
from polarisopt.samples.sample import Sample, SampleStatus
from polarisopt.studies.base import Study, StudyContext
from polarisopt.utils.logging import get_logger

log = get_logger(__name__)

DEFAULT_CUTOFF = 3.0  # Pukelsheim's 3-σ rule
DEFAULT_GRID_SIZE = 8192


@dataclass(slots=True)
class HistoryMatchingWavePhase:
    """Config bundle for one wave — mirrors the fields on
    :class:`HistoryMatchingPhaseConfig` in a runtime-friendlier shape.

    ``warm_up`` is the design that generates the wave's evaluation
    points. In wave 1 this is a full-box LHS; in later waves it can
    be a design restricted to the previous wave's NROY (not shipped
    in v0.31; the runner still supports one-wave-at-a-time).

    ``emulator`` shape (dispatched at ``_compute_nroy`` time):

    - ``{"type": "gp_per_moment", "options": {}}`` (v0.31 default) —
      one :class:`GPSurrogate` per active moment column. Simple, robust.
    - ``{"type": "gp_basis_pca", "options": {variance_retained, max_pcs,
      centering, scaling}}`` (v0.34+) — Higdon 2008 PC-basis GP.
      Fits ``k`` GPs on the top principal components of the moment
      matrix, reconstructs predictions through the basis. Cuts fit
      time from ``M`` to ``k`` GPs; dead moments get zero PC loading
      (auto-downweighted); correlated moments share structure. See
      :func:`_fit_predict_gp_basis_pca`.
    """

    name: str
    warm_up: Design
    emulator: dict[str, Any]
    implausibility: dict[str, Any]
    moments_included: list[str]
    nroy_grid_size: int
    output_dir: Path


class HistoryMatchingStudy(Study):
    """Single history-matching wave.

    See module docstring. Requires the study's metric to be a
    :class:`MomentSetMetric` with ``scalarize='none'``.
    """

    def __init__(self, ctx: StudyContext, phase: HistoryMatchingWavePhase) -> None:
        super().__init__(ctx)
        self.phase = phase
        if not isinstance(ctx.metric, MomentSetMetric):
            raise TypeError(
                f"history_matching requires a moment_set metric; "
                f"got {type(ctx.metric).__name__}"
            )
        if ctx.metric.scalarize != "none":
            raise ValueError(
                f"history_matching requires scalarize='none' so per-moment "
                f"residuals are exposed; got scalarize={ctx.metric.scalarize!r}"
            )
        self._metric: MomentSetMetric = ctx.metric

    def run(self) -> list[Sample]:
        # 1. Warm-up: generate + evaluate the wave's LHS.
        existing = self.ctx.store.list(
            phase=self.phase.name, status=SampleStatus.PENDING,
        )
        if existing:
            wave_samples = existing
        else:
            points = self.phase.warm_up.generate(self.ctx.space, rng=self.ctx.rng)
            wave_samples = [
                Sample(phase=self.phase.name, iteration=0, inputs=row) for row in points
            ]
            wave_samples = self.ctx.store.add_many(wave_samples)
            self._evaluate_batch(wave_samples)

        # 2. Fetch FINISHED (X, Y_moment_vector).
        finished = [
            s for s in self.ctx.store.list(
                phase=self.phase.name, status=SampleStatus.FINISHED,
            )
            if s.metric is not None
        ]
        if len(finished) < 2:
            log.warning(
                "history_matching phase %r has < 2 FINISHED samples; skipping "
                "NROY compute", self.phase.name,
            )
            return list(wave_samples)
        X = np.stack([s.inputs for s in finished])
        Y = np.stack([np.asarray(s.metric).reshape(-1) for s in finished])

        # 3. Per-moment GP + dense-grid implausibility.
        nroy_df = self._compute_nroy(X, Y)

        # 4. Write NROY parquet artifact (P9).
        self._write_nroy(nroy_df, wave_index=1)

        return list(wave_samples)

    def _compute_nroy(self, X: np.ndarray, Y: np.ndarray) -> Any:
        """Fit emulator, evaluate on a Sobol grid, compute
        max-implausibility per grid point, return a DataFrame.

        Emulator type is dispatched from ``self.phase.emulator["type"]``:

        - ``gp_per_moment`` (default, v0.31) — one
          :class:`GPSurrogate` per active moment column. Simple, robust,
          statistically inefficient at low N (fits M independent GPs).
        - ``gp_basis_pca`` (v0.34+, Higdon 2008) — PCA-decompose the
          moment matrix, fit ``k << M`` GPs on the top PC coefficients,
          reconstruct predictions through the basis. Dead moments get
          near-zero PC loading (auto-downweighted); highly-correlated
          moments share a GP. Better sample efficiency at low N.
        """
        import pandas as pd
        from scipy.stats import qmc

        # Column-slice metadata: for each raw column, name of the moment
        # it belongs to and the per-element obs/md std for implausibility.
        col_to_moment: dict[int, str] = {}
        for name, sl in self._metric.moment_slices.items():
            for c in range(sl.start, sl.stop):
                col_to_moment[c] = name
        obs_std = self._metric.obs_noise_std_vector
        md_std = self._metric.model_discrepancy_std_vector

        # Optional per-wave moment subset (moments_included).
        included = set(self.phase.moments_included) if self.phase.moments_included else set()
        active_cols = [
            j for j in range(Y.shape[1])
            if not included or col_to_moment[j] in included
        ]

        # Sobol grid over the parameter box.
        sampler = qmc.Sobol(d=self.ctx.space.ndim, scramble=True, rng=self.ctx.rng)
        n_grid = int(self.phase.nroy_grid_size)
        grid_unit = sampler.random(n=n_grid)
        bounds = self.ctx.space.bounds
        grid = grid_unit * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        grid = self.ctx.space.clip(grid)

        # Emulator dispatch. Both emulators produce (n_grid, len(active_cols))
        # arrays of predictive mean + var; the implausibility computation
        # downstream is emulator-agnostic.
        Y_active = Y[:, active_cols]
        emu_spec = self.phase.emulator or {"type": "gp_per_moment", "options": {}}
        emu_type = str(emu_spec.get("type", "gp_per_moment"))
        emu_opts = dict(emu_spec.get("options", {}) or {})
        if emu_type == "gp_per_moment":
            pred_mean, pred_var = _fit_predict_gp_per_moment(X, Y_active, grid)
        elif emu_type == "gp_basis_pca":
            pred_mean, pred_var = _fit_predict_gp_basis_pca(X, Y_active, grid, **emu_opts)
        else:
            raise ValueError(
                f"history_matching: unknown emulator type {emu_type!r} "
                f"(expected 'gp_per_moment' or 'gp_basis_pca')"
            )
        col_obs_std = obs_std[active_cols]
        col_md_std = md_std[active_cols]
        col_names = [f"resid[{col_to_moment[j]}][{j}]" for j in active_cols]

        # Univariate implausibility per (grid_point, moment_column):
        #   I² = (mean − z)² / (var_gp + var_md + var_obs)
        # z is the target residual — in the residual-vector metric, the
        # target IS zero (we're fitting sim - target directly), so mean²
        # in the numerator.
        denom = pred_var + col_md_std**2 + col_obs_std**2
        impl_sq = (pred_mean**2) / np.maximum(denom, 1e-30)
        impl = np.sqrt(impl_sq)

        # Reduction (max / second_max).
        impl_type = str(self.phase.implausibility.get("type", "max"))
        cutoff = float(self.phase.implausibility.get("cutoff", DEFAULT_CUTOFF))
        if impl_type == "max":
            i_agg = impl.max(axis=1)
            i_second = np.partition(impl, -2, axis=1)[:, -2] if impl.shape[1] >= 2 else i_agg
        elif impl_type == "second_max":
            i_second = np.partition(impl, -2, axis=1)[:, -2] if impl.shape[1] >= 2 else impl.max(axis=1)
            i_agg = i_second
        else:
            raise ValueError(f"implausibility.type must be max or second_max, got {impl_type!r}")

        retained = i_agg < cutoff
        cols: dict[str, Any] = {name: grid[:, i] for i, name in enumerate(self.ctx.space.names)}
        cols["implausibility_max"] = i_agg.astype(np.float32)
        cols["implausibility_second"] = i_second.astype(np.float32)
        cols["retained"] = retained
        # Attach per-moment residuals as a JSON-serialized string per row
        # to keep the parquet schema flat.
        import json
        cols["moment_residuals_json"] = [
            json.dumps({name: float(pred_mean[i, k]) for k, name in enumerate(col_names)})
            for i in range(n_grid)
        ]
        return pd.DataFrame(cols)

    def _write_nroy(self, df: Any, *, wave_index: int) -> None:
        out_dir = Path(self.phase.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"nroy_wave{wave_index}.parquet"
        # pyarrow / fastparquet not a hard dep — fall back to CSV if
        # neither is installed, so users can still see the artifact.
        try:
            df.to_parquet(out_path, index=False)
            log.info(
                "history_matching phase %r: NROY wave %d → %s "
                "(%d/%d retained at cutoff %s)",
                self.phase.name, wave_index, out_path,
                int(df["retained"].sum()), len(df),
                self.phase.implausibility.get("cutoff", DEFAULT_CUTOFF),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "history_matching: parquet write failed (%s); falling back to CSV",
                exc,
            )
            csv_path = out_dir / f"nroy_wave{wave_index}.csv"
            df.to_csv(csv_path, index=False)
            log.info(
                "history_matching phase %r: NROY wave %d → %s (CSV fallback)",
                self.phase.name, wave_index, csv_path,
            )


# ----- emulator implementations (v0.31 gp_per_moment + v0.34 gp_basis_pca) -----


def _fit_predict_gp_per_moment(
    X: np.ndarray, Y: np.ndarray, X_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """v0.31 baseline: one GPSurrogate per Y column.

    Simple and robust. Statistically inefficient when M is large and Y
    columns are correlated — every moment gets its own GP. Use
    :func:`_fit_predict_gp_basis_pca` for the correlated-multivariate
    case.

    Degenerate columns (identical residual across training samples)
    return the constant value with zero variance instead of crashing
    the per-column GP fit.
    """
    from polarisopt.surrogates.gp import GPSurrogate

    n_pred = X_pred.shape[0]
    m = Y.shape[1]
    pred_mean = np.zeros((n_pred, m), dtype=float)
    pred_var = np.zeros((n_pred, m), dtype=float)
    for k in range(m):
        y_col = Y[:, k : k + 1]
        if float(np.ptp(y_col)) == 0.0:
            pred_mean[:, k] = float(y_col[0, 0])
            pred_var[:, k] = 0.0
            continue
        gp = GPSurrogate()
        gp.fit(X, y_col)
        m_i, v_i = gp.predict(X_pred)
        pred_mean[:, k] = m_i.reshape(-1)
        pred_var[:, k] = v_i.reshape(-1)
    return pred_mean, pred_var


def _fit_predict_gp_basis_pca(
    X: np.ndarray,
    Y: np.ndarray,
    X_pred: np.ndarray,
    *,
    variance_retained: float = 0.99,
    max_pcs: int = 5,
    centering: str = "mean",
    scaling: str = "per_moment_std",
) -> tuple[np.ndarray, np.ndarray]:
    """v0.34 emulator: PC-basis GP for multivariate output (Higdon 2008).

    Semantics
    ---------
    Given the moment matrix ``Y (N, M)``:

    1. Center per-column (``mean`` default, or ``median`` / ``none``).
    2. Scale per-column (``per_moment_std`` default, or ``unit`` / ``none``).
    3. SVD: ``Y_c = U · S · V^T``. Take basis ``K = V^T[:k, :]``
       (shape ``(k, M)``) and coefficients ``W = U[:, :k] · diag(S[:k])``
       (shape ``(N, k)``). Choose ``k`` = smallest such that cumulative
       explained variance ≥ ``variance_retained``, capped at ``max_pcs``.
    4. Fit ``k`` independent GPs on ``W[:, i]`` vs ``X``.
    5. At inference: predict ``W_pred_i`` mean + var per PC; reconstruct
       ``Y_c_pred = W_pred @ K``; propagate variance
       ``Var(Y_c_pred_j) = Σ_i K[i, j]² · Var(W_pred_i)``; invert scale
       + center.

    Benefits
    --------
    - ``k`` GPs instead of ``M`` — cuts fit + predict work by ``M/k``.
    - Dead moments (zero-variance columns) get zero PC loading —
      auto-downweighted without ``moments_included`` gymnastics.
    - Correlated moments share the same PC direction — better sample
      efficiency at low N.

    Reference
    ---------
    Higdon, Gattiker, Williams & Rightley (2008, *JASA*):
    "Computer Model Calibration Using High-Dimensional Output".

    Parameters
    ----------
    X, Y : array-like
        Training inputs ``(N, D)`` and multivariate outputs ``(N, M)``.
    X_pred : array-like
        Inference inputs ``(N_pred, D)``.
    variance_retained : float, optional
        Cumulative explained variance threshold for choosing ``k``.
        Default 0.99. Capped by ``max_pcs``.
    max_pcs : int, optional
        Hard cap on ``k``. Default 5.
    centering : {"mean", "median", "none"}, optional
    scaling : {"per_moment_std", "unit", "none"}, optional
    """
    from polarisopt.surrogates.gp import GPSurrogate

    if Y.ndim != 2:
        raise ValueError(f"gp_basis_pca: Y must be 2-D, got shape {Y.shape}")
    n_train, m = Y.shape
    if n_train < 2:
        raise ValueError(
            f"gp_basis_pca: need at least 2 training samples, got {n_train}"
        )
    if variance_retained <= 0 or variance_retained > 1:
        raise ValueError(
            f"gp_basis_pca.variance_retained must be in (0, 1], got {variance_retained}"
        )
    if max_pcs < 1:
        raise ValueError(f"gp_basis_pca.max_pcs must be >= 1, got {max_pcs}")

    # 1. Center.
    if centering == "mean":
        center = Y.mean(axis=0)
    elif centering == "median":
        center = np.median(Y, axis=0)
    elif centering == "none":
        center = np.zeros(m, dtype=float)
    else:
        raise ValueError(f"gp_basis_pca.centering must be mean|median|none, got {centering!r}")
    Y_c = Y - center

    # 2. Scale.
    if scaling == "per_moment_std":
        scale = Y_c.std(axis=0, ddof=1)
        scale = np.where(scale > 0, scale, 1.0)
    elif scaling == "unit" or scaling == "none":
        scale = np.ones(m, dtype=float)
    else:
        raise ValueError(
            f"gp_basis_pca.scaling must be per_moment_std|unit|none, got {scaling!r}"
        )
    Y_s = Y_c / scale

    # 3. SVD → basis K (k, M) and coefficients W (N, k).
    # SciPy default `full_matrices=True` gives U (N, N), V^T (M, M); we
    # only need `min(N, M)` singular vectors.
    U, S, Vt = np.linalg.svd(Y_s, full_matrices=False)
    total_var = float(np.sum(S**2))
    if total_var <= 0:
        # Every column is constant — return the un-scaled center per row
        # with zero variance. Emulator would fit any GP to a flat target
        # anyway; short-circuit for stability.
        pred_mean = np.broadcast_to(center, (X_pred.shape[0], m)).copy()
        pred_var = np.zeros((X_pred.shape[0], m), dtype=float)
        return pred_mean, pred_var
    cum_var = np.cumsum(S**2) / total_var
    k = int(np.searchsorted(cum_var, variance_retained) + 1)
    k = max(1, min(k, max_pcs, len(S)))
    K = Vt[:k, :]                 # (k, M)
    W = U[:, :k] * S[:k]          # (N, k)

    log.info(
        "gp_basis_pca: fit %d GPs on %d-D coefficients "
        "(variance_retained=%.2f cap=%d, explained=%.3f)",
        k, k, variance_retained, max_pcs, float(cum_var[k - 1]),
    )

    # 4. Fit k independent GPs on W columns.
    n_pred = X_pred.shape[0]
    W_pred_mean = np.zeros((n_pred, k), dtype=float)
    W_pred_var = np.zeros((n_pred, k), dtype=float)
    for i in range(k):
        w_col = W[:, i : i + 1]
        if float(np.ptp(w_col)) == 0.0:
            W_pred_mean[:, i] = float(w_col[0, 0])
            W_pred_var[:, i] = 0.0
            continue
        gp = GPSurrogate()
        gp.fit(X, w_col)
        m_i, v_i = gp.predict(X_pred)
        W_pred_mean[:, i] = m_i.reshape(-1)
        W_pred_var[:, i] = v_i.reshape(-1)

    # 5. Reconstruct predictions in standardized space, then un-scale
    # + un-center.
    pred_mean_s = W_pred_mean @ K                    # (n_pred, M)
    # Variance under a linear map: Var(K^T · w) = K^T · Cov(w) · K.
    # Since we fit independent GPs per PC, Cov(w) is diagonal, so:
    #   Var(y_j) = Σ_i K[i, j]² · Var(w_i)
    K_sq = K**2                                       # (k, M)
    pred_var_s = W_pred_var @ K_sq                   # (n_pred, M)
    pred_mean = pred_mean_s * scale + center
    pred_var = pred_var_s * (scale**2)
    return pred_mean, pred_var
