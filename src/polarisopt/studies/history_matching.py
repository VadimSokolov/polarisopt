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
from polarisopt.metrics.moment_set import MomentSetMetric, is_md_auto
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
    - ``{"type": "gbc_iqn", "options": {hdim, nh, training_epochs,
      learning_rate, loss_weights, combine_with_pca,
      pca_variance_retained, pca_max_pcs, n_tau_samples_at_inference,
      seed}}`` (v0.35+) — Implicit Quantile Network from
      Polson-Sokolov 2026 (``github.com/VadimSokolov/gbc``). Handles
      discontinuous / highly non-Gaussian ``(θ → y)`` mappings the GP
      per-column path struggles with. Optional ``combine_with_pca:
      true`` trains IQNs on PC coefficients instead of raw Y — same
      dimensionality benefits as ``gp_basis_pca`` + non-Gaussian
      residual modeling. Requires the ``[calibration]`` extra:
      ``pip install polarisopt[bo,calibration]``. See
      :func:`_fit_predict_gbc_iqn`.
    """

    name: str
    warm_up: Design
    emulator: dict[str, Any]
    implausibility: dict[str, Any]
    moments_included: list[str]
    nroy_grid_size: int
    output_dir: Path
    # v0.36: wave number for the emitted artifact. Was hardcoded to 1,
    # so the documented multi-phase pattern (hm-wave-1/2/3) had every
    # phase overwrite the same nroy_wave1.parquet — waves 1 and 2 were
    # destroyed and wave 3 survived under a filename claiming wave 1.
    # Defaults to 1; the runner derives it from the phase config.
    wave_index: int = 1


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
        # v0.36: reject the 'auto' model-discrepancy sentinel up front.
        # NaN md propagates into `denom` below; np.maximum(nan, 1e-30)
        # returns nan (it is NOT a clamp for NaN), so implausibility is
        # all-NaN, `nan < cutoff` is False for every grid point, and the
        # wave writes a 0/N-retained NROY that is indistinguishable from
        # a legitimate "the model cannot match the targets" result. That
        # is precisely the Vernon 2010 §3.5 failure this library exists
        # to prevent, so it must be an error, not a silent artifact.
        auto_moments = [s.name for s in ctx.metric.moments if is_md_auto(s)]
        if auto_moments:
            raise ValueError(
                f"history_matching cannot run with model_discrepancy_std='auto' "
                f"on moment(s) {auto_moments}. Implausibility is undefined until "
                f"the discrepancy is a number. Run the wave with a scalar "
                f"md (or a static design phase), then `polarisopt calibrate-md` "
                f"on the results, paste the calibrated values into the YAML, and "
                f"re-run this phase."
            )
        self._metric: MomentSetMetric = ctx.metric

    def run(self) -> list[Sample]:
        # 1. Warm-up: generate + evaluate the wave's LHS.
        # v0.36: the resume path must ALSO evaluate. Previously
        # `_evaluate_batch` sat inside the else-branch, so restarting a
        # partially-completed wave picked up the PENDING rows, evaluated
        # NONE of them, and went straight to _compute_nroy on whatever
        # subset had finished before the crash — writing a full-looking
        # NROY built from a fraction of the design. static.py and
        # sequential.py both evaluate on resume; this was the outlier.
        existing = self.ctx.store.list(
            phase=self.phase.name, status=SampleStatus.PENDING,
        )
        if existing:
            log.info(
                "history_matching phase %r: resuming — evaluating %d pending sample(s)",
                self.phase.name, len(existing),
            )
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
        self._write_nroy(nroy_df, wave_index=self.phase.wave_index)

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

        # Optional per-wave moment subset (moments_included). v0.36:
        # validate the names — previously a typo silently produced an
        # empty active_cols, and the failure only surfaced as an opaque
        # "zero-size array to reduction" ValueError at the implausibility
        # step, after the entire wave of simulations had already run.
        included = set(self.phase.moments_included) if self.phase.moments_included else set()
        if included:
            unknown = sorted(included - set(self._metric.moment_names))
            if unknown:
                raise ValueError(
                    f"history_matching.moments_included names not in the metric: "
                    f"{unknown}. Available moments: {list(self._metric.moment_names)}"
                )
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
            # v0.36: previously emu_opts was silently discarded on this
            # branch only (the other two splat it), so a typo'd option
            # under the DEFAULT emulator vanished without a word while
            # the same typo under gp_basis_pca raised TypeError.
            if emu_opts:
                raise ValueError(
                    f"history_matching: emulator type 'gp_per_moment' takes no "
                    f"options; got {sorted(emu_opts)}. (Did you mean "
                    f"'gp_basis_pca' or 'gbc_iqn'?)"
                )
            pred_mean, pred_var = _fit_predict_gp_per_moment(X, Y_active, grid)
        elif emu_type == "gp_basis_pca":
            pred_mean, pred_var = _fit_predict_gp_basis_pca(X, Y_active, grid, **emu_opts)
        elif emu_type == "gbc_iqn":
            pred_mean, pred_var = _fit_predict_gbc_iqn(X, Y_active, grid, **emu_opts)
        else:
            raise ValueError(
                f"history_matching: unknown emulator type {emu_type!r} "
                f"(expected 'gp_per_moment' | 'gp_basis_pca' | 'gbc_iqn')"
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

        # v0.36: drop columns whose residual is constant across the whole
        # training design. Such a column cannot discriminate between any
        # two θ — it contributes the SAME implausibility everywhere — so
        # including it can only shift every grid point uniformly. When
        # that constant is large (the common case: a mode the sim never
        # produces, so residual == −target for every design), it silently
        # rules out the entire NROY for a reason unrelated to θ. That is
        # a moment-specification / model-structure problem the user must
        # be told about, not folded into the retained set.
        dead_mask = np.ptp(Y_active, axis=0) == 0.0
        if dead_mask.any():
            dead_report = [
                f"{col_names[k]} (constant residual {float(Y_active[0, k]):+.4g})"
                for k in np.flatnonzero(dead_mask)
            ]
            log.warning(
                "history_matching phase %r: %d moment column(s) have a constant "
                "residual across all %d training samples and cannot discriminate "
                "between parameter values — EXCLUDED from implausibility: %s. "
                "A large constant residual means the simulator never reproduces "
                "that target category; fix the moment SQL or the target CSV "
                "rather than reading this as an empty NROY.",
                self.phase.name, int(dead_mask.sum()), Y_active.shape[0],
                "; ".join(dead_report),
            )
            impl = impl[:, ~dead_mask]
        if impl.shape[1] == 0:
            raise ValueError(
                f"history_matching phase {self.phase.name!r}: every moment column "
                f"has a constant residual across the design, so no moment can "
                f"discriminate between parameter values. Implausibility is "
                f"undefined. Check that the moment SQL returns varying values "
                f"across samples and that the parameters actually affect it."
            )

        # v0.36: prior terms as virtual moments. Declared in the schema
        # and docs since v0.31 with default True, but never implemented —
        # priors silently failed to constrain the NROY, which is the whole
        # reason the option exists. Vernon's construction treats a prior
        # as one more implausibility contribution:
        #     I_prior,d(θ) = |θ_d − prior_mean_d| / prior_std_d
        # Flat (uniform) priors return std=None and are skipped: they carry
        # no information and would otherwise penalise the box edges purely
        # for being edges.
        if bool(self.phase.implausibility.get("include_prior_terms", True)):
            prior_cols = []
            prior_names = []
            for d, p in enumerate(self.ctx.space.parameters):
                if p.prior is None:
                    continue
                scale = getattr(p.prior, "std", None)
                if scale is None or not np.isfinite(scale) or scale <= 0:
                    continue
                prior_cols.append(np.abs(grid[:, d] - p.prior.mean) / float(scale))
                prior_names.append(p.name)
            if prior_cols:
                log.info(
                    "history_matching phase %r: adding %d prior term(s) to "
                    "implausibility (%s)",
                    self.phase.name, len(prior_cols), ", ".join(prior_names),
                )
                impl = np.concatenate([impl, np.stack(prior_cols, axis=1)], axis=1)

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
        # v0.36: warn rather than silently clobber. Two phases sharing a
        # wave_index (or a re-run over a previous result) used to destroy
        # the earlier artifact with no trace.
        if out_path.exists():
            log.warning(
                "history_matching phase %r: overwriting existing %s. Give each "
                "wave a distinct `wave_index` (or `output_dir`) if you meant to "
                "keep both.",
                self.phase.name, out_path,
            )
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
    n_pcs_out: list[int] | None = None,
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
    n_pcs_out : list, optional
        Out-parameter: when a list is supplied, the number of principal
        components actually retained is appended to it. Lets callers
        (and tests) observe the compression the emulator achieved
        without parsing logs.
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

    if n_pcs_out is not None:
        n_pcs_out.append(int(k))
    explained = float(cum_var[k - 1])
    log.info(
        "gp_basis_pca: fit %d GPs on %d-D coefficients "
        "(variance_retained=%.2f cap=%d, explained=%.3f)",
        k, k, variance_retained, max_pcs, explained,
    )
    if explained < variance_retained:
        log.warning(
            "gp_basis_pca: max_pcs=%d truncated the basis at %d PC(s) explaining "
            "only %.1f%% of variance (requested %.1f%%). The discarded %.1f%% "
            "enters the predictive MEAN as reconstruction bias; the residual "
            "variance below only partially covers it, so implausibility may be "
            "inflated and the NROY over-pruned. Raise max_pcs or lower "
            "variance_retained deliberately.",
            max_pcs, k, 100 * explained, 100 * variance_retained,
            100 * (variance_retained - explained),
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
    # v0.36: add Higdon 2008's truncation-residual variance. The M − k
    # discarded components contribute reconstruction error to the mean
    # but, before this, contributed ZERO to the variance — so the
    # implausibility denominator ignored a bias the numerator carried,
    # systematically over-pruning the NROY whenever `max_pcs` truncated
    # below `variance_retained`. Estimate it as the per-column variance
    # of the training-set reconstruction residual.
    trunc_resid = Y_s - (W @ K)                       # (N, M), standardized
    trunc_var = trunc_resid.var(axis=0, ddof=1) if n_train > 1 else np.zeros(m)
    pred_var_s = pred_var_s + trunc_var               # broadcast over rows
    pred_mean = pred_mean_s * scale + center
    pred_var = pred_var_s * (scale**2)
    return pred_mean, pred_var


def _fit_predict_gbc_iqn(
    X: np.ndarray,
    Y: np.ndarray,
    X_pred: np.ndarray,
    *,
    hdim: int = 64,
    nh: int = 32,
    training_epochs: int = 3000,
    learning_rate: float = 1e-2,
    loss_weights: tuple[float, float, float] = (0.3, 0.3, 0.4),
    n_tau_samples_at_inference: int = 128,
    combine_with_pca: bool = False,
    pca_variance_retained: float = 0.99,
    pca_max_pcs: int = 5,
    pca_centering: str = "mean",
    pca_scaling: str = "per_moment_std",
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """v0.35 emulator: Implicit Quantile Network via gbc (Polson-Sokolov 2026).

    Trains one IQN per output column. IQN(x, τ) returns a ``(n, 2)``
    tensor whose column 0 is the mean estimate and column 1 is the
    τ-quantile of ``y|x``. Predictive mean comes from column 0 at
    ``τ = 0.5``; predictive variance from the empirical variance of
    column 1 across ``n_tau_samples_at_inference`` τ draws.

    Composition with PCA (``combine_with_pca=True``): PCA-decompose
    Y into ``k`` PCs first (per the v0.34 ``gp_basis_pca`` recipe),
    train ``k`` IQNs on the coefficients, reconstruct predictions
    through the basis. Best of both worlds: small IQN output space
    + non-Gaussian residual modeling.

    Reference
    ---------
    Polson & Sokolov 2026 "Generative Bayesian Computation" —
    ``github.com/VadimSokolov/gbc``. Supersedes the GP + DL
    dim-reduction approach of Schultz-Auld-Sokolov 2022
    (``arXiv:2203.04414``) on the sampling side.

    Parameters
    ----------
    X, Y, X_pred
        As for the other emulators.
    hdim, nh
        IQN hyperparameters (hidden width and number of τ-embedding
        harmonics).
    training_epochs, learning_rate
        Adam optimization; single-batch gradient descent per epoch
        (Y is O(hundreds) of samples for typical HM waves so batching
        adds no value).
    loss_weights
        ``(mse, quantile, pinball)`` weights passed to
        :meth:`gbc.IQN.loss_fn`. Defaults match gbc's default.
    n_tau_samples_at_inference
        Number of τ samples in (0, 1) used to estimate predictive
        variance. Larger → smoother variance estimate at the cost of
        one forward pass per τ.
    combine_with_pca
        When True, PCA-decompose Y into ``k`` coefficients and train
        one IQN per coefficient (small output space); reconstruct
        predictions through the basis. Substantially cheaper than
        one IQN per raw Y column when M is large.
    seed
        Optional torch seed for reproducibility of the IQN training.

    Requires the ``gbc`` package (Polson-Sokolov 2026). Import is
    guarded — if gbc isn't installed, the constructor raises with a
    hint to ``pip install polarisopt[bo]`` (torch prereq) and
    ``pip install gbc``.
    """
    try:
        import gbc
        import torch
        from torch.optim import Adam
    except ImportError as exc:
        raise ImportError(
            "gbc_iqn emulator requires the gbc + torch packages. "
            "Install with `pip install polarisopt[bo]` (for torch) and "
            "`pip install gbc`."
        ) from exc

    if Y.ndim != 2:
        raise ValueError(f"gbc_iqn: Y must be 2-D, got shape {Y.shape}")
    n_train, m = Y.shape
    if n_train < 2:
        raise ValueError(
            f"gbc_iqn: need at least 2 training samples, got {n_train}"
        )
    if n_tau_samples_at_inference < 2:
        raise ValueError(
            f"gbc_iqn.n_tau_samples_at_inference must be >= 2, "
            f"got {n_tau_samples_at_inference}"
        )
    if training_epochs < 1:
        raise ValueError(f"gbc_iqn.training_epochs must be >= 1, got {training_epochs}")

    if seed is not None:
        torch.manual_seed(int(seed))

    # Decide the output-space to train IQNs on: raw Y columns or PC
    # coefficients W. In both cases we standardize per-column so the
    # IQN sees zero-mean unit-scale targets (better optimization
    # behavior).
    if combine_with_pca:
        # Reuse gp_basis_pca's basis math; only the surrogate changes.
        Y_c, center, scale = _standardize(Y, centering=pca_centering, scaling=pca_scaling)
        U, S, Vt = np.linalg.svd(Y_c, full_matrices=False)
        total_var = float(np.sum(S**2))
        if total_var <= 0:
            return (
                np.broadcast_to(center, (X_pred.shape[0], m)).copy(),
                np.zeros((X_pred.shape[0], m), dtype=float),
            )
        cum_var = np.cumsum(S**2) / total_var
        k = int(np.searchsorted(cum_var, pca_variance_retained) + 1)
        k = max(1, min(k, pca_max_pcs, len(S)))
        K = Vt[:k, :]
        W_train = U[:, :k] * S[:k]                       # (N, k)
        train_targets = W_train                          # per-PC training
    else:
        # Standardize Y directly; k = M.
        Y_c, center, scale = _standardize(Y, centering=pca_centering, scaling=pca_scaling)
        train_targets = Y_c
        K = None

    n_outputs = train_targets.shape[1]
    log.info(
        "gbc_iqn: training %d IQN(s) (xdim=%d, hdim=%d, nh=%d, epochs=%d%s)",
        n_outputs, X.shape[1], hdim, nh, training_epochs,
        f", PCA {n_outputs}<-{m}" if combine_with_pca else "",
    )

    X_t = torch.as_tensor(X, dtype=torch.float32)
    X_pred_t = torch.as_tensor(X_pred, dtype=torch.float32)
    pred_mean_std = np.zeros((X_pred.shape[0], n_outputs), dtype=float)
    pred_var_std = np.zeros((X_pred.shape[0], n_outputs), dtype=float)
    rng_taus = np.random.default_rng(seed if seed is not None else 0)
    taus = rng_taus.uniform(0.01, 0.99, size=int(n_tau_samples_at_inference))

    for i in range(n_outputs):
        y_t = torch.as_tensor(train_targets[:, i], dtype=torch.float32)
        net = gbc.IQN(xdim=X.shape[1], hdim=int(hdim), nh=int(nh))
        opt = Adam(net.parameters(), lr=float(learning_rate))
        for _ in range(int(training_epochs)):
            opt.zero_grad()
            loss = net.loss_fn(X_t, y_t, w=tuple(loss_weights))
            loss.backward()
            opt.step()
        # Predictive mean: forward at tau=0.5, take column 0 (mean estimator).
        net.eval()
        with torch.no_grad():
            mean_pred = net.forward(X_pred_t, 0.5)[:, 0].cpu().numpy()
            # Predictive variance: quantile column across n_tau samples.
            q_samples = np.stack([
                net.forward(X_pred_t, float(t))[:, 1].cpu().numpy()
                for t in taus
            ], axis=1)                                    # (n_pred, n_tau)
            var_pred = q_samples.var(axis=1, ddof=1)
        pred_mean_std[:, i] = mean_pred
        pred_var_std[:, i] = var_pred

    # Reconstruct in raw Y space.
    if combine_with_pca:
        assert K is not None
        pred_mean_s = pred_mean_std @ K
        pred_var_s = pred_var_std @ (K**2)
    else:
        pred_mean_s = pred_mean_std
        pred_var_s = pred_var_std
    pred_mean = pred_mean_s * scale + center
    pred_var = pred_var_s * (scale**2)
    return pred_mean, pred_var


def _standardize(
    Y: np.ndarray, *, centering: str, scaling: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shared per-column center + scale for gp_basis_pca and gbc_iqn.

    Returns ``(Y_standardized, center, scale)``. Callers un-standardize
    predictions via ``pred_raw = pred_std * scale + center``.
    """
    m = Y.shape[1]
    if centering == "mean":
        center = Y.mean(axis=0)
    elif centering == "median":
        center = np.median(Y, axis=0)
    elif centering == "none":
        center = np.zeros(m, dtype=float)
    else:
        raise ValueError(f"centering must be mean|median|none, got {centering!r}")
    Y_c = Y - center
    if scaling == "per_moment_std":
        scale = Y_c.std(axis=0, ddof=1)
        scale = np.where(scale > 0, scale, 1.0)
    elif scaling in ("unit", "none"):
        scale = np.ones(m, dtype=float)
    else:
        raise ValueError(f"scaling must be per_moment_std|unit|none, got {scaling!r}")
    return Y_c / scale, center, scale
