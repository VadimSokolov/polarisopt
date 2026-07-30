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
        """Fit GP per moment, evaluate on a Sobol grid, compute
        max-implausibility per grid point, return a DataFrame."""
        import pandas as pd
        from scipy.stats import qmc

        from polarisopt.surrogates.gp import GPSurrogate

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

        # Per-column GP predictions on the grid.
        pred_mean = np.zeros((n_grid, len(active_cols)), dtype=float)
        pred_var = np.zeros((n_grid, len(active_cols)), dtype=float)
        col_obs_std = np.zeros(len(active_cols), dtype=float)
        col_md_std = np.zeros(len(active_cols), dtype=float)
        col_names = []
        for k, j in enumerate(active_cols):
            y_col = Y[:, j : j + 1]
            if float(np.ptp(y_col)) == 0.0:
                # Degenerate column — no signal. GP would still fit but the
                # implausibility contribution is dominated by the noise
                # denominator; skip cleanly.
                pred_mean[:, k] = float(y_col[0, 0])
                pred_var[:, k] = 0.0
            else:
                gp = GPSurrogate()
                gp.fit(X, y_col)
                m, v = gp.predict(grid)
                pred_mean[:, k] = m.reshape(-1)
                pred_var[:, k] = v.reshape(-1)
            col_obs_std[k] = obs_std[j]
            col_md_std[k] = md_std[j]
            col_names.append(f"resid[{col_to_moment[j]}][{j}]")

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
