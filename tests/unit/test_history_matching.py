"""Unit tests for v0.31 history_matching phase (P3) + NROY export (P9)."""

from __future__ import annotations

import contextlib
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from polarisopt.design import LHSDesign
from polarisopt.metrics import IdentityMetric, MomentSetMetric
from polarisopt.parameters import Parameter, ParameterSpace
from polarisopt.runners.local import LocalRunner
from polarisopt.samples.sample import Sample, SampleStatus
from polarisopt.samples.store import SampleStore
from polarisopt.simulator import MockSimulator
from polarisopt.studies.base import StudyContext
from polarisopt.studies.history_matching import (
    HistoryMatchingStudy,
    HistoryMatchingWavePhase,
)


def _write_csv(path: Path, header: str, rows: list[str]) -> None:
    path.write_text(header + "\n" + "\n".join(rows) + "\n")


def _mock_moment_metric(target_csv: Path) -> MomentSetMetric:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return MomentSetMetric(moments=[{
            "name": "shares",
            "source_sql": "SELECT purpose, mode, 1.0 AS share FROM Trip",
            "target": str(target_csv),
            "target_key_cols": ["purpose", "mode"],
            "target_value_col": "share",
            "obs_noise_std": 0.05,
            "model_discrepancy_std": 0.10,
        }], scalarize="none")


def _make_ctx(tmp_path: Path, metric, space) -> StudyContext:
    return StudyContext(
        name="hm",
        space=space,
        workspace=tmp_path,
        store=SampleStore.open(tmp_path / "store.db", "hm"),
        runner=LocalRunner(),
        simulator=MockSimulator(function="quadratic"),
        metric=metric,
        rng=np.random.default_rng(0),
        poll_interval=0.05,
        heartbeat_interval=0,
    )


def _seed_finished_samples(
    ctx: StudyContext, target_csv: Path, *, n: int = 30, phase: str = "wave-1",
) -> None:
    """Populate the store with FINISHED samples whose per-moment residuals
    depend on x[0] only — moment_set metric width from target_csv."""
    n_moments = sum(1 for _ in target_csv.open()) - 1
    rng = np.random.default_rng(1)
    for _ in range(n):
        x = rng.uniform(size=ctx.space.ndim)
        residuals = np.full(n_moments, 0.5 * (x[0] - 0.5)) + 0.01 * rng.standard_normal(n_moments)
        s = ctx.store.add(Sample(phase=phase, inputs=x))
        s.status = SampleStatus.FINISHED
        s.metric = residuals
        ctx.store.update(s)


def _space_3d() -> ParameterSpace:
    return ParameterSpace.from_iterable([
        Parameter("x0", "a.json", 0.0, 1.0),
        Parameter("x1", "a.json", 0.0, 1.0),
        Parameter("x2", "a.json", 0.0, 1.0),
    ])


def _finished(ctx: StudyContext, phase: str, x: np.ndarray, resid: np.ndarray) -> Sample:
    """Add one FINISHED sample with an explicit residual vector."""
    s = ctx.store.add(Sample(phase=phase, inputs=x))
    s.status = SampleStatus.FINISHED
    s.metric = resid
    return s


@contextlib.contextmanager
def caplog_at(level: int):
    """Capture polarisopt log records at `level` without pytest's caplog
    (these studies log through polarisopt.utils.logging)."""
    records: list[logging.LogRecord] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Collect(level=level)
    logger = logging.getLogger("polarisopt")
    prev_level, prev_prop = logger.level, logger.propagate
    logger.addHandler(handler)
    logger.setLevel(level)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)
        logger.propagate = prev_prop


def test_history_matching_writes_nroy_parquet(tmp_path: Path) -> None:
    target_csv = tmp_path / "t.csv"
    _write_csv(
        target_csv, "purpose,mode,share",
        ["HBW,auto,0.5", "HBW,transit,0.3", "HBW,walk,0.2"],
    )
    metric = _mock_moment_metric(target_csv)
    space = _space_3d()
    ctx = _make_ctx(tmp_path, metric, space)
    _seed_finished_samples(ctx, target_csv, n=30, phase="wave-1")
    phase = HistoryMatchingWavePhase(
        name="wave-1",
        warm_up=LHSDesign(n=1),
        emulator={"type": "gp_per_moment", "options": {}},
        implausibility={"type": "max", "cutoff": 3.0},
        moments_included=[],
        nroy_grid_size=256,
        output_dir=tmp_path / "hm-out",
    )
    study = HistoryMatchingStudy(ctx, phase)
    study.run()

    parquet_path = tmp_path / "hm-out" / "nroy_wave1.parquet"
    csv_path = tmp_path / "hm-out" / "nroy_wave1.csv"
    assert parquet_path.exists() or csv_path.exists(), (
        "NROY artifact not written to either parquet or CSV fallback"
    )
    import pandas as pd
    df = pd.read_parquet(parquet_path) if parquet_path.exists() else pd.read_csv(csv_path)
    for col in ["x0", "x1", "x2", "implausibility_max", "implausibility_second",
                "retained", "moment_residuals_json"]:
        assert col in df.columns, (col, df.columns.tolist())
    assert 128 <= len(df) <= 1024
    assert df["retained"].sum() > 0


def test_history_matching_rejects_non_moment_set(tmp_path: Path) -> None:
    space = _space_3d()
    metric = IdentityMetric(keys="value")
    ctx = _make_ctx(tmp_path, metric, space)
    with pytest.raises(TypeError, match="moment_set metric"):
        HistoryMatchingStudy(ctx, HistoryMatchingWavePhase(
            name="w", warm_up=LHSDesign(n=1),
            emulator={"type": "gp_per_moment", "options": {}},
            implausibility={"type": "max", "cutoff": 3.0},
            moments_included=[], nroy_grid_size=64,
            output_dir=tmp_path / "hm",
        ))


def test_history_matching_rejects_scalarized_moment_set(tmp_path: Path) -> None:
    """A moment_set metric with scalarize != none has no per-moment
    residuals to feed the emulator."""
    target_csv = tmp_path / "t.csv"
    _write_csv(target_csv, "purpose,mode,share", ["HBW,auto,0.5", "HBW,transit,0.5"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        metric = MomentSetMetric(
            moments=[{
                "name": "shares",
                "source_sql": "SELECT purpose, mode, 1.0 AS share FROM Trip",
                "target": str(target_csv),
                "target_key_cols": ["purpose", "mode"],
                "target_value_col": "share",
                "model_discrepancy_std": 0.05,
            }],
            scalarize="sum_squared_weighted",
        )
    space = _space_3d()
    ctx = _make_ctx(tmp_path, metric, space)
    with pytest.raises(ValueError, match="scalarize='none'"):
        HistoryMatchingStudy(ctx, HistoryMatchingWavePhase(
            name="w", warm_up=LHSDesign(n=1),
            emulator={"type": "gp_per_moment", "options": {}},
            implausibility={"type": "max", "cutoff": 3.0},
            moments_included=[], nroy_grid_size=64,
            output_dir=tmp_path / "hm",
        ))


def test_history_matching_retained_flag_uses_cutoff(tmp_path: Path) -> None:
    """Rows with implausibility_max < cutoff are retained; rows >= cutoff not."""
    target_csv = tmp_path / "t.csv"
    _write_csv(target_csv, "purpose,mode,share",
               ["HBW,auto,0.5", "HBW,transit,0.5"])
    metric = _mock_moment_metric(target_csv)
    space = _space_3d()
    ctx = _make_ctx(tmp_path, metric, space)
    _seed_finished_samples(ctx, target_csv, n=30, phase="wave-1")
    phase = HistoryMatchingWavePhase(
        name="wave-1",
        warm_up=LHSDesign(n=1),
        emulator={"type": "gp_per_moment", "options": {}},
        implausibility={"type": "max", "cutoff": 0.5},
        moments_included=[],
        nroy_grid_size=256,
        output_dir=tmp_path / "hm-tight",
    )
    HistoryMatchingStudy(ctx, phase).run()
    import pandas as pd
    parquet = tmp_path / "hm-tight" / "nroy_wave1.parquet"
    csv = tmp_path / "hm-tight" / "nroy_wave1.csv"
    df = pd.read_parquet(parquet) if parquet.exists() else pd.read_csv(csv)
    assert (df.loc[df["retained"], "implausibility_max"] < 0.5).all()
    assert (df.loc[~df["retained"], "implausibility_max"] >= 0.5).all()


def test_gp_basis_pca_emulator_default_options(tmp_path: Path) -> None:
    """v0.34 P3: gp_basis_pca produces the same-shape (mean, var)
    matrices as gp_per_moment. Sanity: predictions on training X
    should be close to training Y."""
    from polarisopt.studies.history_matching import _fit_predict_gp_basis_pca

    rng = np.random.default_rng(0)
    n, d, m = 25, 3, 8
    X = rng.uniform(size=(n, d))
    # Structured Y: two dominant modes (linear in x0 and x1), rest are noise.
    Y = np.column_stack([
        X[:, 0] + 0.01 * rng.standard_normal(n),
        X[:, 0] + 0.01 * rng.standard_normal(n),
        X[:, 1] + 0.01 * rng.standard_normal(n),
        X[:, 1] + 0.01 * rng.standard_normal(n),
        0.01 * rng.standard_normal(n),  # dead-ish
        0.01 * rng.standard_normal(n),
        0.01 * rng.standard_normal(n),
        0.01 * rng.standard_normal(n),
    ])
    assert Y.shape == (n, m)
    mean, var = _fit_predict_gp_basis_pca(X, Y, X, variance_retained=0.99, max_pcs=5)
    assert mean.shape == (n, m)
    assert var.shape == (n, m)
    # In-sample smoke check: predictions ≈ training data on the informative
    # columns. Loose atol because GP posterior mean at training points still
    # has non-zero shrinkage under the fitted noise term.
    np.testing.assert_allclose(mean[:, 0], Y[:, 0], atol=0.25)
    np.testing.assert_allclose(mean[:, 2], Y[:, 2], atol=0.25)
    # Variances non-negative.
    assert (var >= 0).all()


def test_gp_basis_pca_degenerate_all_constant_columns() -> None:
    """When every Y column is constant, gp_basis_pca short-circuits to
    a broadcast of the center + zero variance (no crash)."""
    from polarisopt.studies.history_matching import _fit_predict_gp_basis_pca

    X = np.random.default_rng(0).uniform(size=(10, 2))
    Y = np.full((10, 4), 0.42)
    mean, var = _fit_predict_gp_basis_pca(X, Y, X)
    np.testing.assert_allclose(mean, np.full_like(Y, 0.42))
    # LAPACK SVD of an all-zero matrix returns tiny non-zero singular
    # values (~1e-10). Variance can leak by the same magnitude when the
    # total_var short-circuit doesn't fire. Loose but non-trivial bound.
    assert (var < 1e-6).all()


def test_gp_basis_pca_rejects_bad_variance_retained() -> None:
    from polarisopt.studies.history_matching import _fit_predict_gp_basis_pca

    X = np.random.default_rng(0).uniform(size=(5, 2))
    Y = np.random.default_rng(0).uniform(size=(5, 3))
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="variance_retained"):
            _fit_predict_gp_basis_pca(X, Y, X, variance_retained=bad)


def test_gp_basis_pca_rejects_bad_centering_scaling() -> None:
    from polarisopt.studies.history_matching import _fit_predict_gp_basis_pca

    X = np.random.default_rng(0).uniform(size=(5, 2))
    Y = np.random.default_rng(0).uniform(size=(5, 3))
    with pytest.raises(ValueError, match="centering"):
        _fit_predict_gp_basis_pca(X, Y, X, centering="quantile")
    with pytest.raises(ValueError, match="scaling"):
        _fit_predict_gp_basis_pca(X, Y, X, scaling="minmax")


def test_history_matching_dispatches_on_emulator_type(tmp_path: Path) -> None:
    """The phase config's ``emulator.type`` picks the emulator; smoke test
    that gp_basis_pca produces an artifact end-to-end."""
    target_csv = tmp_path / "t.csv"
    _write_csv(
        target_csv, "purpose,mode,share",
        ["HBW,auto,0.5", "HBW,transit,0.3", "HBW,walk,0.2"],
    )
    metric = _mock_moment_metric(target_csv)
    space = _space_3d()
    ctx = _make_ctx(tmp_path, metric, space)
    _seed_finished_samples(ctx, target_csv, n=30, phase="wave-1")
    HistoryMatchingStudy(ctx, HistoryMatchingWavePhase(
        name="wave-1", warm_up=LHSDesign(n=1),
        emulator={
            "type": "gp_basis_pca",
            "options": {"variance_retained": 0.99, "max_pcs": 3},
        },
        implausibility={"type": "max", "cutoff": 3.0},
        moments_included=[], nroy_grid_size=128,
        output_dir=tmp_path / "hm-pca",
    )).run()
    import pandas as pd
    parquet = tmp_path / "hm-pca" / "nroy_wave1.parquet"
    csv = tmp_path / "hm-pca" / "nroy_wave1.csv"
    df = pd.read_parquet(parquet) if parquet.exists() else pd.read_csv(csv)
    assert "implausibility_max" in df.columns
    assert len(df) >= 64
    # PCA emulator returns finite implausibilities.
    assert np.isfinite(df["implausibility_max"]).all()


def test_gbc_iqn_emulator_smoke(tmp_path: Path) -> None:
    """v0.35 P4: gbc_iqn returns finite (mean, var) matrices of the
    same shape as its inputs. Smoke test only — IQNs aren't going
    to beat a GP on 25 samples, but they should not crash."""
    pytest.importorskip("gbc")
    from polarisopt.studies.history_matching import _fit_predict_gbc_iqn

    rng = np.random.default_rng(0)
    n, d, m = 25, 3, 4
    X = rng.uniform(size=(n, d))
    Y = np.column_stack([X[:, 0], X[:, 0], X[:, 1], X[:, 1]]) + 0.01 * rng.standard_normal((n, m))
    mean, var = _fit_predict_gbc_iqn(
        X, Y, X,
        hdim=16, nh=8,
        training_epochs=50,          # kept low for test wall time
        n_tau_samples_at_inference=8,
        seed=0,
    )
    assert mean.shape == (n, m)
    assert var.shape == (n, m)
    assert np.isfinite(mean).all()
    assert np.isfinite(var).all()
    assert (var >= 0).all()


def test_gbc_iqn_with_pca_composition_smoke(tmp_path: Path) -> None:
    """combine_with_pca=True trains IQNs on PC coefficients and
    reconstructs Y via the basis. Cuts n_outputs to k."""
    pytest.importorskip("gbc")
    from polarisopt.studies.history_matching import _fit_predict_gbc_iqn

    rng = np.random.default_rng(0)
    n, d, m = 30, 3, 10
    X = rng.uniform(size=(n, d))
    base = np.column_stack([X[:, 0], X[:, 1]])
    Y = base @ rng.uniform(size=(2, m)) + 0.01 * rng.standard_normal((n, m))
    mean, var = _fit_predict_gbc_iqn(
        X, Y, X,
        hdim=16, nh=8,
        training_epochs=30,
        n_tau_samples_at_inference=8,
        combine_with_pca=True,
        pca_variance_retained=0.95,
        pca_max_pcs=3,
        seed=0,
    )
    assert mean.shape == (n, m)
    assert var.shape == (n, m)
    assert np.isfinite(mean).all()


def test_gbc_iqn_rejects_bad_inputs() -> None:
    pytest.importorskip("gbc")
    from polarisopt.studies.history_matching import _fit_predict_gbc_iqn

    X = np.random.default_rng(0).uniform(size=(5, 2))
    Y = np.random.default_rng(0).uniform(size=(5, 3))
    with pytest.raises(ValueError, match="training_epochs"):
        _fit_predict_gbc_iqn(X, Y, X, training_epochs=0)
    with pytest.raises(ValueError, match="n_tau_samples_at_inference"):
        _fit_predict_gbc_iqn(X, Y, X, n_tau_samples_at_inference=1)


def test_history_matching_dispatches_gbc_iqn(tmp_path: Path) -> None:
    """Full-phase smoke: `emulator.type: gbc_iqn` produces an NROY artifact."""
    pytest.importorskip("gbc")
    target_csv = tmp_path / "t.csv"
    _write_csv(
        target_csv, "purpose,mode,share",
        ["HBW,auto,0.5", "HBW,transit,0.3", "HBW,walk,0.2"],
    )
    metric = _mock_moment_metric(target_csv)
    space = _space_3d()
    ctx = _make_ctx(tmp_path, metric, space)
    _seed_finished_samples(ctx, target_csv, n=30, phase="wave-1")
    HistoryMatchingStudy(ctx, HistoryMatchingWavePhase(
        name="wave-1", warm_up=LHSDesign(n=1),
        emulator={
            "type": "gbc_iqn",
            "options": {
                "hdim": 16, "nh": 8,
                "training_epochs": 30,
                "n_tau_samples_at_inference": 8,
                "seed": 0,
            },
        },
        implausibility={"type": "max", "cutoff": 3.0},
        moments_included=[], nroy_grid_size=64,
        output_dir=tmp_path / "hm-iqn",
    )).run()
    import pandas as pd
    parquet = tmp_path / "hm-iqn" / "nroy_wave1.parquet"
    csv = tmp_path / "hm-iqn" / "nroy_wave1.csv"
    df = pd.read_parquet(parquet) if parquet.exists() else pd.read_csv(csv)
    assert "implausibility_max" in df.columns
    assert np.isfinite(df["implausibility_max"]).all()


def test_history_matching_rejects_unknown_emulator_type(tmp_path: Path) -> None:
    target_csv = tmp_path / "t.csv"
    _write_csv(target_csv, "purpose,mode,share", ["HBW,auto,0.5", "HBW,transit,0.5"])
    metric = _mock_moment_metric(target_csv)
    space = _space_3d()
    ctx = _make_ctx(tmp_path, metric, space)
    _seed_finished_samples(ctx, target_csv, n=15, phase="wave-1")
    with pytest.raises(ValueError, match="unknown emulator type"):
        HistoryMatchingStudy(ctx, HistoryMatchingWavePhase(
            name="wave-1", warm_up=LHSDesign(n=1),
            emulator={"type": "kernel_ridge", "options": {}},
            implausibility={"type": "max", "cutoff": 3.0},
            moments_included=[], nroy_grid_size=64,
            output_dir=tmp_path / "hm-badtype",
        )).run()


def test_history_matching_moment_residuals_json_roundtrip(tmp_path: Path) -> None:
    target_csv = tmp_path / "t.csv"
    _write_csv(target_csv, "purpose,mode,share",
               ["HBW,auto,0.5", "HBW,transit,0.5"])
    metric = _mock_moment_metric(target_csv)
    space = _space_3d()
    ctx = _make_ctx(tmp_path, metric, space)
    _seed_finished_samples(ctx, target_csv, n=30, phase="wave-1")
    HistoryMatchingStudy(ctx, HistoryMatchingWavePhase(
        name="wave-1", warm_up=LHSDesign(n=1),
        emulator={"type": "gp_per_moment", "options": {}},
        implausibility={"type": "max", "cutoff": 3.0},
        moments_included=[], nroy_grid_size=128,
        output_dir=tmp_path / "hm-json",
    )).run()
    import pandas as pd
    parquet = tmp_path / "hm-json" / "nroy_wave1.parquet"
    csv = tmp_path / "hm-json" / "nroy_wave1.csv"
    df = pd.read_parquet(parquet) if parquet.exists() else pd.read_csv(csv)
    parsed = json.loads(df["moment_residuals_json"].iloc[0])
    assert len(parsed) == 2  # two moment columns from the 2-row target
    for k in parsed:
        assert isinstance(parsed[k], float)


# ===== v0.36 critical-review regression tests =====


def test_hm_rejects_md_auto_instead_of_silent_empty_nroy(tmp_path: Path) -> None:
    """v0.36 regression: model_discrepancy_std='auto' stores NaN. NaN
    propagates through denom; np.maximum(nan, 1e-30) returns nan (it is
    NOT a NaN clamp), so implausibility is all-NaN, `nan < cutoff` is
    False everywhere, and the wave used to write a 0/N-retained NROY
    indistinguishable from a real 'model cannot match targets' result —
    the exact Vernon 2010 §3.5 failure this library exists to prevent."""
    target_csv = tmp_path / "t.csv"
    _write_csv(target_csv, "purpose,mode,share", ["HBW,auto,0.5", "HBW,transit,0.5"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        metric = MomentSetMetric(moments=[{
            "name": "shares",
            "source_sql": "SELECT purpose, mode, 1.0 AS share FROM Trip",
            "target": str(target_csv),
            "target_key_cols": ["purpose", "mode"],
            "target_value_col": "share",
            "obs_noise_std": 0.005,
            "model_discrepancy_std": "auto",
        }], scalarize="none")
    ctx = _make_ctx(tmp_path, metric, _space_3d())
    with pytest.raises(ValueError, match="cannot run with model_discrepancy_std='auto'"):
        HistoryMatchingStudy(ctx, HistoryMatchingWavePhase(
            name="w", warm_up=LHSDesign(n=1),
            emulator={"type": "gp_per_moment", "options": {}},
            implausibility={"type": "max", "cutoff": 3.0},
            moments_included=[], nroy_grid_size=64,
            output_dir=tmp_path / "hm",
        ))


def test_hm_dead_column_excluded_not_silently_wiping_nroy(tmp_path: Path) -> None:
    """v0.36 regression: a moment column with a constant residual across
    the whole design cannot discriminate between any two theta. Before
    the fix it kept its constant (often large) mean with zero variance,
    giving a constant huge implausibility that emptied the NROY for a
    reason unrelated to theta. It must be excluded, with a warning."""
    import logging

    target_csv = tmp_path / "t.csv"
    _write_csv(target_csv, "purpose,mode,share",
               ["HBW,auto,0.5", "HBW,transit,0.5"])
    metric = _mock_moment_metric(target_csv)
    space = _space_3d()
    ctx = _make_ctx(tmp_path, metric, space)
    # Column 0 varies with x0; column 1 is pinned at a LARGE constant
    # residual (a mode the sim never produces -> residual == -target).
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.uniform(size=3)
        ctx.store.update(_finished(ctx, "wave-1", x, np.array([0.5 * (x[0] - 0.5), -0.5])))
    phase = HistoryMatchingWavePhase(
        name="wave-1", warm_up=LHSDesign(n=1),
        emulator={"type": "gp_per_moment", "options": {}},
        implausibility={"type": "max", "cutoff": 3.0, "include_prior_terms": False},
        moments_included=[], nroy_grid_size=128,
        output_dir=tmp_path / "hm-dead",
    )
    with caplog_at(logging.WARNING) as records:
        HistoryMatchingStudy(ctx, phase).run()
    assert any("constant residual" in r.message for r in records), [
        r.message for r in records
    ]
    import pandas as pd
    parquet = tmp_path / "hm-dead" / "nroy_wave1.parquet"
    csv = tmp_path / "hm-dead" / "nroy_wave1.csv"
    df = pd.read_parquet(parquet) if parquet.exists() else pd.read_csv(csv)
    # With the dead column excluded, the live column can still retain
    # points. Before the fix this was 0.
    assert df["retained"].sum() > 0, "dead column wiped the NROY"


def test_hm_wave_index_names_the_artifact(tmp_path: Path) -> None:
    """v0.36 regression: wave_index was hardcoded to 1, so chaining
    hm-wave-1/2/3 (the documented pattern) had every phase overwrite the
    same nroy_wave1.parquet."""
    target_csv = tmp_path / "t.csv"
    _write_csv(target_csv, "purpose,mode,share",
               ["HBW,auto,0.5", "HBW,transit,0.5"])
    metric = _mock_moment_metric(target_csv)
    ctx = _make_ctx(tmp_path, metric, _space_3d())
    _seed_finished_samples(ctx, target_csv, n=20, phase="wave-2")
    HistoryMatchingStudy(ctx, HistoryMatchingWavePhase(
        name="wave-2", warm_up=LHSDesign(n=1),
        emulator={"type": "gp_per_moment", "options": {}},
        implausibility={"type": "max", "cutoff": 3.0},
        moments_included=[], nroy_grid_size=64,
        output_dir=tmp_path / "hm-w2", wave_index=2,
    )).run()
    out = tmp_path / "hm-w2"
    assert (out / "nroy_wave2.parquet").exists() or (out / "nroy_wave2.csv").exists()
    assert not (out / "nroy_wave1.parquet").exists()
    assert not (out / "nroy_wave1.csv").exists()


def test_hm_rejects_unknown_moments_included(tmp_path: Path) -> None:
    """v0.36: a typo'd moment name used to yield an empty active_cols and
    surface only as 'zero-size array to reduction' AFTER the whole wave
    of simulations had run."""
    target_csv = tmp_path / "t.csv"
    _write_csv(target_csv, "purpose,mode,share",
               ["HBW,auto,0.5", "HBW,transit,0.5"])
    metric = _mock_moment_metric(target_csv)
    ctx = _make_ctx(tmp_path, metric, _space_3d())
    _seed_finished_samples(ctx, target_csv, n=15, phase="wave-1")
    with pytest.raises(ValueError, match="moments_included names not in the metric"):
        HistoryMatchingStudy(ctx, HistoryMatchingWavePhase(
            name="wave-1", warm_up=LHSDesign(n=1),
            emulator={"type": "gp_per_moment", "options": {}},
            implausibility={"type": "max", "cutoff": 3.0},
            moments_included=["sharez"],  # typo
            nroy_grid_size=64, output_dir=tmp_path / "hm-typo",
        )).run()


def test_hm_gp_per_moment_rejects_stray_options(tmp_path: Path) -> None:
    """v0.36: emu_opts was silently dropped for gp_per_moment (the
    DEFAULT emulator) while the same typo raised TypeError under the
    other two."""
    target_csv = tmp_path / "t.csv"
    _write_csv(target_csv, "purpose,mode,share",
               ["HBW,auto,0.5", "HBW,transit,0.5"])
    metric = _mock_moment_metric(target_csv)
    ctx = _make_ctx(tmp_path, metric, _space_3d())
    _seed_finished_samples(ctx, target_csv, n=15, phase="wave-1")
    with pytest.raises(ValueError, match="takes no options"):
        HistoryMatchingStudy(ctx, HistoryMatchingWavePhase(
            name="wave-1", warm_up=LHSDesign(n=1),
            emulator={"type": "gp_per_moment", "options": {"variance_retained": 0.9}},
            implausibility={"type": "max", "cutoff": 3.0},
            moments_included=[], nroy_grid_size=64,
            output_dir=tmp_path / "hm-opts",
        )).run()


def test_include_prior_terms_actually_constrains_nroy(tmp_path: Path) -> None:
    """v0.36: include_prior_terms was documented (schema default True,
    yaml-reference) but NEVER read — priors silently failed to constrain
    the NROY. With a tight prior the retained set must shrink."""
    from polarisopt.parameters import GaussianPrior, Parameter, ParameterSpace

    target_csv = tmp_path / "t.csv"
    _write_csv(target_csv, "purpose,mode,share",
               ["HBW,auto,0.5", "HBW,transit,0.5"])
    # Tight prior on x0 centred at 0.5; the box is [0,1].
    space = ParameterSpace.from_iterable([
        Parameter("x0", "a.json", 0.0, 1.0, prior=GaussianPrior(mean=0.5, std=0.02)),
        Parameter("x1", "a.json", 0.0, 1.0),
        Parameter("x2", "a.json", 0.0, 1.0),
    ])

    def _run(include: bool, tag: str) -> int:
        metric = _mock_moment_metric(target_csv)
        ctx = _make_ctx(tmp_path / tag, metric, space)
        _seed_finished_samples(ctx, target_csv, n=20, phase="w")
        HistoryMatchingStudy(ctx, HistoryMatchingWavePhase(
            name="w", warm_up=LHSDesign(n=1),
            emulator={"type": "gp_per_moment", "options": {}},
            implausibility={
                "type": "max", "cutoff": 3.0, "include_prior_terms": include,
            },
            moments_included=[], nroy_grid_size=256,
            output_dir=tmp_path / tag / "out",
        )).run()
        import pandas as pd
        p = tmp_path / tag / "out" / "nroy_wave1.parquet"
        c = tmp_path / tag / "out" / "nroy_wave1.csv"
        df = pd.read_parquet(p) if p.exists() else pd.read_csv(c)
        return int(df["retained"].sum())

    without = _run(False, "no-prior")
    with_prior = _run(True, "with-prior")
    assert with_prior < without, (
        f"prior terms did not constrain the NROY (with={with_prior}, "
        f"without={without}) — include_prior_terms is being ignored"
    )


def test_gp_basis_pca_reports_retained_pc_count() -> None:
    """v0.36: replaces the previous vacuous test whose NAME claimed PC
    reduction but whose body only asserted finiteness. 8 columns that are
    noisy linear combinations of 2 latent factors must compress to a
    small k."""
    from polarisopt.studies.history_matching import _fit_predict_gp_basis_pca

    rng = np.random.default_rng(1)
    n, d, m = 20, 3, 8
    X = rng.uniform(size=(n, d))
    base = np.column_stack([X[:, 0], X[:, 1]])
    Y = base @ rng.uniform(size=(2, m)) + 0.001 * rng.standard_normal((n, m))
    n_pcs: list[int] = []
    mean, var = _fit_predict_gp_basis_pca(
        X, Y, X, variance_retained=0.99, max_pcs=5, n_pcs_out=n_pcs,
    )
    assert n_pcs, "emulator did not report its PC count"
    # Rank-2 signal -> far fewer PCs than the 8 raw columns.
    assert n_pcs[0] <= 3, f"expected strong compression, got k={n_pcs[0]} of {m}"
    assert mean.shape == (n, m)
    assert np.isfinite(var).all()


def test_gp_basis_pca_truncation_adds_variance() -> None:
    """v0.36: the M-k discarded components contribute reconstruction bias
    to the mean; before the fix they contributed ZERO variance, so the
    implausibility denominator ignored a bias the numerator carried and
    the NROY was over-pruned. Hard-truncating must inflate variance."""
    from polarisopt.studies.history_matching import _fit_predict_gp_basis_pca

    rng = np.random.default_rng(3)
    n, d, m = 25, 3, 10
    X = rng.uniform(size=(n, d))
    # Genuinely high-rank Y so truncation discards real signal.
    Y = X @ rng.uniform(size=(d, m)) + 0.3 * rng.standard_normal((n, m))
    _, var_trunc = _fit_predict_gp_basis_pca(
        X, Y, X, variance_retained=0.999, max_pcs=1,
    )
    _, var_full = _fit_predict_gp_basis_pca(
        X, Y, X, variance_retained=0.999, max_pcs=10,
    )
    assert var_trunc.mean() > var_full.mean(), (
        "hard truncation did not increase predictive variance — the "
        "Higdon residual term is missing"
    )
