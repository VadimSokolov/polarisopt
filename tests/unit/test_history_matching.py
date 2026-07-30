"""Unit tests for v0.31 history_matching phase (P3) + NROY export (P9)."""

from __future__ import annotations

import json
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
