from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from polarisopt.design import LHSDesign, ManualDesign
from polarisopt.metrics import IdentityMetric
from polarisopt.parameters import Parameter, ParameterSpace
from polarisopt.runners.local import LocalRunner
from polarisopt.samples.sample import SampleStatus
from polarisopt.samples.store import SampleStore
from polarisopt.simulator import MockSimulator
from polarisopt.studies.base import StudyContext
from polarisopt.studies.static import StaticDesignStudy


@pytest.fixture
def space() -> ParameterSpace:
    return ParameterSpace.from_iterable(
        [Parameter("x1", "a.json", -5.0, 10.0), Parameter("x2", "a.json", 0.0, 15.0)]
    )


def _ctx(tmp_path: Path, space: ParameterSpace) -> StudyContext:
    return StudyContext(
        name="screen",
        space=space,
        workspace=tmp_path,
        store=SampleStore.open(tmp_path / "store.db", "study"),
        runner=LocalRunner(),
        simulator=MockSimulator(function="branin"),
        metric=IdentityMetric(keys="value"),
        rng=np.random.default_rng(0),
        poll_interval=0.05,
    )


def test_static_phase_evaluates_full_batch(tmp_path: Path, space: ParameterSpace) -> None:
    ctx = _ctx(tmp_path, space)
    study = StaticDesignStudy(ctx, LHSDesign(n=4), phase_name="screen")
    samples = study.run()
    assert len(samples) == 4
    for s in samples:
        assert s.status is SampleStatus.FINISHED, f"sample {s.id}: {s.status} {s.message}"
        assert s.metric is not None and s.metric.shape == (1,)
        assert s.folder is not None and (s.folder / MockSimulator.OUTPUT_FILE).exists()


def test_static_phase_branin_minimum(tmp_path: Path, space: ParameterSpace) -> None:
    """Evaluate the three known global minima of Branin and confirm the metric matches."""
    ctx = _ctx(tmp_path, space)
    minima = [[-np.pi, 12.275], [np.pi, 2.275], [9.42478, 2.475]]
    study = StaticDesignStudy(ctx, ManualDesign(points=minima), phase_name="screen")
    samples = study.run()
    for s in samples:
        assert s.status is SampleStatus.FINISHED
        assert float(s.metric[0]) == pytest.approx(0.397887, abs=1e-3)


def test_static_phase_records_status_in_store(tmp_path: Path, space: ParameterSpace) -> None:
    ctx = _ctx(tmp_path, space)
    study = StaticDesignStudy(ctx, LHSDesign(n=3), phase_name="screen")
    study.run()
    finished = ctx.store.list(status=SampleStatus.FINISHED, phase="screen")
    assert len(finished) == 3
    df = ctx.store.to_dataframe()
    assert len(df) == 3
    assert (df["status"] == "finished").all()


def test_static_phase_resumes_pending_without_regenerating(tmp_path: Path, space: ParameterSpace) -> None:
    ctx = _ctx(tmp_path, space)

    # Seed the store with PENDING samples ourselves.
    from polarisopt.samples.sample import Sample

    pts = [[1.0, 1.0], [2.0, 2.0]]
    for row in pts:
        ctx.store.add(Sample(phase="screen", inputs=np.asarray(row)))
    # Now run; because PENDING rows already exist, design.generate must NOT be called.

    class _ExplodingDesign(LHSDesign):
        def generate(self, *args, **kwargs):  # type: ignore[override]
            raise AssertionError("design.generate should not be called when PENDING rows exist")

    study = StaticDesignStudy(ctx, _ExplodingDesign(n=5), phase_name="screen")
    samples = study.run()
    assert len(samples) == 2
    assert all(s.status is SampleStatus.FINISHED for s in samples)
