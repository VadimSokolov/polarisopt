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


# ---------- max_retries ----------


class _FlakySimulator(MockSimulator):
    """Mock that fails its first N preparations per sample id, then succeeds.

    Lets us assert end-to-end that a transient failure triggers the v0.12
    auto-retry path and the sample eventually FINISHES with retry_count
    populated in extra.
    """

    def __init__(self, *, fail_first_n: int, **kw) -> None:
        super().__init__(**kw)
        self.fail_first_n = fail_first_n
        self.attempts_by_sample: dict[int, int] = {}

    def prepare(self, sample, space, workspace):  # type: ignore[override]
        attempt = self.attempts_by_sample.get(sample.id or -1, 0)
        self.attempts_by_sample[sample.id or -1] = attempt + 1
        spec = super().prepare(sample, space, workspace)
        if attempt < self.fail_first_n:
            # Replace command with a /bin/false so the runner reports FAILED.
            spec.command = "/bin/false"
        return spec


def _ctx_with_retries(
    tmp_path: Path, space: ParameterSpace, max_retries: int, fail_first_n: int,
) -> tuple[StudyContext, _FlakySimulator]:
    sim = _FlakySimulator(function="branin", fail_first_n=fail_first_n)
    ctx = StudyContext(
        name="retry",
        space=space,
        workspace=tmp_path,
        store=SampleStore.open(tmp_path / "store.db", "study"),
        runner=LocalRunner(),
        simulator=sim,
        metric=IdentityMetric(keys="value"),
        rng=np.random.default_rng(0),
        poll_interval=0.05,
        max_retries=max_retries,
    )
    return ctx, sim


def test_max_retries_zero_no_auto_retry(tmp_path: Path, space: ParameterSpace) -> None:
    """Default behavior: max_retries=0 means a FAILED sample stays FAILED."""
    ctx, sim = _ctx_with_retries(tmp_path, space, max_retries=0, fail_first_n=1)
    study = StaticDesignStudy(ctx, ManualDesign(points=[[0.5, 0.5]]), phase_name="retry")
    samples = study.run()
    assert len(samples) == 1
    assert samples[0].status is SampleStatus.FAILED
    assert sim.attempts_by_sample[samples[0].id] == 1  # only one attempt


def test_max_retries_recovers_transient_failure(
    tmp_path: Path, space: ParameterSpace,
) -> None:
    """max_retries=2 lets a sample that fails once succeed on retry."""
    ctx, sim = _ctx_with_retries(tmp_path, space, max_retries=2, fail_first_n=1)
    study = StaticDesignStudy(ctx, ManualDesign(points=[[0.5, 0.5]]), phase_name="retry")
    samples = study.run()
    assert len(samples) == 1
    sample = samples[0]
    assert sample.status is SampleStatus.FINISHED, sample.message
    assert sample.metric is not None
    # Two prepare() calls: one that emitted /bin/false → FAILED, one that succeeded.
    assert sim.attempts_by_sample[sample.id] == 2
    assert sample.extra.get("retry_count") == 1


class _PreWrittenOutputsSimulator(MockSimulator):
    """Mock variant that writes outputs.json synchronously in prepare()
    rather than via a subprocess. Lets the v0.15 in-flight disk recovery
    test run deterministically (no race against subprocess startup).
    """

    def prepare(self, sample, space, workspace):  # type: ignore[override]
        spec = super().prepare(sample, space, workspace)
        # Compute the benchmark value in-process and write outputs.json now.
        import json as _json

        from polarisopt.simulator.benchmarks import BENCHMARKS
        value = BENCHMARKS[self.function](sample.inputs)
        (workspace / self.OUTPUT_FILE).write_text(
            _json.dumps({"value": float(value), "runtime_s": 0.0})
        )
        # Command is a no-op — outputs already on disk.
        spec.command = "true"
        return spec


def test_inflight_disk_recovery_on_unknown(
    tmp_path: Path, space: ParameterSpace, monkeypatch,
) -> None:
    """v0.15 in-flight disk reconcile: when runner.status returns UNKNOWN
    for a sample whose outputs are already on disk, the master harvests
    them inline (no resume restart needed).
    """
    from polarisopt.runners.base import JobStatus
    from polarisopt.runners.local import LocalRunner

    sim = _PreWrittenOutputsSimulator(function="quadratic")
    ctx = StudyContext(
        name="zombie",
        space=space,
        workspace=tmp_path,
        store=SampleStore.open(tmp_path / "store.db", "study"),
        runner=LocalRunner(),
        simulator=sim,
        metric=IdentityMetric(keys="value"),
        rng=np.random.default_rng(0),
        poll_interval=0.05,
        orphan_threshold=3,
    )

    # Make runner.status always return UNKNOWN — simulating PBS losing
    # the jobid from accounting.
    def _always_unknown(self, job):  # noqa: ARG001
        job.status = JobStatus.UNKNOWN
        return job

    monkeypatch.setattr(LocalRunner, "status", _always_unknown)

    study = StaticDesignStudy(ctx, ManualDesign(points=[[0.5, 0.5]]), phase_name="zombie")
    samples = study.run()
    # Sample finishes via the in-flight disk-recovery path, not the
    # orphan-threshold-marks-FAILED path.
    assert len(samples) == 1
    sample = samples[0]
    assert sample.status is SampleStatus.FINISHED, (sample.status, sample.message)
    assert sample.metric is not None


def test_max_retries_exhausts_budget(
    tmp_path: Path, space: ParameterSpace,
) -> None:
    """A persistently failing sample exhausts the retry budget and stays FAILED."""
    ctx, sim = _ctx_with_retries(tmp_path, space, max_retries=2, fail_first_n=99)
    study = StaticDesignStudy(ctx, ManualDesign(points=[[0.5, 0.5]]), phase_name="retry")
    samples = study.run()
    assert len(samples) == 1
    sample = samples[0]
    assert sample.status is SampleStatus.FAILED
    # 1 original + 2 retries = 3 prepare() calls total.
    assert sim.attempts_by_sample[sample.id] == 3
    assert sample.extra.get("retry_count") == 2
    # The retry audit trail lives in extra (message gets overwritten by
    # each new FAILED transition).
    log = sample.extra.get("retry_log") or []
    assert len(log) == 2
    assert log[0]["attempt"] == 1 and log[0]["max"] == 2
    assert log[1]["attempt"] == 2 and log[1]["max"] == 2
