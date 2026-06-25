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


def test_finalize_terminal_sample_is_idempotent(
    tmp_path: Path, space: ParameterSpace,
) -> None:
    """v0.17.1 regression: calling _finalize_terminal_sample twice on the
    same sample must be a no-op the second time. Without the idempotence
    guard the second call re-runs collect_output (against a workspace
    that cleanup_on_success may have deleted) and DOWNGRADES the sample
    from FINISHED to FAILED with "scenario file missing".

    Reproduces the bug the DFW DOE agent hit in iter 2 of a BO loop:
    12 of ~32 samples ended up marked FAILED with empty metric, but
    their results were on VMS and the binary had run successfully.
    """
    from polarisopt.runners.base import Job, JobSpec, JobStatus

    sim = MockSimulator(function="quadratic")
    ctx = StudyContext(
        name="idempotence",
        space=space,
        workspace=tmp_path,
        store=SampleStore.open(tmp_path / "store.db", "study"),
        runner=LocalRunner(),
        simulator=sim,
        metric=IdentityMetric(keys="value"),
        rng=np.random.default_rng(0),
        poll_interval=0.05,
        heartbeat_interval=0,
    )
    # Run one sample through to FINISHED.
    study = StaticDesignStudy(
        ctx, ManualDesign(points=[[0.5, 0.5]]), phase_name="idempotence",
    )
    samples = study.run()
    [s] = samples
    assert s.status is SampleStatus.FINISHED
    finished_metric = s.metric

    # Now simulate the bug: delete the workspace (mimicking
    # cleanup_on_success), then call _finalize_terminal_sample again with
    # job.status=FINISHED. Pre-v0.17.1 this would re-run collect_output,
    # fail with "scenario file missing", and DOWNGRADE the sample to
    # FAILED. With the guard the call is a no-op and FINISHED is preserved.
    import shutil
    shutil.rmtree(s.folder)
    fake_job = Job(
        spec=JobSpec(name="x", command="", cwd=s.folder),
        task_id="reused",
        status=JobStatus.FINISHED,
    )
    deltas: dict[str, int] = {"FINISHED": 0, "FAILED": 0, "RECOVERED": 0}
    study._finalize_terminal_sample(s, fake_job, deltas)

    # FINISHED stays FINISHED — no downgrade, no clobbered metric.
    assert s.status is SampleStatus.FINISHED, s.message
    assert s.metric is not None
    assert (s.metric == finished_metric).all()
    # The duplicate call did not count toward the "actually transitioned"
    # heartbeat delta.
    assert deltas["FAILED"] == 0
    assert deltas["FINISHED"] == 0


def test_finalize_terminal_sample_preserves_finished_when_re_collect_raises(
    tmp_path: Path, space: ParameterSpace, monkeypatch,
) -> None:
    """Defense-in-depth: even if the idempotence guard is somehow
    bypassed and collect_output raises, a sample already FINISHED in
    the store must not be downgraded to FAILED. The exception handler
    re-reads the store row and refuses to overwrite a FINISHED record.
    """
    from polarisopt.runners.base import Job, JobSpec, JobStatus

    sim = MockSimulator(function="quadratic")
    ctx = StudyContext(
        name="defense",
        space=space,
        workspace=tmp_path,
        store=SampleStore.open(tmp_path / "store.db", "study"),
        runner=LocalRunner(),
        simulator=sim,
        metric=IdentityMetric(keys="value"),
        rng=np.random.default_rng(0),
        poll_interval=0.05,
        heartbeat_interval=0,
    )
    study = StaticDesignStudy(
        ctx, ManualDesign(points=[[0.5, 0.5]]), phase_name="defense",
    )
    [s] = study.run()
    assert s.status is SampleStatus.FINISHED
    # Bypass the idempotence guard by flipping the in-memory status to
    # RUNNING (mimicking a coding bug). The STORE row stays FINISHED.
    s.status = SampleStatus.RUNNING
    # Force collect_output to raise.
    def _boom(_self_sample):
        raise RuntimeError("scenario file missing in /lcrc/...")
    monkeypatch.setattr(sim, "collect_output", _boom)
    fake_job = Job(
        spec=JobSpec(name="x", command="", cwd=s.folder),
        task_id="reused",
        status=JobStatus.FINISHED,
    )
    deltas: dict[str, int] = {"FINISHED": 0, "FAILED": 0, "RECOVERED": 0}
    study._finalize_terminal_sample(s, fake_job, deltas)

    # In-memory sample is healed back to FINISHED from the store row.
    assert s.status is SampleStatus.FINISHED
    # FAILED delta was NOT incremented for this duplicate call.
    assert deltas["FAILED"] == 0


class _PostSuccessTrackingSimulator(MockSimulator):
    """Tracks which post-success hooks fire to verify v0.17 per-sample
    finalize calls them inline (not at end-of-batch)."""

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.success_hook_calls: list[int] = []

    def cleanup_after_success(self, sample) -> None:
        self.success_hook_calls.append(sample.id)


def test_per_sample_finalize_fires_post_success_hooks_inline(
    tmp_path: Path, space: ParameterSpace,
) -> None:
    """v0.17 regression: post-success hooks fire as each sample
    finalizes inside the poll loop, not deferred to end-of-batch.

    Pre-v0.17 the hooks fired in "step 3" after the loop. Per-sample
    finalize means they fire as soon as collect_output succeeds — so
    a 100-sample batch doesn't pile up 100 transfer + cleanup calls
    at the end.
    """
    sim = _PostSuccessTrackingSimulator(function="quadratic")
    ctx = StudyContext(
        name="hooks",
        space=space,
        workspace=tmp_path,
        store=SampleStore.open(tmp_path / "store.db", "study"),
        runner=LocalRunner(),
        simulator=sim,
        metric=IdentityMetric(keys="value"),
        rng=np.random.default_rng(0),
        poll_interval=0.05,
        heartbeat_interval=0,
    )
    study = StaticDesignStudy(
        ctx,
        ManualDesign(points=[[0.5, 0.5], [0.2, 0.7], [0.8, 0.3]]),
        phase_name="hooks",
    )
    samples = study.run()
    assert len(samples) == 3
    for s in samples:
        assert s.status is SampleStatus.FINISHED, f"{s.id}: {s.message}"
    # Every sample's success hook fired exactly once.
    assert sorted(sim.success_hook_calls) == [s.id for s in samples]


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
