"""Orphan-job detection: UNKNOWN polls > threshold → sample FAILED."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from polarisopt.design import ManualDesign
from polarisopt.metrics import IdentityMetric
from polarisopt.parameters import Parameter, ParameterSpace
from polarisopt.runners.base import Job, JobSpec, JobStatus, Runner
from polarisopt.samples.sample import SampleStatus
from polarisopt.samples.store import SampleStore
from polarisopt.simulator import MockSimulator
from polarisopt.studies.base import StudyContext
from polarisopt.studies.static import StaticDesignStudy


class AlwaysUnknownRunner(Runner):
    """Returns UNKNOWN on every status() call, simulating a lost job."""

    def __init__(self) -> None:
        self._next_id = 0
        self.cancelled: list[str] = []

    def submit(self, spec: JobSpec) -> Job:
        self._next_id += 1
        return Job(spec=spec, task_id=f"orphan-{self._next_id}", status=JobStatus.RUNNING)

    def status(self, job: Job) -> Job:
        job.status = JobStatus.UNKNOWN
        job.message = "vanished"
        return job

    def cancel(self, job: Job) -> Job:
        self.cancelled.append(job.task_id)
        job.status = JobStatus.CANCELLED
        return job


def test_orphan_threshold_marks_sample_failed(tmp_path: Path) -> None:
    space = ParameterSpace.from_iterable([Parameter("x", "a.json", 0.0, 1.0)])
    ctx = StudyContext(
        name="orphan",
        space=space,
        workspace=tmp_path,
        store=SampleStore.open(tmp_path / "db.sqlite", "orphan-study"),
        runner=AlwaysUnknownRunner(),
        simulator=MockSimulator(function="quadratic"),
        metric=IdentityMetric(keys="value"),
        rng=np.random.default_rng(0),
        poll_interval=0.01,
        orphan_threshold=2,
    )
    design = ManualDesign(points=[[0.5]])
    study = StaticDesignStudy(ctx, design, phase_name="orphan")
    samples = study.run()
    assert len(samples) == 1
    s = samples[0]
    assert s.status is SampleStatus.FAILED
    assert s.message is not None and "orphan" in s.message.lower()


def test_orphan_threshold_default_is_three(tmp_path: Path) -> None:
    """Smoke test: default StudyContext orphan_threshold is 3 (sensible default)."""
    space = ParameterSpace.from_iterable([Parameter("x", "a.json", 0.0, 1.0)])
    ctx = StudyContext(
        name="defaults",
        space=space,
        workspace=tmp_path,
        store=SampleStore.open(tmp_path / "db.sqlite", "study"),
        runner=AlwaysUnknownRunner(),
        simulator=MockSimulator(function="quadratic"),
        metric=IdentityMetric(keys="value"),
        rng=np.random.default_rng(0),
    )
    assert ctx.orphan_threshold == 3
