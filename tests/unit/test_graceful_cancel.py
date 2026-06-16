"""Tests for KeyboardInterrupt-driven graceful cancellation in _evaluate_batch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from polarisopt.design import ManualDesign
from polarisopt.metrics import IdentityMetric
from polarisopt.parameters import Parameter, ParameterSpace
from polarisopt.runners.base import Job, JobSpec, JobStatus, Runner
from polarisopt.samples.sample import SampleStatus
from polarisopt.samples.store import SampleStore
from polarisopt.simulator import MockSimulator
from polarisopt.studies.base import StudyContext
from polarisopt.studies.static import StaticDesignStudy


class InterruptingRunner(Runner):
    """Runner that raises KeyboardInterrupt on the 2nd status() call.

    Simulates the user pressing Ctrl-C while the master is polling.
    """

    def __init__(self) -> None:
        self._next_id = 0
        self._calls = 0
        self.cancelled: list[str] = []

    def submit(self, spec: JobSpec) -> Job:
        self._next_id += 1
        return Job(spec=spec, task_id=f"job-{self._next_id}", status=JobStatus.RUNNING)

    def status(self, job: Job) -> Job:
        self._calls += 1
        if self._calls >= 2:
            raise KeyboardInterrupt
        return job  # still RUNNING

    def cancel(self, job: Job) -> Job:
        self.cancelled.append(job.task_id)
        job.status = JobStatus.CANCELLED
        return job


def test_keyboard_interrupt_cancels_outstanding_and_marks_samples(tmp_path: Path) -> None:
    space = ParameterSpace.from_iterable([Parameter("x", "a.json", 0.0, 1.0)])
    runner = InterruptingRunner()
    ctx = StudyContext(
        name="cancel-test",
        space=space,
        workspace=tmp_path,
        store=SampleStore.open(tmp_path / "db.sqlite", "study"),
        runner=runner,
        simulator=MockSimulator(function="quadratic"),
        metric=IdentityMetric(keys="value"),
        rng=np.random.default_rng(0),
        poll_interval=0.0,
    )
    design = ManualDesign(points=[[0.5], [0.7], [0.9]])
    study = StaticDesignStudy(ctx, design, phase_name="p")

    with pytest.raises(KeyboardInterrupt):
        study.run()

    # All three samples were submitted (so runner.cancel was called on all).
    assert len(runner.cancelled) == 3
    # All three samples are now CANCELLED in the store with the shutdown message.
    rows = ctx.store.list()
    assert len(rows) == 3
    for row in rows:
        assert row.status is SampleStatus.CANCELLED
        assert row.message is not None and "master shutdown" in row.message
