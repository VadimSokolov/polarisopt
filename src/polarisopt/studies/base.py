"""Study ABC and shared helpers.

A Study is a single phase of work over a SampleStore: generate samples,
submit them to a runner, monitor, collect metrics. The CLI/runner glue
chains multiple Study instances (e.g. static screening → sequential BO).
"""

from __future__ import annotations

import contextlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from polarisopt.metrics.base import Metric
from polarisopt.parameters import ParameterSpace
from polarisopt.runners.base import Job, JobStatus, Runner
from polarisopt.samples.sample import Sample, SampleStatus
from polarisopt.samples.store import SampleStore
from polarisopt.simulator.base import Simulator
from polarisopt.utils.logging import get_logger

log = get_logger(__name__)


class StudyError(RuntimeError):
    """Study-level orchestration failure (not a per-sample simulation error)."""


@dataclass
class StudyContext:
    """The shared state every Study needs.

    Aggregating these into one struct keeps Study subclasses' __init__
    short and makes them easier to test.
    """

    name: str
    space: ParameterSpace
    workspace: Path
    store: SampleStore
    runner: Runner
    simulator: Simulator
    metric: Metric
    rng: np.random.Generator
    poll_interval: float = 5.0


class Study(ABC):
    """Phase of a study. Implementations override :meth:`run`."""

    def __init__(self, ctx: StudyContext) -> None:
        self.ctx = ctx

    @abstractmethod
    def run(self) -> list[Sample]:
        """Execute the phase. Returns the samples it owns (typically all
        rows in the store with this phase's name)."""

    # ------- shared helpers used by static + sequential phases -------

    def _experiments_dir(self) -> Path:
        d = self.ctx.workspace / "experiments"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _evaluate_batch(self, samples: list[Sample]) -> list[Sample]:
        """Submit, monitor, and collect a batch of samples synchronously.

        Each sample's status becomes FINISHED (with metric set) or FAILED
        (with message set). Returns the samples in submission order.
        """
        if not samples:
            return []

        ctx = self.ctx
        jobs: dict[int, Job] = {}

        # 1) Submit
        for sample in samples:
            sample.folder = self._experiments_dir() / f"sim-{sample.id:06d}"
            spec = ctx.simulator.prepare(sample, ctx.space, sample.folder)
            try:
                job = ctx.runner.submit(spec)
            except Exception as exc:
                log.exception("submit failed for sample %s", sample.id)
                sample.status = SampleStatus.FAILED
                sample.message = f"submit failed: {exc}"
                ctx.store.update(sample)
                continue
            sample.runner_task_id = job.task_id
            sample.status = SampleStatus.RUNNING
            ctx.store.update(sample)
            assert sample.id is not None
            jobs[sample.id] = job

        # 2) Poll until all jobs terminate
        outstanding = dict(jobs)
        while outstanding:
            for sid, job in list(outstanding.items()):
                ctx.runner.status(job)
                if job.status.is_terminal():
                    outstanding.pop(sid)
            if outstanding:
                time.sleep(ctx.poll_interval)

        # 3) Collect metrics
        for sample in samples:
            if sample.status is SampleStatus.FAILED:
                continue
            assert sample.id is not None
            job = jobs.get(sample.id)
            if job is None:
                continue
            if job.status is JobStatus.FAILED:
                sample.status = SampleStatus.FAILED
                sample.message = job.message or "runner reported FAILED"
                ctx.store.update(sample)
                continue
            if job.status is JobStatus.CANCELLED:
                sample.status = SampleStatus.CANCELLED
                sample.message = job.message or "runner reported CANCELLED"
                ctx.store.update(sample)
                continue
            try:
                output = ctx.simulator.collect_output(sample)
                sample.metric = ctx.metric.compute(output)
            except Exception as exc:
                log.exception("collect/metric failed for sample %s", sample.id)
                sample.status = SampleStatus.FAILED
                sample.message = f"metric failed: {exc}"
                ctx.store.update(sample)
                continue
            sample.status = SampleStatus.FINISHED
            if "runtime_s" in output:
                with contextlib.suppress(TypeError, ValueError):
                    sample.runtime_s = float(output["runtime_s"])
            ctx.store.update(sample)

        return samples
