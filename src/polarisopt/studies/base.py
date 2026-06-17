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


def _fmt_elapsed(secs: float) -> str:
    if secs < 60:
        return f"{secs:.0f}s"
    if secs < 3600:
        return f"{secs / 60:.1f}m"
    return f"{secs / 3600:.1f}h"


class StudyError(RuntimeError):
    """Study-level orchestration failure (not a per-sample simulation error)."""


@dataclass
class StudyContext:
    """The shared state every Study needs.

    Aggregating these into one struct keeps Study subclasses' __init__
    short and makes them easier to test.

    Parameters
    ----------
    name :
        Logical name of the study (matches the YAML ``name``).
    space :
        :class:`~polarisopt.parameters.ParameterSpace`.
    workspace :
        Root directory for experiment folders, logs, scripts, and the DB.
    store :
        Open :class:`~polarisopt.samples.store.SampleStore`.
    runner :
        Backend that submits and polls jobs.
    simulator :
        Bridges Sample inputs into a :class:`JobSpec` and reads outputs back.
    metric :
        Reduces simulator output to an objective vector.
    rng :
        Shared random number generator (persisted across restart).
    poll_interval :
        Seconds between :meth:`Runner.status` calls. Default 5s.
    orphan_threshold :
        Number of consecutive ``JobStatus.UNKNOWN`` polls before a sample
        is force-marked FAILED. Catches Slurm jobs that disappear without
        a finished sentinel. Default 3. Set to 0 to disable orphan
        detection (legacy behavior — poll forever).
    heartbeat_interval :
        Seconds between INFO-level "still running" log lines summarizing
        every in-flight sample. Default 300s (5 min). Set to 0 to disable
        heartbeats (the log only fires on state transitions, legacy
        behavior). The first heartbeat fires one ``heartbeat_interval``
        after submission, not immediately.
    config_fingerprint :
        Short hash of the simulator/runner config (see
        :func:`polarisopt.studies.ops.simulator_config_fingerprint`).
        Recorded on every sample at submit so
        :func:`polarisopt.studies.ops.retry_failed` can refuse to mix
        runs across edited configs. ``None`` to skip recording.
    max_retries :
        Number of automatic retries to attempt on a sample that hits a
        FAILED transition (transient OOM, node failure, timeout, etc.).
        Default 0 — no auto-retry, current behavior. Set to e.g. 2 and
        a sample is allowed 3 total attempts before it stays FAILED.
        Retry count is persisted to ``sample.extra["retry_count"]``
        so ``polarisopt status -v`` can show "1× failed" vs "3× failed
        → probably a real bug." Configurable per study via
        ``runner.options.max_retries``.
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
    orphan_threshold: int = 3
    heartbeat_interval: float = 300.0
    config_fingerprint: str | None = None
    max_retries: int = 0


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

    def _log_heartbeat(self, outstanding: dict[int, Job], batch_started: float) -> None:
        """INFO-log a one-line summary of every still-running sample.

        Fires periodically inside :meth:`_evaluate_batch`'s poll loop so
        that long-running batches don't go silent between state
        transitions (a poll cycle on a 24h ABM run might only log on
        submit + finish, leaving the user blind in between).
        """
        elapsed_total = time.monotonic() - batch_started
        # Cluster samples by status so 100-sample batches stay readable.
        by_status: dict[str, list[int]] = {}
        for sid, job in outstanding.items():
            by_status.setdefault(job.status.name, []).append(sid)
        summary = ", ".join(
            f"{status}={len(sids)}" for status, sids in sorted(by_status.items())
        )
        log.info(
            "[heartbeat] %d sample(s) outstanding after %s — %s",
            len(outstanding),
            _fmt_elapsed(elapsed_total),
            summary,
        )

    def _cancel_outstanding(
        self,
        samples: list[Sample],
        jobs: dict[int, Job],
        outstanding: dict[int, Job],
    ) -> None:
        """Cancel every still-running job and mark its sample CANCELLED."""
        ctx = self.ctx
        sid_to_sample = {s.id: s for s in samples if s.id is not None}
        for sid, job in outstanding.items():
            try:
                ctx.runner.cancel(job)
            except Exception:  # noqa: BLE001 — best-effort cleanup
                log.exception("runner.cancel failed for sample %s", sid)
            sample = sid_to_sample.get(sid)
            if sample is None:
                continue
            sample.status = SampleStatus.CANCELLED
            sample.message = (sample.message or "") + " | cancelled by master shutdown"
            try:
                ctx.store.update(sample)
            except Exception:  # noqa: BLE001
                log.exception("store.update failed for cancelled sample %s", sid)

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
            if ctx.config_fingerprint is not None:
                # Record the config the sample was run under so a later
                # retry-failed can refuse if the user edits the YAML.
                from polarisopt.studies.ops import EXTRA_FINGERPRINT_KEY
                sample.extra[EXTRA_FINGERPRINT_KEY] = ctx.config_fingerprint
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

        # 2) Poll until all jobs terminate. Track per-sample UNKNOWN counts
        # so orphaned Slurm jobs (e.g. lost from squeue + sacct) don't make
        # us poll forever. KeyboardInterrupt cleanly cancels all in-flight
        # jobs and re-raises so the orchestrator can shut down.
        outstanding = dict(jobs)
        unknown_counts: dict[int, int] = {sid: 0 for sid in outstanding}
        batch_started = time.monotonic()
        last_heartbeat = batch_started
        try:
            while outstanding:
                for sid, job in list(outstanding.items()):
                    ctx.runner.status(job)
                    if job.status.is_terminal():
                        outstanding.pop(sid)
                        continue
                    if job.status is JobStatus.UNKNOWN:
                        unknown_counts[sid] = unknown_counts.get(sid, 0) + 1
                        if (
                            ctx.orphan_threshold > 0
                            and unknown_counts[sid] >= ctx.orphan_threshold
                        ):
                            log.warning(
                                "sample %s: jobid %s orphaned after %d UNKNOWN polls",
                                sid,
                                job.task_id,
                                unknown_counts[sid],
                            )
                            job.status = JobStatus.FAILED
                            job.message = (
                                f"job orphaned (Slurm lost track of jobid={job.task_id})"
                            )
                            outstanding.pop(sid)
                    else:
                        unknown_counts[sid] = 0
                if outstanding:
                    now = time.monotonic()
                    if (
                        ctx.heartbeat_interval > 0
                        and now - last_heartbeat >= ctx.heartbeat_interval
                    ):
                        self._log_heartbeat(outstanding, batch_started)
                        last_heartbeat = now
                    time.sleep(ctx.poll_interval)
        except KeyboardInterrupt:
            log.warning(
                "KeyboardInterrupt — cancelling %d in-flight job(s) before exit",
                len(outstanding),
            )
            self._cancel_outstanding(samples, jobs, outstanding)
            raise

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

        # 4) Auto-retry FAILED samples up to ctx.max_retries times. Catches
        # transient failures (OOM on a contended node, node failure, time
        # limit, sacct hiccup) without manual `retry-failed` intervention.
        # Permanent failures still get rejected after exhausting the budget.
        # The per-sample folder is reused — we trust the simulator to
        # overwrite its outputs (POLARIS does).
        if ctx.max_retries > 0:
            retriable = self._collect_retriable(samples)
            if retriable:
                log.info(
                    "[retry] re-submitting %d sample(s) after FAILED transition",
                    len(retriable),
                )
                self._evaluate_batch(retriable)

        return samples

    def _collect_retriable(self, samples: list[Sample]) -> list[Sample]:
        """Flip eligible FAILED samples back to PENDING and return them.

        A sample is eligible if it's currently FAILED and its recorded
        retry_count (``sample.extra["retry_count"]``, 0 if unset) is
        below ``ctx.max_retries``. The flip updates retry_count, clears
        the stale runner_task_id, sets status to PENDING, and persists.
        Returns the in-place-modified samples for the recursive
        ``_evaluate_batch`` call.
        """
        ctx = self.ctx
        retriable: list[Sample] = []
        for sample in samples:
            if sample.status is not SampleStatus.FAILED:
                continue
            attempts = int(sample.extra.get("retry_count", 0))
            if attempts >= ctx.max_retries:
                continue
            # message gets overwritten by the next FAILED transition, so
            # the audit trail lives in extra. Use sample.message to record
            # the *current* attempt's failure reason; extra["retry_log"]
            # to record what we've already burned through.
            retry_log = list(sample.extra.get("retry_log", []))
            retry_log.append(
                {
                    "attempt": attempts + 1,
                    "max": ctx.max_retries,
                    "prior_message": sample.message,
                }
            )
            sample.extra["retry_log"] = retry_log
            sample.extra["retry_count"] = attempts + 1
            sample.status = SampleStatus.PENDING
            sample.runner_task_id = None
            ctx.store.update(sample)
            retriable.append(sample)
        return retriable
