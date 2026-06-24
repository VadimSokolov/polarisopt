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
    config_snapshot :
        Full dict the fingerprint was hashed from (v0.17+). Stored
        alongside the fingerprint on every sample so
        ``ConfigDriftError`` can show the field-level diff between
        recorded and current config, not just "hashes differ".
        ``None`` to skip snapshot recording (fingerprint still works).
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
    config_snapshot: dict | None = None
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

    def _finalize_terminal_sample(
        self, sample: Sample, job: Job, deltas: dict[str, int],
    ) -> None:
        """Transition a sample to its terminal SampleStore state.

        Called from inside the poll loop as soon as ``job.status`` goes
        terminal. Was previously deferred to a post-loop "step 3" — that
        meant a slow last-sample blocked metric collection for the rest
        of the batch (the v0.15 stuck-master bug). Per-sample finalize
        means each sample makes progress independently.

        Mutates ``sample`` in-place and increments the heartbeat
        ``deltas`` counter so a working master shows non-zero progress
        in the heartbeat log.
        """
        ctx = self.ctx
        if job.status is JobStatus.FAILED:
            sample.status = SampleStatus.FAILED
            sample.message = job.message or "runner reported FAILED"
            ctx.store.update(sample)
            deltas["FAILED"] += 1
            return
        if job.status is JobStatus.CANCELLED:
            sample.status = SampleStatus.CANCELLED
            sample.message = job.message or "runner reported CANCELLED"
            ctx.store.update(sample)
            return
        # FINISHED — try collect_output + metric.
        try:
            output = ctx.simulator.collect_output(sample)
            sample.metric = ctx.metric.compute(output)
        except Exception as exc:
            log.exception("collect/metric failed for sample %s", sample.id)
            sample.status = SampleStatus.FAILED
            sample.message = f"metric failed: {exc}"
            ctx.store.update(sample)
            deltas["FAILED"] += 1
            return
        sample.status = SampleStatus.FINISHED
        # Clear any stale message left by prior retry attempts ("auto-retry
        # 1/3", etc.) — a FINISHED row shouldn't carry the audit text from
        # how it got there. The retry history is preserved in
        # sample.extra["retry_log"] for users who want the audit trail.
        sample.message = None
        if isinstance(output, dict) and "runtime_s" in output:
            with contextlib.suppress(TypeError, ValueError):
                sample.runtime_s = float(output["runtime_s"])
        ctx.store.update(sample)
        deltas["FINISHED"] += 1
        # Post-success hooks fire immediately so disk pressure stays low
        # and remote-storage pushes don't pile up at the end of a batch.
        self._post_success_hooks(sample, output)

    def _post_success_hooks(self, sample: Sample, output: dict) -> None:
        """Run results_transfer + cleanup_after_success on a FINISHED sample.

        v0.16+. Both hooks are optional and best-effort:
        ``transfer_results`` pushes the simulation outputs to durable
        storage (Globus / cfs2 / wherever the simulator's
        ``results_transfer`` is configured for);
        ``cleanup_after_success`` rm-rf's or prunes the per-sample
        workspace to free local disk. Order matters — transfer runs
        first because cleanup may delete the source.

        A raise from either hook logs a WARNING but leaves the sample
        FINISHED. The metric is already persisted; durable copy and
        local cleanup are belt-and-suspenders, not part of the
        correctness path.
        """
        transfer_hook = getattr(self.ctx.simulator, "transfer_results", None)
        if callable(transfer_hook):
            try:
                transfer_hook(sample, output)
            except Exception:  # noqa: BLE001 — best-effort
                log.exception(
                    "transfer_results raised for sample %s; FINISHED preserved",
                    sample.id,
                )
        cleanup_hook = getattr(self.ctx.simulator, "cleanup_after_success", None)
        if callable(cleanup_hook):
            try:
                cleanup_hook(sample)
            except Exception:  # noqa: BLE001 — best-effort
                log.exception(
                    "cleanup_after_success raised for sample %s; FINISHED preserved",
                    sample.id,
                )

    def _try_disk_recovery_inline(self, sample: Sample) -> bool:
        """Attempt to harvest outputs from disk for a single in-flight sample.

        Thin wrapper around :func:`polarisopt.studies.ops._try_recover_from_disk`
        — kept here so the import stays local to the recovery path and
        doesn't pull `ops` into module-level imports (avoids circulars).
        Returns True if the sample transitioned to FINISHED.
        """
        from polarisopt.studies.ops import _try_recover_from_disk
        ctx = self.ctx
        return _try_recover_from_disk(sample, ctx.simulator, ctx.metric, ctx.store)

    def _log_heartbeat(
        self,
        outstanding: dict[int, Job],
        batch_started: float,
        transitions: dict[str, int] | None = None,
    ) -> None:
        """INFO-log a one-line summary of every still-running sample.

        Fires periodically inside :meth:`_evaluate_batch`'s poll loop so
        that long-running batches don't go silent between state
        transitions.

        ``transitions`` (v0.15) carries the FINISHED / FAILED / RECOVERED
        deltas since the previous heartbeat — surfaces a stalled-master
        signal even when the runner-side counts haven't moved.

        v0.17: also queries the SampleStore for samples that have been
        RUNNING for more than 1 hour and emits a separate WARNING line
        if any are stale. The "is the master doing anything?" question
        becomes self-evident in the log.
        """
        elapsed_total = time.monotonic() - batch_started
        # Cluster samples by status so 100-sample batches stay readable.
        by_status: dict[str, list[int]] = {}
        for sid, job in outstanding.items():
            by_status.setdefault(job.status.name, []).append(sid)
        summary = ", ".join(
            f"{status}={len(sids)}" for status, sids in sorted(by_status.items())
        )
        delta = ""
        if transitions:
            delta = " — " + ", ".join(
                f"+{n} {label} since last"
                for label, n in transitions.items()
                if n > 0
            )
            if not any(n > 0 for n in transitions.values()):
                delta = " — 0 transitions since last (master may be stalled)"
        log.info(
            "[heartbeat] %d sample(s) outstanding after %s — %s%s",
            len(outstanding),
            _fmt_elapsed(elapsed_total),
            summary,
            delta,
        )
        self._warn_on_stale_running()

    def _warn_on_stale_running(self, stale_threshold_s: float = 3600.0) -> None:
        """Query the store for RUNNING samples whose updated_at is more
        than ``stale_threshold_s`` seconds in the past, and WARN if any.

        Acts as a canary against bugs the inline finalize fix can't
        catch (e.g. a sample whose ``collect_output`` hangs indefinitely
        without raising). Best-effort — query failure logs DEBUG, not
        WARNING, so a busted store doesn't spam.
        """
        from datetime import datetime

        from polarisopt.utils._compat import UTC

        try:
            running = self.ctx.store.list(status=SampleStatus.RUNNING)
        except Exception:  # noqa: BLE001
            log.debug("stale-running scan: store.list failed", exc_info=True)
            return
        if not running:
            return
        now = datetime.now(UTC)
        stale_ids: list[int] = []
        for s in running:
            if s.updated_at is None or s.id is None:
                continue
            updated = s.updated_at
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=UTC)
            if (now - updated).total_seconds() >= stale_threshold_s:
                stale_ids.append(s.id)
        if stale_ids:
            log.warning(
                "[heartbeat] WARNING: %d sample(s) RUNNING >%.0fh in SampleStore: %s "
                "— consider polarisopt recover-from-disk if the master is stuck.",
                len(stale_ids),
                stale_threshold_s / 3600,
                sorted(stale_ids)[:20],  # cap to first 20 to keep the log line readable
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
                # retry-failed / resume can refuse (and DIFF, v0.17+) if
                # the user edits the YAML.
                from polarisopt.studies.ops import (
                    EXTRA_CONFIG_SNAPSHOT_KEY,
                    EXTRA_FINGERPRINT_KEY,
                )
                sample.extra[EXTRA_FINGERPRINT_KEY] = ctx.config_fingerprint
                if ctx.config_snapshot is not None:
                    sample.extra[EXTRA_CONFIG_SNAPSHOT_KEY] = ctx.config_snapshot
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
        sid_to_sample = {s.id: s for s in samples if s.id is not None}
        # Heartbeat transition deltas — reset on each heartbeat emit.
        deltas: dict[str, int] = {"FINISHED": 0, "FAILED": 0, "RECOVERED": 0}
        try:
            while outstanding:
                for sid, job in list(outstanding.items()):
                    ctx.runner.status(job)
                    if job.status.is_terminal():
                        # v0.17: per-sample finalize INSIDE the poll loop.
                        # Previously we only popped from outstanding here
                        # and deferred collect_output + metric.compute +
                        # store.update to a "step 3" after the entire
                        # batch finished. That meant one hung sample held
                        # 15 already-FINISHED siblings in RUNNING state in
                        # the SampleStore until the last one returned
                        # (the v0.15-shipped "self-heal" only covered the
                        # UNKNOWN path, not the FINISHED-but-not-collected
                        # path). Now each sample makes progress
                        # independently.
                        sample = sid_to_sample.get(sid)
                        if sample is not None:
                            self._finalize_terminal_sample(sample, job, deltas)
                        outstanding.pop(sid)
                        continue
                    if job.status is JobStatus.UNKNOWN:
                        # Disk-first self-heal (v0.15+). Before treating
                        # UNKNOWN as a step toward "orphan", try to harvest
                        # outputs from disk — the job almost always finished
                        # cleanly and PBS / Slurm just lost it from accounting.
                        sample = sid_to_sample.get(sid)
                        if sample is not None and self._try_disk_recovery_inline(sample):
                            log.info(
                                "sample %s: recovered from disk after UNKNOWN status",
                                sid,
                            )
                            job.status = JobStatus.FINISHED
                            deltas["RECOVERED"] += 1
                            outstanding.pop(sid)
                            continue
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
                            # Per-sample finalize on the FAILED transition.
                            sample = sid_to_sample.get(sid)
                            if sample is not None:
                                self._finalize_terminal_sample(sample, job, deltas)
                            outstanding.pop(sid)
                    else:
                        unknown_counts[sid] = 0
                if outstanding:
                    now = time.monotonic()
                    if (
                        ctx.heartbeat_interval > 0
                        and now - last_heartbeat >= ctx.heartbeat_interval
                    ):
                        self._log_heartbeat(outstanding, batch_started, deltas)
                        last_heartbeat = now
                        # Reset deltas; next heartbeat reports activity
                        # since this one.
                        deltas = {"FINISHED": 0, "FAILED": 0, "RECOVERED": 0}
                    time.sleep(ctx.poll_interval)
        except KeyboardInterrupt:
            log.warning(
                "KeyboardInterrupt — cancelling %d in-flight job(s) before exit",
                len(outstanding),
            )
            self._cancel_outstanding(samples, jobs, outstanding)
            raise

        # 3) (v0.17+) Per-sample finalize now runs inline in step 2 as soon
        # as each job goes terminal. The old "wait for the whole batch then
        # collect all" pattern is gone — it produced the v0.15-era
        # stuck-in-RUNNING-after-PBS-says-FINISHED zombies because a single
        # slow sample blocked metric collection for the other 15 in its
        # batch. Defensive sweep below catches any sample that somehow
        # ended up RUNNING after the loop (shouldn't happen post-refactor;
        # warns and falls back to disk recovery if it does).
        for sample in samples:
            if sample.status is not SampleStatus.RUNNING:
                continue
            log.warning(
                "sample %s exited poll loop still RUNNING; falling back to disk recovery",
                sample.id,
            )
            self._try_disk_recovery_inline(sample)

        # 4) Auto-retry FAILED samples up to ctx.max_retries times. Catches
        # transient failures (OOM on a contended node, node failure, time
        # limit, sacct hiccup) without manual `retry-failed` intervention.
        # Permanent failures still get rejected after exhausting the budget.
        # The per-sample folder is reused — we trust the simulator to
        # overwrite its outputs (POLARIS does).
        retriable_ids: set[int] = set()
        if ctx.max_retries > 0:
            retriable = self._collect_retriable(samples)
            if retriable:
                retriable_ids = {s.id for s in retriable if s.id is not None}
                log.info(
                    "[retry] re-submitting %d sample(s) after FAILED transition",
                    len(retriable),
                )
                self._evaluate_batch(retriable)

        # 5) Cleanup terminally-FAILED samples if the simulator opted in.
        # Skip samples that were retried at THIS level — the recursive
        # _evaluate_batch above has already run cleanup on them if they
        # exhausted their budget. Without this gate, a sample that
        # retries N times has cleanup_hook called N+1 times (one per
        # recursion level). Best-effort — failures are logged but don't
        # change sample state.
        cleanup_hook = getattr(ctx.simulator, "cleanup_after_failure", None)
        if callable(cleanup_hook):
            for sample in samples:
                if sample.status is SampleStatus.FAILED and sample.id not in retriable_ids:
                    try:
                        cleanup_hook(sample)
                    except Exception:  # noqa: BLE001 — cleanup is best-effort
                        log.exception(
                            "cleanup_after_failure raised for sample %s", sample.id,
                        )

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
