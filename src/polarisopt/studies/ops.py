"""Operational helpers for an already-running study.

The functions in this module act on the SampleStore + Runner without
running a full :class:`StudyRunner`. Used by the CLI subcommands
``cancel`` / ``abort`` / ``logs`` so users can manage in-flight studies
from the command line.

Notes
-----
All operations are idempotent: cancelling an already-terminal sample is
a no-op; aborting an empty store does nothing.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from pathlib import Path

from polarisopt.config.schema import StudyConfig
from polarisopt.metrics.base import Metric, make_metric
from polarisopt.runners.base import Job, JobSpec, JobStatus, Runner
from polarisopt.runners.factory import make_runner
from polarisopt.samples.sample import Sample, SampleStatus
from polarisopt.samples.store import SampleStore
from polarisopt.simulator.base import Simulator, make_simulator
from polarisopt.utils.logging import get_logger
from polarisopt.utils.paths import workspace_layout

EXTRA_FINGERPRINT_KEY = "config_fingerprint"

# Runner options that affect orchestrator behavior (polling cadence, log
# verbosity, retry policy) but not simulation outcomes. Excluded from the
# fingerprint so tweaking these doesn't trigger a spurious drift error on
# retry-failed / resume.
_ORCHESTRATOR_RUNNER_OPTIONS = frozenset(
    {"poll_interval", "orphan_threshold", "heartbeat_interval", "max_retries"}
)


def simulator_config_fingerprint(config: StudyConfig) -> str:
    """Short stable hash of the bits that affect simulation outcome.

    Used by :func:`retry_failed` to detect that the user edited the
    simulator/runner config between the original run and the retry —
    silently mixing outputs at different parameter scales inside one
    SampleStore has burned users in practice.

    Includes ``simulator`` and ``runner`` blocks (type + options) except
    orchestrator-only knobs (``poll_interval``, ``orphan_threshold``,
    ``heartbeat_interval``). Excludes things that don't change what the
    binary does: ``name``, ``workspace``, ``seed`` (different seed = a
    different next sample, not stale-result contamination), ``phases``,
    ``parameters``, ``metric``.
    """
    runner_opts = {
        k: v for k, v in config.runner.options.items()
        if k not in _ORCHESTRATOR_RUNNER_OPTIONS
    }
    payload = {
        "simulator": {"type": config.simulator.type, "options": config.simulator.options},
        "runner": {"type": config.runner.type, "options": runner_opts},
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]

log = get_logger(__name__)


def open_store(config: StudyConfig) -> SampleStore:
    """Open the SampleStore at the workspace defined by ``config``."""
    layout = workspace_layout(config.workspace)
    if not layout["db"].exists():
        raise FileNotFoundError(
            f"no SampleStore at {layout['db']}; has the study been started?"
        )
    return SampleStore.open(layout["db"], config.name)


def build_runner(config: StudyConfig) -> Runner:
    """Re-instantiate the Runner declared in ``config``.

    Mirrors :class:`StudyRunner.__init__`'s strip of orchestrator-only
    knobs (``poll_interval`` / ``orphan_threshold`` /
    ``heartbeat_interval``). Without this the resume path crashes on
    ``SlurmRunner.__init__() got an unexpected keyword argument
    'poll_interval'`` because ``reconcile_running`` and the
    cancel/abort helpers all go through here while ``polarisopt run``
    goes through ``StudyRunner`` (which pops). Single behavior across
    every entry point.
    """
    runner_options = {
        k: v for k, v in config.runner.options.items()
        if k not in _ORCHESTRATOR_RUNNER_OPTIONS
    }
    return make_runner({"type": config.runner.type, "options": runner_options})


def _runner_job_for(sample: Sample) -> Job:
    """Build a minimal Job referring to ``sample.runner_task_id``.

    Most concrete runners only need the task_id to call ``status``/``cancel``;
    the JobSpec is supplied to satisfy the dataclass contract.
    """
    folder = sample.folder if sample.folder is not None else Path(".")
    spec = JobSpec(name=f"sample-{sample.id}", command="", cwd=folder)
    return Job(spec=spec, task_id=sample.runner_task_id or "")


def cancel_sample(
    sample_id: int, *, config: StudyConfig, store: SampleStore | None = None
) -> Sample:
    """Cancel one sample. Returns the updated Sample.

    Parameters
    ----------
    sample_id :
        Sample primary key in the SampleStore.
    config :
        The study YAML (needed to rebuild the Runner).
    store :
        Optional pre-opened SampleStore; if ``None`` we open it from config.

    Notes
    -----
    Idempotent. If the sample is already terminal the function returns it
    unmodified. If the sample never had a runner task assigned, only the
    store row is updated to CANCELLED.
    """
    store = store or open_store(config)
    sample = store.get(sample_id)
    if sample.is_terminal():
        return sample
    if sample.runner_task_id:
        runner = build_runner(config)
        runner.cancel(_runner_job_for(sample))
    sample.status = SampleStatus.CANCELLED
    sample.message = (sample.message or "") + " | cancelled via CLI"
    return store.update(sample)


def abort_study(config: StudyConfig, *, store: SampleStore | None = None) -> list[Sample]:
    """Cancel every non-terminal sample in the study. Returns the list of
    samples that were transitioned to CANCELLED.
    """
    store = store or open_store(config)
    runner = build_runner(config)
    cancelled: list[Sample] = []
    for sample in store.list():
        if sample.is_terminal():
            continue
        if sample.runner_task_id:
            try:
                runner.cancel(_runner_job_for(sample))
            except Exception as exc:  # noqa: BLE001 — log and continue
                log.warning("runner.cancel failed for sample %s: %s", sample.id, exc)
        sample.status = SampleStatus.CANCELLED
        sample.message = (sample.message or "") + " | aborted"
        store.update(sample)
        cancelled.append(sample)
    return cancelled


class ConfigDriftError(RuntimeError):
    """Raised when retrying a sample whose recorded simulator/runner config
    differs from the current YAML.

    Mixing samples evaluated at different parameter scales in one
    SampleStore silently corrupts downstream analysis — refuse instead
    of letting it through. The CLI surfaces a ``--force`` flag for the
    case where the user genuinely means "retry under the new config."
    """


def retry_failed(
    config: StudyConfig,
    *,
    sample_ids: list[int] | None = None,
    store: SampleStore | None = None,
    force: bool = False,
) -> list[Sample]:
    """Flip FAILED samples back to PENDING so a subsequent :func:`run_study`
    or ``polarisopt resume`` re-evaluates them.

    Parameters
    ----------
    config : StudyConfig
        The validated study config.
    sample_ids : list of int or None
        If given, retry only these samples (filtering to those that are
        currently FAILED). If ``None``, retry every FAILED sample.
    store : SampleStore or None
        Optional pre-opened store; otherwise opened from ``config``.
    force : bool
        Skip the simulator/runner config-drift check. Use when the user
        genuinely wants to retry under an edited YAML — the new samples
        will land in the same store and ``polarisopt diff`` won't help
        because there's only one store. Default ``False``.

    Returns
    -------
    list of Sample
        The samples that were transitioned FAILED → PENDING.

    Raises
    ------
    ConfigDriftError
        If any candidate sample's recorded ``config_fingerprint`` differs
        from the current config's fingerprint and ``force`` is False.

    Notes
    -----
    The previous failure message is preserved by appending
    ``" | retry"`` so the audit trail isn't lost. Resume picks up the
    PENDING samples automatically before running new iterations.
    """
    store = store or open_store(config)
    if sample_ids is not None:
        wanted = set(sample_ids)
        candidates = [s for s in store.list(status=SampleStatus.FAILED) if s.id in wanted]
        missing = wanted - {s.id for s in candidates}
        if missing:
            raise ValueError(
                f"sample ids {sorted(missing)} are not FAILED in this study "
                f"(check 'polarisopt status' first)"
            )
    else:
        candidates = store.list(status=SampleStatus.FAILED)

    if not force:
        current_fp = simulator_config_fingerprint(config)
        drifted = [
            s for s in candidates
            if (recorded := s.extra.get(EXTRA_FINGERPRINT_KEY)) is not None
            and recorded != current_fp
        ]
        if drifted:
            recorded_set = sorted({
                s.extra.get(EXTRA_FINGERPRINT_KEY) for s in drifted
            })
            raise ConfigDriftError(
                f"simulator/runner config has changed since {len(drifted)} of "
                f"{len(candidates)} failed sample(s) ran (recorded fingerprints: "
                f"{recorded_set}; current: {current_fp!r}). "
                f"Pass force=True (CLI: --force) to retry under the new config, "
                f"or start a fresh workspace to keep the run history clean."
            )

    retried: list[Sample] = []
    for sample in candidates:
        sample.status = SampleStatus.PENDING
        sample.message = (sample.message or "") + " | retry"
        # Clear stale runner_task_id so resume gets a fresh sbatch.
        sample.runner_task_id = None
        store.update(sample)
        retried.append(sample)
    return retried


def _try_recover_from_disk(
    sample: Sample,
    simulator: Simulator,
    metric: Metric,
    store: SampleStore,
) -> bool:
    """Try to harvest a sample's outputs from disk and mark it FINISHED.

    Reads ``sample.folder`` via ``simulator.collect_output`` and runs
    ``metric.compute``. If both succeed the sample is transitioned to
    FINISHED with the metric value persisted. Returns ``True`` on
    success, ``False`` on any failure (folder missing, schema mismatch,
    metric raised) — caller falls back to the runner-status verdict.

    This is the path the master would have taken at finish time if it
    hadn't died. Used by :func:`reconcile_running` and
    :func:`recover_from_disk`.
    """
    if sample.folder is None or not sample.folder.exists():
        return False
    try:
        output = simulator.collect_output(sample)
        metric_value = metric.compute(output)
    except Exception as exc:  # noqa: BLE001
        log.info(
            "disk recovery failed for sample %s (%s): %s",
            sample.id, type(exc).__name__, exc,
        )
        return False
    sample.status = SampleStatus.FINISHED
    sample.metric = metric_value
    sample.message = (sample.message or "") + " | recovered from disk"
    if isinstance(output, dict) and "runtime_s" in output:
        with contextlib.suppress(TypeError, ValueError):
            sample.runtime_s = float(output["runtime_s"])
    store.update(sample)
    return True


def reconcile_running(
    config: StudyConfig,
    *,
    store: SampleStore | None = None,
) -> list[Sample]:
    """Reconcile RUNNING samples with the runner backend at resume time.

    Disk-first: for each previously-RUNNING sample we check the runner
    status, and unless it's actively RUNNING/QUEUED we try to harvest
    outputs from disk via ``simulator.collect_output`` +
    ``metric.compute``. Disk artifacts win over the runner's verdict —
    a sample whose binary finished writing and exited cleanly before
    the master died is FINISHED, regardless of whether ``sacct`` later
    forgets the job (the v0.5 "orphaned on resume" path).

    Status transitions:

    - runner RUNNING / QUEUED → leave RUNNING (don't race a live write)
    - runner CANCELLED → mark CANCELLED (preserve user intent; skip disk)
    - runner FINISHED + disk artifacts → FINISHED with metric
    - runner FINISHED + no disk artifacts → FAILED (output missing)
    - runner FAILED + disk artifacts → FINISHED (binary wrote before exit)
    - runner FAILED + no disk artifacts → FAILED (original message preserved)
    - runner UNKNOWN + disk artifacts → FINISHED (rescued from accounting GC)
    - runner UNKNOWN + no disk artifacts → FAILED (orphan)

    Parameters
    ----------
    config : StudyConfig
        Validated study config (provides Runner + Simulator + Metric).
    store : SampleStore or None
        Optional pre-opened store; otherwise opened from ``config``.

    Returns
    -------
    list of Sample
        Samples transitioned to a terminal state by this reconcile.
    """
    store = store or open_store(config)
    running = [s for s in store.list(status=SampleStatus.RUNNING) if s.runner_task_id]
    if not running:
        return []
    runner = build_runner(config)
    simulator = make_simulator(
        {"type": config.simulator.type, "options": config.simulator.options}
    )
    metric = make_metric({"type": config.metric.type, "options": config.metric.options})
    reconciled: list[Sample] = []
    for sample in running:
        job = _runner_job_for(sample)
        try:
            runner.status(job)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "runner.status failed for sample %s (jobid=%s): %s",
                sample.id, sample.runner_task_id, exc,
            )
            continue
        # Live jobs: don't race a partial write. Leave RUNNING; the
        # caller's poll loop (or a follow-up reconcile) handles them.
        if job.status in (JobStatus.RUNNING, JobStatus.QUEUED):
            continue
        # User-cancelled: preserve intent, skip disk harvest.
        if job.status is JobStatus.CANCELLED:
            sample.status = SampleStatus.CANCELLED
            sample.message = (sample.message or "") + " | runner CANCELLED on resume"
            store.update(sample)
            reconciled.append(sample)
            continue
        # FINISHED / FAILED / UNKNOWN: try the disk first.
        if _try_recover_from_disk(sample, simulator, metric, store):
            reconciled.append(sample)
            continue
        # No usable artifacts — fall back to the runner verdict.
        if job.status is JobStatus.FAILED:
            sample.message = (sample.message or "") + (
                f" | runner FAILED on resume: {job.message or ''}"
            )
        elif job.status is JobStatus.UNKNOWN:
            sample.message = (sample.message or "") + (
                f" | orphaned on resume (Slurm lost jobid={sample.runner_task_id})"
            )
        else:  # FINISHED but disk recovery failed
            sample.message = (sample.message or "") + (
                " | runner FINISHED on resume but output is missing/unreadable"
            )
        sample.status = SampleStatus.FAILED
        store.update(sample)
        reconciled.append(sample)
    return reconciled


def recover_from_disk(
    config: StudyConfig,
    *,
    store: SampleStore | None = None,
    include_cancelled: bool = False,
) -> list[Sample]:
    """Sweep RUNNING + FAILED samples and harvest any whose outputs are on disk.

    Use this when ``reconcile_running`` can't help — typically because
    ``sacct`` retention has aged the jobids out (days to weeks later)
    so the runner says UNKNOWN with no way to distinguish a real orphan
    from a job that finished cleanly. The on-disk artifacts are the
    ground truth.

    Mirrors :func:`_try_recover_from_disk` per sample without consulting
    the runner at all. Samples with usable disk artifacts become
    FINISHED with the metric set; samples with no artifacts (or
    schema-incompatible artifacts) are left untouched.

    Parameters
    ----------
    config : StudyConfig
        Validated study config. Provides Simulator + Metric.
    store : SampleStore or None
        Optional pre-opened store; otherwise opened from ``config``.
    include_cancelled : bool, optional
        If True, also sweep CANCELLED samples (default False — user
        intent is preserved). FINISHED samples are always skipped.

    Returns
    -------
    list of Sample
        Samples transitioned to FINISHED by this sweep.
    """
    store = store or open_store(config)
    simulator = make_simulator(
        {"type": config.simulator.type, "options": config.simulator.options}
    )
    metric = make_metric({"type": config.metric.type, "options": config.metric.options})
    statuses = {SampleStatus.RUNNING, SampleStatus.FAILED}
    if include_cancelled:
        statuses.add(SampleStatus.CANCELLED)
    candidates = [s for s in store.list() if s.status in statuses]
    recovered: list[Sample] = []
    for sample in candidates:
        if _try_recover_from_disk(sample, simulator, metric, store):
            recovered.append(sample)
    return recovered


def sample_log_paths(sample: Sample) -> list[Path]:
    """Return every readable ``*.log`` / ``*.out`` / ``*.err`` file in
    ``sample.folder``, sorted by mtime.

    Empty list if the folder isn't set or contains no such files.
    """
    if sample.folder is None or not sample.folder.exists():
        return []
    patterns = ("*.log", "*.out", "*.err")
    files: list[Path] = []
    for pat in patterns:
        files.extend(sample.folder.glob(pat))
    files = [p for p in files if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime)
    return files
