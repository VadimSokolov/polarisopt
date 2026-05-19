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

from pathlib import Path

from polarisopt.config.schema import StudyConfig
from polarisopt.runners.base import Job, JobSpec, Runner
from polarisopt.runners.factory import make_runner
from polarisopt.samples.sample import Sample, SampleStatus
from polarisopt.samples.store import SampleStore
from polarisopt.utils.logging import get_logger
from polarisopt.utils.paths import workspace_layout

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
    """Re-instantiate the Runner declared in ``config``."""
    return make_runner({"type": config.runner.type, "options": config.runner.options})


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


def retry_failed(
    config: StudyConfig,
    *,
    sample_ids: list[int] | None = None,
    store: SampleStore | None = None,
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

    Returns
    -------
    list of Sample
        The samples that were transitioned FAILED → PENDING.

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

    retried: list[Sample] = []
    for sample in candidates:
        sample.status = SampleStatus.PENDING
        sample.message = (sample.message or "") + " | retry"
        # Clear stale runner_task_id so resume gets a fresh sbatch.
        sample.runner_task_id = None
        store.update(sample)
        retried.append(sample)
    return retried


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
