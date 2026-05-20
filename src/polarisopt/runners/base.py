"""Runner ABC — submit a shell command, poll status, fetch logs, cancel.

Runners are intentionally narrow: one shell command per :class:`Job`. Higher
level concerns (parameter injection, output collection, metric computation)
live in :mod:`polarisopt.simulator`, :mod:`polarisopt.metrics`, and the study
orchestrators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from polarisopt.utils._compat import StrEnum
from polarisopt.utils.registry import Registry


class JobStatus(StrEnum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"

    def is_terminal(self) -> bool:
        return self in {JobStatus.FINISHED, JobStatus.FAILED, JobStatus.CANCELLED}


class RunnerError(RuntimeError):
    """Raised when the runner backend itself misbehaves (e.g. sbatch nonzero)."""


@dataclass
class JobSpec:
    """What to run.

    The ``command`` is a shell snippet executed by the runner in ``cwd``.
    Runners may write it to a script file (Slurm) or exec it directly (local).
    Resource hints go in ``extra``; concrete runners interpret them.
    """

    name: str
    command: str
    cwd: Path
    stdout: Path | None = None
    stderr: Path | None = None
    env: dict[str, str] = field(default_factory=dict)
    extra: dict[str, object] = field(default_factory=dict)


@dataclass
class Job:
    """A submitted job — its handle and last-known status."""

    spec: JobSpec
    task_id: str
    status: JobStatus = JobStatus.QUEUED
    message: str | None = None
    exit_code: int | None = None


class Runner(ABC):
    """Submit shell commands and track them to completion.

    Concrete runners must be safe to call from multiple threads, since the
    Study orchestrator may submit a batch of samples concurrently.
    """

    @abstractmethod
    def submit(self, spec: JobSpec) -> Job:
        """Submit a job. Returns a :class:`Job` whose ``task_id`` identifies it."""

    @abstractmethod
    def status(self, job: Job) -> Job:
        """Refresh and return the job's status (mutates and returns ``job``)."""

    @abstractmethod
    def cancel(self, job: Job) -> Job:
        """Request cancellation. Best-effort; refresh and return."""

    def logs(self, job: Job) -> tuple[str, str]:
        """Return ``(stdout, stderr)`` from disk. Empty strings if unavailable."""
        stdout = job.spec.stdout.read_text() if job.spec.stdout and job.spec.stdout.exists() else ""
        stderr = job.spec.stderr.read_text() if job.spec.stderr and job.spec.stderr.exists() else ""
        return stdout, stderr


runner_registry: Registry[Runner] = Registry("runner")
