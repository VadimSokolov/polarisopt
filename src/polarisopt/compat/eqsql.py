"""EQSQL-shaped API backed by plain Slurm.

This module is a drop-in-style shim for Argonne users coming from
``polaris.hpc.eqsql``. The API mirrors the surface of polarislib's EQSQL
(``insert_task``, ``Task.from_id``, ``Task.status`` and friends) but every
operation routes to ``sbatch`` / ``squeue`` / ``scancel`` underneath. No
Postgres. No worker pool. No pinned-worker regex lesson.

Status strings match polarislib's EQSQL exactly so existing client code
that compares against ``"queued"``, ``"running"``, ``"finished"``,
``"failed"``, ``"cancelled"`` keeps working without changes.

Example
-------

>>> from polarisopt.compat import eqsql
>>> with eqsql.open_queue("/path/to/workspace") as queue:
...     result = queue.insert_task(
...         definition={"task-type": "bash-script", "command": "/bin/echo hi"},
...         exp_id="hello-world",
...     )
...     task = result.value
...     while not task.is_terminal():
...         task = queue.task_from_id(task.task_id)
...         time.sleep(5)

Limitations vs real EQSQL
-------------------------

- ``task-type`` must be ``"bash-script"``. Other types (``python-script``,
  ``python-module``, ``bash-module``, ``control-task``) are not supported.
- ``worker_id`` regex is accepted but only used to tag the row in the
  store; there's no worker pool to route to.
- ``priority`` is recorded but Slurm priority handling depends on your
  cluster's QOS configuration.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
)
from sqlalchemy.engine import Engine

from polarisopt.runners.base import JobSpec, JobStatus
from polarisopt.runners.slurm import SlurmResources, SlurmRunner
from polarisopt.utils.logging import get_logger

log = get_logger(__name__)


# ---- status strings (match polarislib eqsql.eq_db exactly) ----

QUEUED = "queued"
RUNNING = "running"
FINISHED = "finished"
FAILED = "failed"
CANCELLING = "cancelling"
CANCELLED = "cancelled"

_TERMINAL_STATUSES = {FINISHED, FAILED, CANCELLED}


class TaskStatus(StrEnum):
    QUEUED = QUEUED
    RUNNING = RUNNING
    FINISHED = FINISHED
    FAILED = FAILED
    CANCELLING = CANCELLING
    CANCELLED = CANCELLED


# ---- result wrapper (matches polarislib's DBResult) ----


@dataclass
class Result:
    """Mirrors ``polaris.hpc.eqsql.eq_db.DBResult``."""

    succeeded: bool
    value: Task | None = None
    reason: str | None = None

    @classmethod
    def success(cls, value: Task) -> Result:
        return cls(succeeded=True, value=value)

    @classmethod
    def failure(cls, reason: str) -> Result:
        return cls(succeeded=False, reason=reason)


# ---- task ----


@dataclass
class Task:
    """Mirrors the public fields of ``polaris.hpc.eqsql.task.Task``."""

    task_id: int
    exp_id: str
    worker_id: str | None
    priority: int
    definition: dict[str, Any]
    input: str | None
    output: str | None
    status: str
    message: str | None
    slurm_jobid: str | None
    created_at: datetime
    updated_at: datetime
    _queue: TaskQueue | None = field(default=None, repr=False, compare=False)

    def is_terminal(self) -> bool:
        return self.status in _TERMINAL_STATUSES

    def cancel(self) -> Task:
        """Cancel the underlying Slurm job and mark the task cancelled."""
        if self._queue is None:
            raise RuntimeError("Task is detached from its queue; use TaskQueue.cancel(task)")
        return self._queue.cancel(self)

    def get_logs(self) -> list[dict[str, Any]]:
        """Return the task log entries (mirrors ``Task.get_logs`` shape, list of dicts)."""
        if self._queue is None:
            raise RuntimeError("Task is detached from its queue; use TaskQueue.task_logs(task)")
        return self._queue.task_logs(self)


# ---- internal storage ----

_metadata = MetaData()

_tasks_table = Table(
    "eqsql_compat_tasks",
    _metadata,
    Column("task_id", Integer, primary_key=True, autoincrement=True),
    Column("exp_id", String, nullable=False),
    Column("worker_id", String, nullable=True),
    Column("priority", Integer, nullable=False, default=1),
    Column("definition_json", Text, nullable=False),
    Column("input_json", Text, nullable=True),
    Column("output", Text, nullable=True),
    Column("status", String, nullable=False),
    Column("message", Text, nullable=True),
    Column("slurm_jobid", String, nullable=True),
    Column("script_path", String, nullable=True),
    Column("stdout_path", String, nullable=True),
    Column("stderr_path", String, nullable=True),
    Column("created_at", DateTime, nullable=False),
    Column("updated_at", DateTime, nullable=False),
)

_task_log_table = Table(
    "eqsql_compat_task_log",
    _metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("task_id", Integer, nullable=False),
    Column("message", Text, nullable=False),
    Column("created_at", DateTime, nullable=False),
)


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


# ---- queue ----


class TaskQueue:
    """Lightweight task queue: SQLite store + Slurm runner."""

    def __init__(
        self,
        *,
        engine: Engine,
        runner: SlurmRunner,
        workspace: Path,
    ) -> None:
        self._engine = engine
        self._runner = runner
        self._workspace = workspace
        self._lock = threading.Lock()
        _metadata.create_all(engine)

    # ----- insertion -----

    def insert_task(
        self,
        *,
        definition: dict[str, Any],
        input: str | dict[str, Any] | None = None,
        exp_id: str,
        worker_id: str | None = None,
        task_type: int = 1,  # noqa: ARG002 — accepted for API parity
        priority: int = 1,
        resources: SlurmResources | None = None,
    ) -> Result:
        """Insert + sbatch a task. Returns a :class:`Result` carrying the :class:`Task`."""
        try:
            command = self._extract_command(definition)
        except ValueError as exc:
            return Result.failure(str(exc))

        with self._lock:
            with self._engine.begin() as conn:
                now = _now()
                row_id = conn.execute(
                    _tasks_table.insert().values(
                        exp_id=exp_id,
                        worker_id=worker_id,
                        priority=priority,
                        definition_json=json.dumps(definition),
                        input_json=_input_to_json(input),
                        status=QUEUED,
                        created_at=now,
                        updated_at=now,
                    )
                ).inserted_primary_key[0]

            scripts_dir = self._workspace / "scripts"
            logs_dir = self._workspace / "logs"
            scripts_dir.mkdir(parents=True, exist_ok=True)
            logs_dir.mkdir(parents=True, exist_ok=True)
            stdout = logs_dir / f"task-{row_id}.out"
            stderr = logs_dir / f"task-{row_id}.err"

            spec = JobSpec(
                name=f"{exp_id}-{row_id}",
                command=command,
                cwd=scripts_dir,
                stdout=stdout,
                stderr=stderr,
                extra={"resources": resources} if resources else {},
            )
            try:
                slurm_job = self._runner.submit(spec)
            except Exception as exc:
                log.exception("sbatch failed for task %d", row_id)
                with self._engine.begin() as conn:
                    conn.execute(
                        _tasks_table.update()
                        .where(_tasks_table.c.task_id == row_id)
                        .values(status=FAILED, message=str(exc), updated_at=_now())
                    )
                return Result.failure(str(exc))

            with self._engine.begin() as conn:
                conn.execute(
                    _tasks_table.update()
                    .where(_tasks_table.c.task_id == row_id)
                    .values(
                        slurm_jobid=slurm_job.task_id,
                        script_path=str(slurm_job.script_path) if slurm_job.script_path else None,
                        stdout_path=str(stdout),
                        stderr_path=str(stderr),
                        updated_at=_now(),
                    )
                )

        self._log(row_id, f"submitted as slurm jobid {slurm_job.task_id}")
        return Result.success(self.task_from_id(int(row_id)))

    # ----- lookup -----

    def task_from_id(self, task_id: int) -> Task:
        """Refresh from Slurm and return the :class:`Task`."""
        with self._engine.begin() as conn:
            row = conn.execute(
                _tasks_table.select().where(_tasks_table.c.task_id == task_id)
            ).mappings().first()
        if row is None:
            raise KeyError(f"No task with task_id={task_id}")

        # If not yet terminal, ask Slurm for the latest state.
        # If Slurm can't tell us (e.g. squeue empty + sacct empty for a just-submitted job),
        # we keep the previous status — never speculatively promote queued→running.
        if row["status"] not in _TERMINAL_STATUSES and row["slurm_jobid"]:
            new_status, message, exit_code = self._poll_slurm(row["slurm_jobid"])
            if new_status is not None and (
                new_status != row["status"] or (message and message != row["message"])
            ):
                with self._engine.begin() as conn:
                    conn.execute(
                        _tasks_table.update()
                        .where(_tasks_table.c.task_id == task_id)
                        .values(status=new_status, message=message, updated_at=_now())
                    )
                if new_status in _TERMINAL_STATUSES:
                    self._log(task_id, f"terminal via slurm: {new_status} (exit={exit_code})")
                with self._engine.begin() as conn:
                    row = conn.execute(
                        _tasks_table.select().where(_tasks_table.c.task_id == task_id)
                    ).mappings().first()

        return _row_to_task(row, queue=self)

    def list_tasks(self, *, exp_id: str | None = None) -> list[Task]:
        stmt = _tasks_table.select().order_by(_tasks_table.c.task_id)
        if exp_id is not None:
            stmt = stmt.where(_tasks_table.c.exp_id == exp_id)
        with self._engine.begin() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [_row_to_task(r, queue=self) for r in rows]

    # ----- cancellation -----

    def cancel(self, task: Task) -> Task:
        if task.is_terminal():
            return task
        if task.slurm_jobid:
            self._runner.cancel(_synthetic_job(task.slurm_jobid))
        with self._engine.begin() as conn:
            conn.execute(
                _tasks_table.update()
                .where(_tasks_table.c.task_id == task.task_id)
                .values(status=CANCELLED, message="cancelled via TaskQueue.cancel", updated_at=_now())
            )
        self._log(task.task_id, "cancelled by user")
        return self.task_from_id(task.task_id)

    # ----- logs -----

    def task_logs(self, task: Task) -> list[dict[str, Any]]:
        with self._engine.begin() as conn:
            rows = conn.execute(
                _task_log_table.select()
                .where(_task_log_table.c.task_id == task.task_id)
                .order_by(_task_log_table.c.id)
            ).mappings().all()
        return [{"created_at": r["created_at"], "message": r["message"]} for r in rows]

    # ----- internals -----

    def _log(self, task_id: int, message: str) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                _task_log_table.insert().values(
                    task_id=task_id,
                    message=message,
                    created_at=_now(),
                )
            )

    def _poll_slurm(self, jobid: str) -> tuple[str | None, str | None, int | None]:
        """Return (eqsql_status, message, exit_code). status is None if indeterminate."""
        job = _synthetic_job(jobid)
        refreshed = self._runner.status(job)
        return _slurm_status_to_eqsql(refreshed.status), refreshed.message, refreshed.exit_code

    @staticmethod
    def _extract_command(definition: dict[str, Any]) -> str:
        task_type = definition.get("task-type")
        if task_type != "bash-script":
            raise ValueError(
                f"polarisopt.compat.eqsql only supports task-type='bash-script'; got {task_type!r}"
            )
        command = definition.get("command")
        if not command or not isinstance(command, str):
            raise ValueError("bash-script definition must include a non-empty 'command' string")
        return command


# ---- factory ----


@contextmanager
def open_queue(
    workspace: Path | str,
    *,
    db_path: Path | str | None = None,
    runner: SlurmRunner | None = None,
    default_resources: SlurmResources | None = None,
) -> Iterator[TaskQueue]:
    """Context manager: open the SQLite queue at ``workspace/eqsql_compat.db``."""
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    db = Path(db_path) if db_path else workspace / "eqsql_compat.db"
    engine = create_engine(f"sqlite:///{db}", future=True)
    actual_runner = runner or SlurmRunner(default_resources=default_resources)
    try:
        yield TaskQueue(engine=engine, runner=actual_runner, workspace=workspace)
    finally:
        engine.dispose()


# ---- module-level convenience (mirrors polarislib's top-level functions) ----

_DEFAULT_QUEUE: TaskQueue | None = None


def configure_default_queue(queue: TaskQueue) -> None:
    """Install a queue as the module default so ``insert_task``/``task_from_id``
    at module scope behave like polarislib's free functions."""
    global _DEFAULT_QUEUE
    _DEFAULT_QUEUE = queue


def insert_task(
    *,
    definition: dict[str, Any],
    input: str | dict[str, Any] | None = None,
    exp_id: str,
    worker_id: str | None = None,
    task_type: int = 1,
    priority: int = 1,
    resources: SlurmResources | None = None,
) -> Result:
    """Module-level convenience — requires :func:`configure_default_queue` first."""
    if _DEFAULT_QUEUE is None:
        raise RuntimeError(
            "No default queue configured. Call polarisopt.compat.eqsql.configure_default_queue() "
            "or use open_queue() as a context manager."
        )
    return _DEFAULT_QUEUE.insert_task(
        definition=definition,
        input=input,
        exp_id=exp_id,
        worker_id=worker_id,
        task_type=task_type,
        priority=priority,
        resources=resources,
    )


def task_from_id(task_id: int) -> Task:
    if _DEFAULT_QUEUE is None:
        raise RuntimeError("No default queue configured")
    return _DEFAULT_QUEUE.task_from_id(task_id)


# ---- helpers ----


def _input_to_json(value: str | dict[str, Any] | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value)


def _slurm_status_to_eqsql(s: JobStatus) -> str | None:
    """Map a SlurmRunner JobStatus to an EQSQL status string.

    Returns ``None`` for UNKNOWN: the caller should keep whatever status
    the task currently holds rather than speculate.
    """
    return {
        JobStatus.PENDING: QUEUED,
        JobStatus.QUEUED: QUEUED,
        JobStatus.RUNNING: RUNNING,
        JobStatus.FINISHED: FINISHED,
        JobStatus.FAILED: FAILED,
        JobStatus.CANCELLED: CANCELLED,
        JobStatus.UNKNOWN: None,
    }[s]


def _synthetic_job(jobid: str):
    """Build a minimal Slurm Job stub so SlurmRunner.status/cancel can act on a bare jobid."""
    from polarisopt.runners.slurm import SlurmJob

    return SlurmJob(
        spec=JobSpec(name="compat", command="", cwd=Path(".")),
        task_id=jobid,
    )


def _row_to_task(row: dict[str, Any], *, queue: TaskQueue) -> Task:
    return Task(
        task_id=row["task_id"],
        exp_id=row["exp_id"],
        worker_id=row["worker_id"],
        priority=row["priority"],
        definition=json.loads(row["definition_json"]),
        input=row["input_json"],
        output=row["output"],
        status=row["status"],
        message=row["message"],
        slurm_jobid=row["slurm_jobid"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        _queue=queue,
    )
