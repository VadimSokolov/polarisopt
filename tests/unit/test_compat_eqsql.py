from __future__ import annotations

import subprocess
from pathlib import Path

from polarisopt.compat import eqsql
from polarisopt.compat.eqsql import (
    CANCELLED,
    FINISHED,
    QUEUED,
    RUNNING,
    Result,
    TaskQueue,
)
from polarisopt.runners.slurm import SlurmRunner


def _ok(stdout: str = "", stderr: str = "", rc: int = 0):
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout=stdout, stderr=stderr)


def _make_queue(tmp_path: Path, shell) -> TaskQueue:
    runner = SlurmRunner(shell_runner=shell)
    # Build the queue manually so we share the workspace dir explicitly
    from sqlalchemy import create_engine

    engine = create_engine(f"sqlite:///{tmp_path / 'q.db'}", future=True)
    return TaskQueue(engine=engine, runner=runner, workspace=tmp_path)


def test_insert_task_routes_to_sbatch(tmp_path: Path, fake_shell) -> None:
    fake_shell.responses.append(_ok(stdout="Submitted batch job 1234\n"))
    queue = _make_queue(tmp_path, fake_shell)

    result = queue.insert_task(
        definition={"task-type": "bash-script", "command": "/bin/echo hello"},
        input={"foo": "bar"},
        exp_id="hello",
        worker_id="xover.vsokolov.*",
    )
    assert isinstance(result, Result)
    assert result.succeeded, result.reason
    task = result.value
    assert task is not None
    assert task.exp_id == "hello"
    assert task.worker_id == "xover.vsokolov.*"
    assert task.slurm_jobid == "1234"
    assert task.status == QUEUED


def test_insert_task_rejects_non_bash_script(tmp_path: Path, fake_shell) -> None:
    queue = _make_queue(tmp_path, fake_shell)
    result = queue.insert_task(
        definition={"task-type": "python-script", "script": "foo.py"},
        exp_id="x",
    )
    assert not result.succeeded
    assert result.reason and "bash-script" in result.reason


def test_task_from_id_polls_slurm_and_transitions_to_finished(tmp_path: Path, fake_shell) -> None:
    # First call: sbatch submits
    # Subsequent: squeue (forgot) then sacct returns COMPLETED

    def handler(cmd):
        head = cmd[0]
        if head == "sbatch":
            return _ok(stdout="Submitted batch job 77\n")
        if head == "squeue":
            return _ok(stdout="")  # forgot
        if head == "sacct":
            return _ok(stdout="COMPLETED|0:0\n")
        return _ok()

    fake_shell.set_handler(handler)
    queue = _make_queue(tmp_path, fake_shell)
    task = queue.insert_task(
        definition={"task-type": "bash-script", "command": "true"},
        exp_id="exp",
    ).value
    assert task is not None

    refreshed = queue.task_from_id(task.task_id)
    assert refreshed.status == FINISHED


def test_cancel_calls_scancel(tmp_path: Path, fake_shell) -> None:
    def handler(cmd):
        if cmd[0] == "sbatch":
            return _ok(stdout="Submitted batch job 5\n")
        return _ok()

    fake_shell.set_handler(handler)
    queue = _make_queue(tmp_path, fake_shell)
    task = queue.insert_task(
        definition={"task-type": "bash-script", "command": "sleep 9999"},
        exp_id="cancel-me",
    ).value
    assert task is not None

    cancelled = queue.cancel(task)
    assert cancelled.status == CANCELLED
    # scancel should have been called with the slurm jobid
    assert any(c[0] == "scancel" and c[1] == "5" for c in fake_shell.calls)


def test_task_logs_recorded(tmp_path: Path, fake_shell) -> None:
    fake_shell.set_handler(lambda cmd: _ok(stdout="Submitted batch job 9\n") if cmd[0] == "sbatch" else _ok())
    queue = _make_queue(tmp_path, fake_shell)
    task = queue.insert_task(
        definition={"task-type": "bash-script", "command": "echo"},
        exp_id="logs",
    ).value
    assert task is not None

    logs = queue.task_logs(task)
    assert len(logs) >= 1
    assert any("slurm jobid 9" in log["message"] for log in logs)


def test_open_queue_context_manager(tmp_path: Path, fake_shell) -> None:
    fake_shell.set_handler(lambda cmd: _ok(stdout="Submitted batch job 1\n") if cmd[0] == "sbatch" else _ok())
    with eqsql.open_queue(tmp_path, runner=SlurmRunner(shell_runner=fake_shell)) as queue:
        result = queue.insert_task(
            definition={"task-type": "bash-script", "command": "echo"},
            exp_id="ctx",
        )
        assert result.succeeded


def test_status_string_constants_match_polarislib() -> None:
    # The whole point of this shim: these must match polarislib.hpc.eqsql.eq_db
    assert QUEUED == "queued"
    assert RUNNING == "running"
    assert FINISHED == "finished"
    assert CANCELLED == "cancelled"
