from __future__ import annotations

import time
from pathlib import Path

import pytest

from polarisopt.runners.base import JobSpec, JobStatus
from polarisopt.runners.local import LocalRunner


def _wait_terminal(runner: LocalRunner, job, timeout: float = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        runner.status(job)
        if job.status.is_terminal():
            return
        time.sleep(0.05)
    pytest.fail(f"Job did not reach terminal status in time; last={job.status}")


def test_local_runner_succeeds(tmp_path: Path) -> None:
    runner = LocalRunner()
    out = tmp_path / "out"
    spec = JobSpec(name="echo", command="echo hi", cwd=tmp_path, stdout=out)
    job = runner.submit(spec)
    _wait_terminal(runner, job)
    assert job.status == JobStatus.FINISHED
    assert job.exit_code == 0
    assert "hi" in out.read_text()


def test_local_runner_failure(tmp_path: Path) -> None:
    runner = LocalRunner()
    job = runner.submit(JobSpec(name="fail", command="false", cwd=tmp_path))
    _wait_terminal(runner, job)
    assert job.status == JobStatus.FAILED
    assert job.exit_code != 0


def test_local_runner_cancel(tmp_path: Path) -> None:
    runner = LocalRunner()
    job = runner.submit(JobSpec(name="sleep", command="sleep 30", cwd=tmp_path))
    # let it start
    time.sleep(0.1)
    runner.cancel(job)
    assert job.status == JobStatus.CANCELLED
