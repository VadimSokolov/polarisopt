from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from polarisopt.runners.base import JobSpec, JobStatus, RunnerError
from polarisopt.runners.slurm import SlurmResources, SlurmRunner


def _ok(stdout: str = "", stderr: str = "", rc: int = 0):
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout=stdout, stderr=stderr)


def test_submit_parses_jobid_and_writes_script(tmp_path: Path, fake_shell) -> None:
    fake_shell.responses.append(_ok(stdout="Submitted batch job 4242\n"))
    runner = SlurmRunner(
        default_resources=SlurmResources(partition="bdwall", time="01:00:00", cpus_per_task=4),
        shell_runner=fake_shell,
    )
    spec = JobSpec(
        name="hello",
        command="echo hi",
        cwd=tmp_path / "run",
        stdout=tmp_path / "run" / "out",
        stderr=tmp_path / "run" / "err",
        env={"FOO": "bar"},
    )

    job = runner.submit(spec)

    assert job.task_id == "4242"
    assert job.status == JobStatus.QUEUED
    assert job.script_path is not None
    script = job.script_path.read_text()
    assert "#SBATCH --job-name=hello" in script
    assert "#SBATCH --partition=bdwall" in script
    assert "#SBATCH --time=01:00:00" in script
    assert "#SBATCH --cpus-per-task=4" in script
    assert "export FOO=bar" in script
    assert "echo hi" in script
    # invocation
    assert fake_shell.calls[0][0] == "sbatch"


def test_exclusive_flag_renders_sbatch_directive(tmp_path: Path, fake_shell) -> None:
    """``exclusive: true`` on SlurmResources must emit ``#SBATCH --exclusive``.

    Without this, polarisopt samples co-locate on the same node and the
    kernel OOM-kills them when their combined working set exceeds
    physical RAM — even when each per-job ``--mem`` is within its own
    limit. Crossover/TPS hit this on the DFW DTA smoke run.
    """
    fake_shell.responses.append(_ok(stdout="Submitted batch job 5050\n"))
    runner = SlurmRunner(
        default_resources=SlurmResources(
            partition="TPS", account="tps", exclusive=True, mem="64G",
        ),
        shell_runner=fake_shell,
    )
    spec = JobSpec(name="exclu", command="echo hi", cwd=tmp_path / "run")
    job = runner.submit(spec)
    script = job.script_path.read_text()
    assert "#SBATCH --exclusive" in script
    # And exclusive=False (default) doesn't emit it.
    fake_shell.responses.append(_ok(stdout="Submitted batch job 5051\n"))
    runner2 = SlurmRunner(
        default_resources=SlurmResources(partition="TPS", mem="64G"),
        shell_runner=fake_shell,
    )
    job2 = runner2.submit(
        JobSpec(name="shared", command="echo hi", cwd=tmp_path / "run2")
    )
    assert "#SBATCH --exclusive" not in job2.script_path.read_text()


def test_submit_retries_on_transient_qosgrp_limit(tmp_path: Path, fake_shell) -> None:
    """Slurm QOSGrpJobsLimit is transient (queue gets drained); v0.15
    retries with backoff instead of marking the sample FAILED."""
    fake_shell.responses.append(
        _ok(rc=1, stderr="sbatch: error: QOSGrpJobsLimit"),
    )
    fake_shell.responses.append(_ok(stdout="Submitted batch job 9999\n"))
    runner = SlurmRunner(shell_runner=fake_shell, submit_retry_backoff_s=(0, 0, 0))
    spec = JobSpec(name="x", command="echo hi", cwd=tmp_path / "r")
    job = runner.submit(spec)
    assert job.task_id == "9999"
    sbatch_calls = [c for c in fake_shell.calls if c[0] == "sbatch"]
    assert len(sbatch_calls) == 2


def test_submit_raises_on_sbatch_failure(tmp_path: Path, fake_shell) -> None:
    # Use a non-transient error so the v0.15 retry-with-backoff path
    # doesn't kick in. "Invalid partition" is a permanent submission
    # error, not something a retry would fix.
    fake_shell.responses.append(_ok(rc=1, stderr="sbatch: error: Invalid partition specified"))
    runner = SlurmRunner(shell_runner=fake_shell)
    spec = JobSpec(name="x", command="false", cwd=tmp_path)
    with pytest.raises(RunnerError, match="sbatch failed"):
        runner.submit(spec)


def test_status_via_squeue_running(tmp_path: Path, fake_shell) -> None:
    fake_shell.set_handler(lambda cmd: _ok(stdout="RUNNING\n") if cmd[0] == "squeue" else _ok())
    runner = SlurmRunner(shell_runner=fake_shell)
    spec = JobSpec(name="x", command="true", cwd=tmp_path)
    runner.submit.__wrapped__ if hasattr(runner.submit, "__wrapped__") else None  # placate ruff
    # build a Job by hand to bypass submit
    from polarisopt.runners.slurm import SlurmJob
    j = SlurmJob(spec=spec, task_id="100", status=JobStatus.QUEUED)
    runner.status(j)
    assert j.status == JobStatus.RUNNING


def test_status_falls_back_to_sacct_when_squeue_empty(tmp_path: Path, fake_shell) -> None:
    def handler(cmd):
        if cmd[0] == "squeue":
            return _ok(stdout="")  # squeue forgot
        if cmd[0] == "sacct":
            return _ok(stdout="COMPLETED|0:0\n")
        return _ok()

    fake_shell.set_handler(handler)
    runner = SlurmRunner(shell_runner=fake_shell)
    from polarisopt.runners.slurm import SlurmJob
    j = SlurmJob(spec=JobSpec(name="x", command="true", cwd=tmp_path), task_id="999")
    runner.status(j)
    assert j.status == JobStatus.FINISHED
    assert j.exit_code == 0


def test_status_unknown_when_both_fail(tmp_path: Path, fake_shell) -> None:
    def handler(cmd):
        if cmd[0] == "squeue":
            return _ok(stdout="")
        if cmd[0] == "sacct":
            return _ok(rc=1, stderr="db down")
        return _ok()

    fake_shell.set_handler(handler)
    runner = SlurmRunner(shell_runner=fake_shell)
    from polarisopt.runners.slurm import SlurmJob
    j = SlurmJob(spec=JobSpec(name="x", command="true", cwd=tmp_path), task_id="1")
    runner.status(j)
    assert j.status == JobStatus.UNKNOWN
    assert j.message == "db down"


def test_cancel_invokes_scancel(tmp_path: Path, fake_shell) -> None:
    fake_shell.responses.append(_ok())
    runner = SlurmRunner(shell_runner=fake_shell)
    from polarisopt.runners.slurm import SlurmJob
    j = SlurmJob(spec=JobSpec(name="x", command="true", cwd=tmp_path), task_id="42")
    runner.cancel(j)
    assert j.status == JobStatus.CANCELLED
    assert fake_shell.calls[0] == ["scancel", "42"]
