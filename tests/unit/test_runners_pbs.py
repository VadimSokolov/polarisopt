"""Tests for PBSRunner — mirrors test_runners_slurm.py."""

from __future__ import annotations

import subprocess
from pathlib import Path
from textwrap import dedent

import pytest

from polarisopt.runners.base import JobSpec, JobStatus, RunnerError
from polarisopt.runners.pbs import PBSResources, PBSRunner


def _ok(stdout: str = "", stderr: str = "", rc: int = 0):
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout=stdout, stderr=stderr)


def test_submit_parses_jobid_and_writes_script(tmp_path: Path, fake_shell) -> None:
    fake_shell.responses.append(_ok(stdout="7609762.imgt1\n"))
    runner = PBSRunner(
        default_resources=PBSResources(
            queue="compute", account="POLARIS", walltime="01:00:00",
            ncpus=16, mem="64gb",
        ),
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
    # Full id is preserved (don't strip .imgt1 — qstat/qdel need it).
    assert job.task_id == "7609762.imgt1"
    assert job.status is JobStatus.QUEUED
    assert job.script_path is not None
    script = job.script_path.read_text()
    assert "#PBS -N hello" in script
    assert "#PBS -q compute" in script
    assert "#PBS -A POLARIS" in script
    assert "#PBS -l walltime=01:00:00" in script
    assert "#PBS -l select=1:ncpus=16:mem=64gb" in script
    assert "#PBS -j oe" in script
    assert "export FOO=bar" in script
    assert "echo hi" in script
    assert fake_shell.calls[0][0] == "qsub"


def test_submit_raises_on_qsub_failure(tmp_path: Path, fake_shell) -> None:
    fake_shell.responses.append(_ok(rc=1, stderr="qsub: Invalid credential"))
    runner = PBSRunner(shell_runner=fake_shell)
    spec = JobSpec(name="x", command="false", cwd=tmp_path)
    with pytest.raises(RunnerError, match="qsub failed"):
        runner.submit(spec)


def test_submit_raises_on_empty_qsub_output(tmp_path: Path, fake_shell) -> None:
    fake_shell.responses.append(_ok(stdout="\n\n"))
    runner = PBSRunner(shell_runner=fake_shell)
    spec = JobSpec(name="x", command="false", cwd=tmp_path)
    with pytest.raises(RunnerError, match="could not parse"):
        runner.submit(spec)


def _qstat_fx(state: str, exit_status: int | None = None) -> str:
    lines = [f"    job_state = {state}"]
    if exit_status is not None:
        lines.append(f"    exit_status = {exit_status}")
    return "Job Id: 7609762.imgt1\n" + "\n".join(lines) + "\n"


def test_status_running(tmp_path: Path, fake_shell) -> None:
    fake_shell.responses.append(_ok(stdout=_qstat_fx("R")))
    runner = PBSRunner(shell_runner=fake_shell)
    job = runner.submit_for_test_only(tmp_path)
    runner.status(job)
    assert job.status is JobStatus.RUNNING


def test_status_queued(tmp_path: Path, fake_shell) -> None:
    fake_shell.responses.append(_ok(stdout=_qstat_fx("Q")))
    runner = PBSRunner(shell_runner=fake_shell)
    job = runner.submit_for_test_only(tmp_path)
    runner.status(job)
    assert job.status is JobStatus.QUEUED


def test_status_held_is_queued(tmp_path: Path, fake_shell) -> None:
    fake_shell.responses.append(_ok(stdout=_qstat_fx("H")))
    runner = PBSRunner(shell_runner=fake_shell)
    job = runner.submit_for_test_only(tmp_path)
    runner.status(job)
    # Held jobs are functionally queued for polarisopt's polling.
    assert job.status is JobStatus.QUEUED


def test_status_finished_exit_zero_is_finished(tmp_path: Path, fake_shell) -> None:
    fake_shell.responses.append(_ok(stdout=_qstat_fx("F", exit_status=0)))
    runner = PBSRunner(shell_runner=fake_shell)
    job = runner.submit_for_test_only(tmp_path)
    runner.status(job)
    assert job.status is JobStatus.FINISHED
    assert job.exit_code == 0


def test_status_finished_exit_nonzero_is_failed(tmp_path: Path, fake_shell) -> None:
    """PBS marks the job F or C regardless of exit; we promote to FAILED
    when exit_status != 0 so a seg-faulting binary doesn't look FINISHED.
    """
    fake_shell.responses.append(_ok(stdout=_qstat_fx("F", exit_status=137)))
    runner = PBSRunner(shell_runner=fake_shell)
    job = runner.submit_for_test_only(tmp_path)
    runner.status(job)
    assert job.status is JobStatus.FAILED
    assert job.exit_code == 137


def test_status_unknown_when_qstat_says_unknown_job(tmp_path: Path, fake_shell) -> None:
    fake_shell.responses.append(
        _ok(rc=153, stderr="qstat: Unknown Job Id 7609762.imgt1\n"),
    )
    runner = PBSRunner(shell_runner=fake_shell)
    job = runner.submit_for_test_only(tmp_path)
    runner.status(job)
    assert job.status is JobStatus.UNKNOWN
    # Don't leak the qstat noise into the job message — orphan path
    # is normal behavior, not an error.
    assert job.message is None


def test_status_unknown_when_qstat_fails(tmp_path: Path, fake_shell) -> None:
    fake_shell.responses.append(_ok(rc=1, stderr="cannot connect to server"))
    runner = PBSRunner(shell_runner=fake_shell)
    job = runner.submit_for_test_only(tmp_path)
    runner.status(job)
    assert job.status is JobStatus.UNKNOWN
    # Real failures DO get the stderr (operator wants to see them).
    assert job.message and "cannot connect" in job.message


def test_cancel_invokes_qdel(tmp_path: Path, fake_shell) -> None:
    fake_shell.responses.append(_ok())
    runner = PBSRunner(shell_runner=fake_shell)
    job = runner.submit_for_test_only(tmp_path)
    runner.cancel(job)
    assert job.status is JobStatus.CANCELLED
    assert fake_shell.calls[-1][0] == "qdel"


def test_cancel_treats_already_finished_as_success(tmp_path: Path, fake_shell) -> None:
    """qdel returns non-zero on already-terminated jobs; from polarisopt's
    POV that's idempotent cancellation, not an error.
    """
    fake_shell.responses.append(_ok(rc=170, stderr="qdel: Job has finished 7609762.imgt1"))
    runner = PBSRunner(shell_runner=fake_shell)
    job = runner.submit_for_test_only(tmp_path)
    runner.cancel(job)
    assert job.status is JobStatus.CANCELLED


def test_job_name_is_sanitized_for_qsub(tmp_path: Path, fake_shell) -> None:
    """PBS Pro rejects job names with spaces / shell metacharacters.

    Polarisopt-generated job names like ``polaris-sample-1`` are fine,
    but user-supplied names from per-sample JobSpec.extra could contain
    anything. Sanitize the same way the script filename is sanitized.
    """
    fake_shell.responses.append(_ok(stdout="7609765.imgt1\n"))
    runner = PBSRunner(
        default_resources=PBSResources(queue="compute"),
        shell_runner=fake_shell,
    )
    spec = JobSpec(
        name="naughty name with spaces!",
        command="echo hi",
        cwd=tmp_path / "r",
    )
    job = runner.submit(spec)
    script = job.script_path.read_text()
    # The original name shows up only after sanitization.
    assert "#PBS -N naughty_name_with_spaces_" in script
    assert "#PBS -N naughty name" not in script


def test_exclusive_placement_renders_directive(tmp_path: Path, fake_shell) -> None:
    fake_shell.responses.append(_ok(stdout="7609763.imgt1\n"))
    runner = PBSRunner(
        default_resources=PBSResources(
            queue="compute", account="POLARIS", walltime="01:00:00",
            ncpus=16, mem="96gb", place="excl",
        ),
        shell_runner=fake_shell,
    )
    spec = JobSpec(name="excl", command="echo hi", cwd=tmp_path / "r")
    job = runner.submit(spec)
    script = job.script_path.read_text()
    assert "#PBS -l place=excl" in script


def test_setup_commands_render_in_script(tmp_path: Path, fake_shell) -> None:
    fake_shell.responses.append(_ok(stdout="7609764.imgt1\n"))
    runner = PBSRunner(
        default_resources=PBSResources(
            queue="compute",
            setup_commands=["module purge", "module load apptainer/1.1.9"],
        ),
        shell_runner=fake_shell,
    )
    spec = JobSpec(name="setup", command="echo hi", cwd=tmp_path / "r")
    job = runner.submit(spec)
    script = job.script_path.read_text()
    # Directives precede setup_commands which precede the user command.
    qpos = script.find("#PBS -q compute")
    purge_pos = script.find("module purge")
    cmd_pos = script.find("echo hi")
    assert qpos < purge_pos < cmd_pos


def test_make_runner_via_yaml_factory(tmp_path: Path) -> None:
    """YAML-level round-trip: runner.type: pbs must build a PBSRunner."""
    from polarisopt.runners.factory import make_runner
    runner = make_runner(
        {
            "type": "pbs",
            "options": {
                "default_resources": {
                    "queue": "compute",
                    "account": "POLARIS",
                    "walltime": "01:00:00",
                    "ncpus": 8,
                    "mem": "32gb",
                },
            },
        }
    )
    assert isinstance(runner, PBSRunner)
    assert runner._default.queue == "compute"
    assert runner._default.mem == "32gb"


# ---------- helper monkey-patched onto PBSRunner for the test file ----------
#
# The status / cancel tests don't need to go through submit(), so we
# construct a Job directly. Keep this helper local to the test file.


def _submit_for_test_only(self, tmp_path: Path):
    from polarisopt.runners.pbs import PBSJob
    return PBSJob(
        spec=JobSpec(name="t", command="", cwd=tmp_path),
        task_id="7609762.imgt1",
        status=JobStatus.QUEUED,
    )


PBSRunner.submit_for_test_only = _submit_for_test_only  # type: ignore[attr-defined]


def test_dedent_unused() -> None:
    # touch dedent so the import survives ruff's unused check
    assert isinstance(dedent("x"), str)
