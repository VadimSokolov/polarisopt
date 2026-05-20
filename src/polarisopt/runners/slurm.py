"""Slurm runner — sbatch / squeue / scancel.

Submits a generated sbatch script. Status is polled via ``squeue`` for
in-flight jobs and ``sacct`` for terminal jobs (squeue forgets them).

For testability, all shell interactions go through ``_run`` which can be
replaced in tests via the ``shell_runner`` constructor argument.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from polarisopt.runners.base import (
    Job,
    JobSpec,
    JobStatus,
    Runner,
    RunnerError,
    runner_registry,
)
from polarisopt.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class SlurmResources:
    """Resource hints written into the sbatch script header.

    Anything you'd put after ``#SBATCH``. Use the dataclass for common knobs
    and ``extra_directives`` for the long tail (``--qos``, ``--mail-user`` ...).

    ``setup_commands`` is a list of shell commands run **after** the
    directives but **before** the user command. Useful for ``module load``,
    activating a virtualenv, sourcing env files, etc. — anything that
    can't be cleanly represented as a static env dict.
    """

    partition: str | None = None
    account: str | None = None
    time: str | None = None
    nodes: int | None = None
    ntasks: int | None = None
    cpus_per_task: int | None = None
    mem: str | None = None
    extra_directives: list[str] = field(default_factory=list)
    setup_commands: list[str] = field(default_factory=list)

    def to_directives(self, *, job_name: str, stdout: Path | None, stderr: Path | None) -> list[str]:
        lines: list[str] = [f"#SBATCH --job-name={job_name}"]
        if self.partition:
            lines.append(f"#SBATCH --partition={self.partition}")
        if self.account:
            lines.append(f"#SBATCH --account={self.account}")
        if self.time:
            lines.append(f"#SBATCH --time={self.time}")
        if self.nodes is not None:
            lines.append(f"#SBATCH --nodes={self.nodes}")
        if self.ntasks is not None:
            lines.append(f"#SBATCH --ntasks={self.ntasks}")
        if self.cpus_per_task is not None:
            lines.append(f"#SBATCH --cpus-per-task={self.cpus_per_task}")
        if self.mem:
            lines.append(f"#SBATCH --mem={self.mem}")
        if stdout:
            lines.append(f"#SBATCH --output={stdout}")
        if stderr:
            lines.append(f"#SBATCH --error={stderr}")
        lines.extend(self.extra_directives)
        return lines


@dataclass
class SlurmJob(Job):
    """Job with the path of the generated sbatch script (handy for debugging)."""

    script_path: Path | None = None


ShellRunner = Callable[[list[str]], "subprocess.CompletedProcess[str]"]


def _default_shell_runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


_SBATCH_JOBID_RE = re.compile(r"Submitted batch job\s+(\d+)")


@runner_registry.register("slurm")
class SlurmRunner(Runner):
    """Submit jobs via ``sbatch``; track them via ``squeue``/``sacct``.

    Each :meth:`submit` writes a generated sbatch script next to the job's
    ``cwd`` and submits it. Per-job resource overrides flow in via
    ``JobSpec.extra['resources']``; otherwise the ``default_resources``
    supplied at construction time apply.

    Parameters
    ----------
    default_resources : SlurmResources or None
        Cluster resources applied to every submission unless overridden
        per JobSpec. ``None`` → empty defaults (you must override per-job).
    shell_runner : callable, optional
        Function that takes ``list[str]`` argv and returns a
        :class:`subprocess.CompletedProcess[str]`. Defaults to
        :func:`subprocess.run`. Tests inject a fake here so the suite
        runs without a real Slurm cluster.

    Raises
    ------
    RunnerError
        If ``sbatch`` exits nonzero or its stdout cannot be parsed for
        a jobid.

    Examples
    --------
    >>> from polarisopt.runners.slurm import SlurmRunner, SlurmResources
    >>> runner = SlurmRunner(default_resources=SlurmResources(           # doctest: +SKIP
    ...     partition="bdwall",
    ...     account="POLARIS",
    ...     time="02:00:00",
    ...     cpus_per_task=16,
    ...     mem="64G",
    ... ))
    """

    def __init__(
        self,
        *,
        default_resources: SlurmResources | None = None,
        shell_runner: ShellRunner | None = None,
    ) -> None:
        self._default = default_resources or SlurmResources()
        self._shell = shell_runner or _default_shell_runner

    def submit(self, spec: JobSpec) -> SlurmJob:
        spec.cwd.mkdir(parents=True, exist_ok=True)
        resources = spec.extra.get("resources", self._default)
        if not isinstance(resources, SlurmResources):
            raise TypeError("JobSpec.extra['resources'] must be a SlurmResources")
        script_path = spec.cwd / f"{_safe_name(spec.name)}.slurm"
        script_path.write_text(self._render_script(spec, resources))

        result = self._shell(["sbatch", str(script_path)])
        if result.returncode != 0:
            raise RunnerError(f"sbatch failed (rc={result.returncode}): {result.stderr.strip() or result.stdout.strip()}")
        match = _SBATCH_JOBID_RE.search(result.stdout)
        if not match:
            raise RunnerError(f"could not parse sbatch output: {result.stdout!r}")
        jobid = match.group(1)
        log.info("SlurmRunner submitted %s as jobid %s", spec.name, jobid)
        return SlurmJob(spec=spec, task_id=jobid, status=JobStatus.QUEUED, script_path=script_path)

    def status(self, job: Job) -> Job:
        # squeue first — it knows about active jobs
        result = self._shell(["squeue", "-h", "-j", job.task_id, "-o", "%T"])
        if result.returncode == 0 and result.stdout.strip():
            state = result.stdout.strip().split()[0]
            job.status = _squeue_state_to_status(state)
            return job
        # squeue forgets terminal jobs — fall back to sacct
        sacct = self._shell(["sacct", "-j", job.task_id, "-X", "-n", "-P", "-o", "State,ExitCode"])
        if sacct.returncode != 0:
            job.status = JobStatus.UNKNOWN
            job.message = (sacct.stderr or "").strip() or "sacct failed"
            return job
        first_line = sacct.stdout.strip().splitlines()[0] if sacct.stdout.strip() else ""
        if not first_line:
            job.status = JobStatus.UNKNOWN
            return job
        state, _, exit_str = first_line.partition("|")
        job.status = _sacct_state_to_status(state.strip())
        if "|" in first_line:
            job.exit_code = _parse_exit_code(exit_str.strip())
        return job

    def cancel(self, job: Job) -> Job:
        result = self._shell(["scancel", job.task_id])
        if result.returncode != 0:
            job.message = f"scancel failed: {result.stderr.strip()}"
            return job
        job.status = JobStatus.CANCELLED
        return job

    def _render_script(self, spec: JobSpec, resources: SlurmResources) -> str:
        directives = resources.to_directives(job_name=spec.name, stdout=spec.stdout, stderr=spec.stderr)
        env_lines = [f"export {k}={shlex.quote(v)}" for k, v in spec.env.items()]
        # All #SBATCH directives MUST precede the first executable line — slurm
        # stops parsing directives at the first non-comment line.  Keep
        # `set -euo pipefail` after the directive block.
        lines = ["#!/bin/bash", *directives, "", "set -euo pipefail", ""]
        if env_lines:
            lines.extend(env_lines)
            lines.append("")
        if resources.setup_commands:
            lines.extend(resources.setup_commands)
            lines.append("")
        lines.append(f"cd {shlex.quote(str(spec.cwd))}")
        lines.append(spec.command)
        return "\n".join(lines) + "\n"


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def _squeue_state_to_status(state: str) -> JobStatus:
    s = state.upper()
    if s in {"PENDING", "PD", "CONFIGURING", "CF"}:
        return JobStatus.QUEUED
    if s in {"RUNNING", "R", "COMPLETING", "CG"}:
        return JobStatus.RUNNING
    if s in {"COMPLETED", "CD"}:
        return JobStatus.FINISHED
    if s in {"FAILED", "F", "TIMEOUT", "TO", "NODE_FAIL", "NF", "BOOT_FAIL", "BF", "OUT_OF_MEMORY", "OOM"}:
        return JobStatus.FAILED
    if s in {"CANCELLED", "CA"}:
        return JobStatus.CANCELLED
    return JobStatus.UNKNOWN


def _sacct_state_to_status(state: str) -> JobStatus:
    s = state.upper().split()[0]  # sacct may print "CANCELLED by 12345"
    if s == "COMPLETED":
        return JobStatus.FINISHED
    if s == "RUNNING":
        return JobStatus.RUNNING
    if s in {"PENDING"}:
        return JobStatus.QUEUED
    if s == "CANCELLED":
        return JobStatus.CANCELLED
    if s in {"FAILED", "TIMEOUT", "NODE_FAIL", "BOOT_FAIL", "OUT_OF_MEMORY"}:
        return JobStatus.FAILED
    return JobStatus.UNKNOWN


def _parse_exit_code(s: str) -> int | None:
    """sacct ExitCode is ``<code>:<signal>``; we report ``<code>``."""
    if not s:
        return None
    head = s.split(":")[0]
    try:
        return int(head)
    except ValueError:
        return None
