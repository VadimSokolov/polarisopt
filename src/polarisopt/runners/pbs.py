"""PBS Professional runner — parallel to SlurmRunner for clusters that use
PBS Pro instead of Slurm (LCRC Improv, Bebop, and many ANL partners).

YAML usage is identical to Slurm; only ``runner.type`` and the resource
field names change::

    runner:
      type: pbs
      options:
        default_resources:
          queue: compute
          account: POLARIS
          walltime: "04:00:00"
          select: 1
          ncpus: 16
          mem: 96gb                # lowercase — PBS Pro convention
          setup_commands:
            - "module load apptainer/1.1.9 miniforge3"

The submission lifecycle mirrors SlurmRunner: a generated ``.pbs``
script is written next to the job's ``cwd``, ``qsub`` returns
``<jobid>.<server>``, status is polled via ``qstat -fx`` (which covers
both live and historical jobs — no separate ``sacct`` step), and
``qdel`` cancels.

Status mapping (PBS ``job_state`` → polarisopt ``JobStatus``):

- ``Q`` (queued), ``H`` (held)           → ``QUEUED``
- ``R`` (running), ``E`` (exiting)       → ``RUNNING``
- ``C`` (complete), ``F`` (finished)     → ``FINISHED`` *unless*
  ``exit_status != 0`` in which case ``FAILED``
- ``S`` (suspended), unknown             → ``UNKNOWN``

LCRC-Improv specifics (verified 2026-06-17):

- Queues: ``compute`` (main), ``bigmem``, ``debug`` (use for tutorials),
  ``routing_queue``. Default to ``compute`` in production YAMLs.
- Account: ``POLARIS`` for the calibration project.
- Job IDs come back as ``<number>.<host>`` (e.g. ``7609762.imgt1``);
  polarisopt stores the full string and forwards it to ``qstat`` /
  ``qdel`` unchanged — don't strip the host suffix.

Module loads on Improv (matches ``polarislib/bin/hpc/worker_loop_lcrc.sh``
for that cluster)::

    module load gcc/11.4.0 hdf5/1.14.2-gcc-11.4.0 miniforge3 \\
                libspatialite apptainer
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
class PBSResources:
    """Resource hints written into the qsub script header.

    Anything you'd put after ``#PBS``. Use the dataclass for common
    knobs and ``extra_directives`` for the long tail (``-W``, ``-m``,
    ``-M``, ``-r``, …).

    ``setup_commands`` is a list of shell commands run after the
    directives but before the user command. Useful for ``module load``,
    activating a virtualenv, sourcing env files, etc.

    Set ``place="excl"`` for whole-node allocation (parallel to
    Slurm's ``--exclusive``). ``place="shared"`` packs onto a partially-
    used node (parallel to ``--oversubscribe``). Important for
    memory-hungry POLARIS workloads: without exclusive placement,
    multiple polarisopt samples can co-locate on one node and the
    kernel OOM-kills them.
    """

    queue: str | None = None
    account: str | None = None
    walltime: str | None = None
    select: int = 1
    ncpus: int | None = None
    mpiprocs: int | None = None
    mem: str | None = None  # lowercase, e.g. "96gb"
    place: str | None = None  # "excl" | "shared" | "free"
    join_output: bool = True
    extra_directives: list[str] = field(default_factory=list)
    setup_commands: list[str] = field(default_factory=list)

    def to_directives(
        self, *, job_name: str, stdout: Path | None, stderr: Path | None
    ) -> list[str]:
        # PBS Pro restricts job names to alphanumeric / dash / underscore;
        # spaces or shell metacharacters cause qsub to reject the script.
        lines: list[str] = [f"#PBS -N {_safe_name(job_name)}"]
        if self.queue:
            lines.append(f"#PBS -q {self.queue}")
        if self.account:
            lines.append(f"#PBS -A {self.account}")
        if self.walltime:
            lines.append(f"#PBS -l walltime={self.walltime}")
        select_parts = [f"select={self.select}"]
        if self.ncpus is not None:
            select_parts.append(f"ncpus={self.ncpus}")
        if self.mpiprocs is not None:
            select_parts.append(f"mpiprocs={self.mpiprocs}")
        if self.mem:
            select_parts.append(f"mem={self.mem}")
        lines.append(f"#PBS -l {':'.join(select_parts)}")
        if self.place:
            lines.append(f"#PBS -l place={self.place}")
        if self.join_output:
            lines.append("#PBS -j oe")
            if stdout:
                lines.append(f"#PBS -o {stdout}")
        else:
            if stdout:
                lines.append(f"#PBS -o {stdout}")
            if stderr:
                lines.append(f"#PBS -e {stderr}")
        lines.extend(self.extra_directives)
        return lines


@dataclass
class PBSJob(Job):
    """Job with the path of the generated qsub script (handy for debugging)."""

    script_path: Path | None = None


ShellRunner = Callable[[list[str]], "subprocess.CompletedProcess[str]"]


def _default_shell_runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


_JOB_STATE_RE = re.compile(r"job_state\s*=\s*(\w)")
_EXIT_STATUS_RE = re.compile(r"exit_status\s*=\s*(-?\d+)")


@runner_registry.register("pbs")
class PBSRunner(Runner):
    """Submit jobs via ``qsub``; track them via ``qstat -fx`` / ``qdel``.

    Each :meth:`submit` writes a generated qsub script next to the job's
    ``cwd`` and submits it. Per-job resource overrides flow in via
    ``JobSpec.extra['resources']``; otherwise the ``default_resources``
    supplied at construction time apply.

    Parameters
    ----------
    default_resources : PBSResources or None
        Cluster resources applied to every submission unless overridden
        per JobSpec. ``None`` → empty defaults (you must override
        per-job).
    shell_runner : callable, optional
        Function that takes ``list[str]`` argv and returns a
        :class:`subprocess.CompletedProcess[str]`. Defaults to
        :func:`subprocess.run`. Tests inject a fake here so the suite
        runs without a real PBS cluster.

    Raises
    ------
    RunnerError
        If ``qsub`` exits nonzero or its stdout cannot be parsed for a
        jobid.

    Examples
    --------
    >>> from polarisopt.runners.pbs import PBSRunner, PBSResources
    >>> runner = PBSRunner(default_resources=PBSResources(             # doctest: +SKIP
    ...     queue="compute",
    ...     account="POLARIS",
    ...     walltime="04:00:00",
    ...     ncpus=16,
    ...     mem="96gb",
    ...     place="excl",
    ... ))
    """

    # Backoff schedule (seconds) for transient qsub failures — per-user
    # queue limits, "PBS server busy", brief controller hiccups. After
    # the schedule is exhausted, the submit raises RunnerError as before.
    # Override per-instance for tests or aggressive clusters.
    SUBMIT_RETRY_BACKOFF_S: tuple[int, ...] = (10, 30, 60, 120, 240)

    def __init__(
        self,
        *,
        default_resources: PBSResources | None = None,
        shell_runner: ShellRunner | None = None,
        submit_retry_backoff_s: tuple[int, ...] | list[int] | None = None,
    ) -> None:
        self._default = default_resources or PBSResources()
        self._shell = shell_runner or _default_shell_runner
        self._submit_backoff: tuple[int, ...] = (
            tuple(submit_retry_backoff_s)
            if submit_retry_backoff_s is not None
            else self.SUBMIT_RETRY_BACKOFF_S
        )

    def submit(self, spec: JobSpec) -> PBSJob:
        import time
        spec.cwd.mkdir(parents=True, exist_ok=True)
        resources = spec.extra.get("resources", self._default)
        if not isinstance(resources, PBSResources):
            raise TypeError("JobSpec.extra['resources'] must be a PBSResources")
        script_path = spec.cwd / f"{_safe_name(spec.name)}.pbs"
        script_path.write_text(self._render_script(spec, resources))

        attempt = 0
        while True:
            result = self._shell(["qsub", str(script_path)])
            if result.returncode == 0:
                break
            stderr_text = (result.stderr or "").strip()
            if (
                attempt < len(self._submit_backoff)
                and _is_transient_qsub_error(result.returncode, stderr_text)
            ):
                wait = self._submit_backoff[attempt]
                log.warning(
                    "PBSRunner.submit: transient qsub error (rc=%d, attempt %d/%d) "
                    "- retrying in %ds: %s",
                    result.returncode, attempt + 1, len(self._submit_backoff),
                    wait, stderr_text,
                )
                time.sleep(wait)
                attempt += 1
                continue
            raise RunnerError(
                f"qsub failed (rc={result.returncode}): "
                f"{stderr_text or (result.stdout or '').strip()}"
            )
        # qsub prints just the jobid (``<number>.<server>``) on stdout.
        # Strip and take the last non-empty line in case a wrapper added
        # banner text. Don't strip the ``.<server>`` suffix — qstat/qdel
        # want the full id.
        jobid = ""
        for line in reversed(result.stdout.splitlines()):
            if line.strip():
                jobid = line.strip()
                break
        if not jobid:
            raise RunnerError(f"could not parse qsub output: {result.stdout!r}")
        log.info("PBSRunner submitted %s as jobid %s", spec.name, jobid)
        return PBSJob(
            spec=spec, task_id=jobid, status=JobStatus.QUEUED, script_path=script_path,
        )

    def status(self, job: Job) -> Job:
        """Poll ``qstat -fx <jobid>``.

        ``-f`` requests full output; ``-x`` includes terminated jobs in
        the lookup (no separate ``sacct``-equivalent needed for PBS Pro).
        Job-not-found maps to ``UNKNOWN`` — same orphan-handling story
        as the Slurm runner.
        """
        result = self._shell(["qstat", "-fx", job.task_id])
        if result.returncode != 0:
            err = (result.stderr or "").strip()
            # PBS prints "qstat: Unknown Job Id <id>" when the job has
            # aged out of the history. Don't propagate the stderr in
            # that case — it's the orphan path.
            if "Unknown Job" in err or "Unknown Job" in (result.stdout or ""):
                job.status = JobStatus.UNKNOWN
                return job
            job.status = JobStatus.UNKNOWN
            job.message = err or "qstat failed"
            return job
        state_match = _JOB_STATE_RE.search(result.stdout)
        if not state_match:
            job.status = JobStatus.UNKNOWN
            return job
        state = state_match.group(1).upper()
        if state in {"C", "F"}:
            # Job terminated; promote to FAILED if exit_status != 0.
            exit_match = _EXIT_STATUS_RE.search(result.stdout)
            if exit_match:
                exit_code = int(exit_match.group(1))
                job.exit_code = exit_code
                if exit_code != 0:
                    job.status = JobStatus.FAILED
                    return job
            job.status = JobStatus.FINISHED
            return job
        job.status = _qstat_state_to_status(state)
        return job

    def cancel(self, job: Job) -> Job:
        result = self._shell(["qdel", job.task_id])
        if result.returncode != 0:
            err = (result.stderr or "").strip().lower()
            # qdel on an already-terminated job returns nonzero with
            # "Job has finished" or "Unknown Job Id" — treat both as
            # successfully cancelled from polarisopt's POV.
            if "finished" in err or "unknown job" in err:
                job.status = JobStatus.CANCELLED
                return job
            job.message = f"qdel failed: {result.stderr.strip()}"
            return job
        job.status = JobStatus.CANCELLED
        return job

    def _render_script(self, spec: JobSpec, resources: PBSResources) -> str:
        directives = resources.to_directives(
            job_name=spec.name, stdout=spec.stdout, stderr=spec.stderr,
        )
        env_lines = [f"export {k}={shlex.quote(v)}" for k, v in spec.env.items()]
        # PBS directives must precede the first executable line (same
        # as Slurm — qsub stops parsing #PBS at the first non-comment).
        # Keep ``set -euo pipefail`` after the directive block.
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


_TRANSIENT_QSUB_PATTERNS = (
    "would exceed complex's per-user limit",  # PBS Pro user-job limit
    "would exceed queue's per-user limit",
    "would exceed server's per-user limit",
    "resource temporarily unavailable",
    "server is busy",
    "cannot connect to pbs server",
    "request rejected as server shutting down",
    "system error",  # transient server-side; safe to retry
)


def _is_transient_qsub_error(rc: int, stderr: str) -> bool:
    """Decide whether a non-zero qsub should be retried with backoff.

    Pattern-matching on the stderr text. PBS doesn't have a stable
    machine-readable error code (rc 38 is the most common "limit"
    case but the text varies by version), so we match strings that
    polarisopt has observed correlating with transient cluster
    state vs. permanent submission errors (bad queue, no account, etc.).
    """
    text = stderr.lower()
    return any(pat in text for pat in _TRANSIENT_QSUB_PATTERNS)


def _qstat_state_to_status(state: str) -> JobStatus:
    """Map a single-letter PBS ``job_state`` to polarisopt's enum.

    ``C`` and ``F`` are handled in the caller because their interpretation
    depends on ``exit_status`` (FINISHED vs FAILED). Here we only handle
    the unambiguous in-flight states.
    """
    if state == "Q":
        return JobStatus.QUEUED
    if state == "H":
        return JobStatus.QUEUED  # held — semantically queued
    if state in {"R", "E"}:
        return JobStatus.RUNNING
    # S (suspended) is rare and ambiguous; treat as UNKNOWN.
    return JobStatus.UNKNOWN
