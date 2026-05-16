"""Local subprocess runner. Useful for tests and small studies on the head node."""

from __future__ import annotations

import os
import subprocess
import threading
import uuid

from polarisopt.runners.base import Job, JobSpec, JobStatus, Runner, runner_registry
from polarisopt.utils.logging import get_logger

log = get_logger(__name__)


@runner_registry.register("local")
class LocalRunner(Runner):
    """Forks the command via ``subprocess.Popen`` and tracks the child PID.

    Each :meth:`submit` returns immediately with a job in RUNNING state. The
    runner stores the ``Popen`` handle internally so that :meth:`status` and
    :meth:`cancel` can act on it.
    """

    def __init__(self) -> None:
        self._procs: dict[str, subprocess.Popen[bytes]] = {}
        self._lock = threading.Lock()

    def submit(self, spec: JobSpec) -> Job:
        spec.cwd.mkdir(parents=True, exist_ok=True)
        task_id = uuid.uuid4().hex
        # stdout/stderr file handles are intentionally owned by the child
        # subprocess. The Popen below inherits them and Python's gc closes
        # them once the child exits. Using a context manager here would close
        # them before the child can write.
        stdout = open(spec.stdout, "wb") if spec.stdout else subprocess.DEVNULL  # noqa: SIM115
        stderr = open(spec.stderr, "wb") if spec.stderr else subprocess.DEVNULL  # noqa: SIM115
        env = {**os.environ, **spec.env}
        proc = subprocess.Popen(  # noqa: S603 - shell=True is intentional for command flexibility
            spec.command,
            shell=True,
            cwd=str(spec.cwd),
            env=env,
            stdout=stdout,
            stderr=stderr,
        )
        with self._lock:
            self._procs[task_id] = proc
        log.info("LocalRunner submitted %s (pid=%d, task_id=%s)", spec.name, proc.pid, task_id)
        return Job(spec=spec, task_id=task_id, status=JobStatus.RUNNING)

    def status(self, job: Job) -> Job:
        with self._lock:
            proc = self._procs.get(job.task_id)
        if proc is None:
            job.status = JobStatus.UNKNOWN
            job.message = "no local handle"
            return job
        rc = proc.poll()
        if rc is None:
            job.status = JobStatus.RUNNING
            return job
        job.exit_code = rc
        job.status = JobStatus.FINISHED if rc == 0 else JobStatus.FAILED
        return job

    def cancel(self, job: Job) -> Job:
        with self._lock:
            proc = self._procs.get(job.task_id)
        if proc is None or proc.poll() is not None:
            return self.status(job)
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        job.status = JobStatus.CANCELLED
        job.message = "cancelled"
        job.exit_code = proc.returncode
        return job
