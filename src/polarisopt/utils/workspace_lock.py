"""Exclusive flock on a workspace so two masters can't race the same SampleStore.

The failure mode this prevents: user submits the master as a Slurm job
(per docs/how-to/run-on-slurm.md), forgets it's running, fires
``polarisopt resume`` from a login shell. Now two orchestrators are
both polling Slurm, both calling ``collect_output``, both deciding
what's next — duplicate submissions, cancelled-but-still-alive jobs,
state thrash.

``WAL`` mode protects the SQLite layer from concurrent reads vs writes,
but doesn't stop two writers from making conflicting *decisions* in
the application layer. That's what flock guards.

Lock semantics
--------------
- The lock is an exclusive ``flock(2)`` on ``<workspace>/.polarisopt.lock``.
- Acquired non-blocking: if another live master holds it, we fail fast
  with a friendly error pointing at the lock-holder's PID, hostname,
  start time, and polarisopt version.
- Auto-released on process death — flock is kernel-managed, no stale
  state to clean up. The sidecar metadata file is best-effort cleaned
  on graceful exit; a stale metadata file alongside a free lock is
  benign (lock acquisition still succeeds).
- ``force=True`` skips the acquisition and proceeds anyway. Use when
  the operator knowingly accepts the racing-masters consequences (or
  in rare filesystems where flock is unreliable). Loud WARNING logged.

Filesystem support
------------------
Works on GPFS / Lustre / ext4 / xfs / local tmpfs. NFS support is
spotty (Linux NFSv4 supports flock; older NFS implementations don't).
On unsupported filesystems flock returns success without enforcing —
the lock is then a hint, not a guarantee. polarisopt deployments on
LCRC use GPFS, where flock is reliable.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import socket
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from polarisopt.utils._compat import UTC

log = logging.getLogger(__name__)


LOCK_FILENAME = ".polarisopt.lock"
META_FILENAME = ".polarisopt.lock.meta"


class WorkspaceLockError(RuntimeError):
    """Raised when another live master holds the workspace lock.

    The message includes the holder's PID, hostname, start time, and
    polarisopt version so the operator can decide whether to kill the
    other process, wait for it, or pass ``--force``.
    """


def _read_meta(meta_path: Path) -> dict[str, str] | None:
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _format_contention(meta: dict[str, str] | None, lock_path: Path) -> str:
    if meta is None:
        return (
            f"another polarisopt master holds the workspace lock at {lock_path} "
            f"(no metadata sidecar to identify the holder). "
            f"If that master is dead the lock will auto-release; otherwise wait "
            f"for it to finish or pass --force to bypass (not recommended)."
        )
    return (
        f"another polarisopt master holds the workspace lock:\n"
        f"  PID:      {meta.get('pid', '?')}\n"
        f"  host:     {meta.get('hostname', '?')}\n"
        f"  started:  {meta.get('started_at', '?')}\n"
        f"  version:  {meta.get('version', '?')}\n"
        f"  action:   {meta.get('action', '?')}\n"
        f"  lock:     {lock_path}\n"
        f"\n"
        f"If that master is dead the lock will auto-release (flock is kernel-managed).\n"
        f"Otherwise wait for it to finish or pass --force to bypass (not recommended)."
    )


@contextlib.contextmanager
def workspace_lock(
    workspace: Path,
    *,
    action: str,
    force: bool = False,
) -> Iterator[None]:
    """Hold the workspace lock for the duration of the ``with`` block.

    Parameters
    ----------
    workspace :
        Per-study workspace directory (``cfg.workspace``).
    action :
        Short verb stored in the metadata sidecar so contention errors
        can tell the operator what the other master is doing (e.g.
        ``"run"`` vs ``"resume"``).
    force :
        Skip the acquisition entirely. Logs a WARNING and proceeds.
        Use only when you knowingly accept the racing-masters risk.

    Raises
    ------
    WorkspaceLockError
        If another live master holds the lock and ``force`` is False.

    Yields
    ------
    None
        Lock is held until the ``with`` block exits, then released.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    lock_path = workspace / LOCK_FILENAME
    meta_path = workspace / META_FILENAME

    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    acquired = False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError:
            existing = _read_meta(meta_path)
            if not force:
                raise WorkspaceLockError(_format_contention(existing, lock_path)) from None
            log.warning(
                "--force: bypassing workspace lock held by %s",
                existing or "<unknown master>",
            )

        if acquired:
            from polarisopt import __version__

            meta = {
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "started_at": datetime.now(UTC).isoformat(),
                "version": __version__,
                "action": action,
            }
            try:
                meta_path.write_text(json.dumps(meta, indent=2))
            except OSError:
                log.warning("failed to write workspace lock metadata at %s", meta_path)

        try:
            yield
        finally:
            if acquired:
                with contextlib.suppress(OSError):
                    meta_path.unlink()
    finally:
        if acquired:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            os.close(fd)
