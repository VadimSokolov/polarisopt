"""LocalTransfer — local filesystem copies via shutil / rsync-equivalent."""

from __future__ import annotations

import errno
import shutil
from pathlib import Path

from polarisopt.transfer.base import (
    QuotaExceededError,
    Transfer,
    TransferError,
    transfer_registry,
)
from polarisopt.utils.logging import get_logger

log = get_logger(__name__)

# Errno values that mean "the destination filesystem can't accept more
# bytes." EDQUOT is the user-quota case (most common on shared GPFS /
# Lustre); ENOSPC is the filesystem-full case.
_QUOTA_ERRNOS = {errno.EDQUOT, errno.ENOSPC}


@transfer_registry.register("local")
class LocalTransfer(Transfer):
    """Copy files / directories on the local filesystem.

    Implementation note: uses :func:`shutil.copy2` for files and
    :func:`shutil.copytree` (with ``dirs_exist_ok=True``) for trees.
    Quota / disk-full failures surface as :class:`QuotaExceededError`
    so the master loop can report them cleanly instead of a 50-line
    traceback.
    """

    def copy(self, src: Path | str, dst: Path | str, *, recursive: bool = False) -> None:
        src_p, dst_p = Path(src), Path(dst)
        if not src_p.exists():
            raise TransferError(f"source does not exist: {src_p}")
        dst_p.parent.mkdir(parents=True, exist_ok=True)
        try:
            if src_p.is_dir():
                if not recursive:
                    raise TransferError(
                        f"source {src_p} is a directory; pass recursive=True"
                    )
                shutil.copytree(src_p, dst_p, dirs_exist_ok=True)
            else:
                shutil.copy2(src_p, dst_p)
        except OSError as exc:
            errno_value = _find_quota_errno(exc)
            if errno_value is not None:
                raise QuotaExceededError(
                    f"copy {src_p} -> {dst_p} hit "
                    f"{errno.errorcode.get(errno_value, errno_value)}: {exc}"
                ) from exc
            raise TransferError(f"copy {src_p} -> {dst_p} failed: {exc}") from exc
        log.debug("LocalTransfer copied %s -> %s", src_p, dst_p)


def _find_quota_errno(exc: OSError) -> int | None:
    """Return the quota-class errno if ``exc`` (or any of its raised-from
    parents inside shutil's batched error report) carries one.

    ``shutil.copytree`` collects per-file errors into a ``shutil.Error``
    whose ``args[0]`` is a list of (src, dst, why) tuples — the original
    OSError is buried in there. We walk both the immediate ``errno`` and
    any nested error strings for the canonical quota signals.
    """
    if exc.errno in _QUOTA_ERRNOS:
        return exc.errno
    # shutil.Error wraps per-file failures. Check each tuple's reason text.
    if isinstance(exc, shutil.Error) and exc.args and isinstance(exc.args[0], list):
        for _src, _dst, why in exc.args[0]:
            text = str(why).lower()
            if "errno 122" in text or "edquot" in text or "quota" in text:
                return errno.EDQUOT
            if "errno 28" in text or "enospc" in text or "no space" in text:
                return errno.ENOSPC
    return None
