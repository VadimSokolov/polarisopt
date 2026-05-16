"""LocalTransfer — local filesystem copies via shutil / rsync-equivalent."""

from __future__ import annotations

import shutil
from pathlib import Path

from polarisopt.transfer.base import Transfer, TransferError, transfer_registry
from polarisopt.utils.logging import get_logger

log = get_logger(__name__)


@transfer_registry.register("local")
class LocalTransfer(Transfer):
    """Copy files / directories on the local filesystem.

    Implementation note: uses :func:`shutil.copy2` for files and
    :func:`shutil.copytree` (with ``dirs_exist_ok=True``) for trees.
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
            raise TransferError(f"copy {src_p} -> {dst_p} failed: {exc}") from exc
        log.debug("LocalTransfer copied %s -> %s", src_p, dst_p)
