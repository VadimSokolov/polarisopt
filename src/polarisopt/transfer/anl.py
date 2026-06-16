"""ANL Globus-aware transfer via polarislib's ``magic_copy``.

Auto-routes through Globus when paths sit on a registered endpoint (VMS,
LCRC mounts) and falls back to local copy otherwise. Requires the
``polaris-studio`` package (install with ``polarisopt[anl]``).
"""

from __future__ import annotations

from pathlib import Path

from polarisopt.transfer.base import Transfer, TransferError, transfer_registry
from polarisopt.utils.logging import get_logger


def _load_magic_copy():
    """Import ``polaris.utils.copy_utils.magic_copy`` lazily.

    Kept out of module import so ``polarisopt.transfer`` can register the
    ``anl`` backend on every install — users only hit the ImportError
    when they actually try to use it.
    """
    try:
        from polaris.utils.copy_utils import magic_copy
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "polarisopt.transfer.anl requires the [anl] extra: "
            "pip install 'polarisopt[anl]'"
        ) from exc
    return magic_copy

log = get_logger(__name__)


@transfer_registry.register("anl")
class AnlTransfer(Transfer):
    """Globus-aware transfer via polarislib's ``magic_copy``.

    ``magic_copy`` inspects src / dst and chooses Globus or local cp depending
    on whether either side is a registered endpoint path (e.g. ``/mnt/VMS_*``).
    """

    def copy(self, src: Path | str, dst: Path | str, *, recursive: bool = False) -> None:
        magic_copy = _load_magic_copy()
        src_p, dst_p = Path(src), Path(dst)
        try:
            magic_copy(src_p, dst_p, recursive=recursive)
        except Exception as exc:
            raise TransferError(f"magic_copy {src_p} -> {dst_p} failed: {exc}") from exc
        log.debug("AnlTransfer copied %s -> %s (recursive=%s)", src_p, dst_p, recursive)
