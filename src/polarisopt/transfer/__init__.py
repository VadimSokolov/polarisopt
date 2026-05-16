"""File transfer backends — local, ANL Globus, ..."""

from polarisopt.transfer.base import (
    Transfer,
    TransferError,
    make_transfer,
    transfer_registry,
)
from polarisopt.transfer.local import LocalTransfer

__all__ = [
    "LocalTransfer",
    "Transfer",
    "TransferError",
    "make_transfer",
    "transfer_registry",
]


def _autoload_optional() -> None:
    """Register the ANL/Globus backend if polarislib is installed."""
    import contextlib

    with contextlib.suppress(ImportError):
        from polarisopt.transfer import anl  # noqa: F401


_autoload_optional()
