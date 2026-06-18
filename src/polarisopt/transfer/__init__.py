"""File transfer backends — local, ANL Globus, ..."""

from polarisopt.transfer.base import (
    QuotaExceededError,
    Transfer,
    TransferError,
    make_transfer,
    transfer_registry,
)
from polarisopt.transfer.local import LocalTransfer

__all__ = [
    "LocalTransfer",
    "QuotaExceededError",
    "Transfer",
    "TransferError",
    "make_transfer",
    "transfer_registry",
]


def _autoload_optional() -> None:
    """Register the optional ANL Globus-aware and Globus-SDK-direct backends.

    Both modules import cleanly without their underlying deps; the
    ImportError fires only when ``.copy()`` is actually called. We
    always register so the registry surface is discoverable.
    """
    from polarisopt.transfer import anl, globus  # noqa: F401


_autoload_optional()
