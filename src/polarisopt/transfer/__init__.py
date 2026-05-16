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
    """Register the ANL Globus-aware backend.

    The module imports cleanly without polaris-studio; the ImportError
    only fires when :meth:`AnlTransfer.copy` is actually called. So we
    always register the backend so ``transfer_registry.get("anl")``
    works for discoverability.
    """
    from polarisopt.transfer import anl  # noqa: F401


_autoload_optional()
