"""Transfer ABC — copy files between local FS, shared FS, and Globus endpoints.

The orchestrator uses Transfer to move POLARIS model inputs from the user's
storage (VMS / cluster FS / wherever) to a per-sample workspace, and to copy
results back. Implementations choose the right physical mechanism (cp, rsync,
Globus) based on the paths involved.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from polarisopt.utils.registry import Registry


class TransferError(RuntimeError):
    """Raised when a transfer fails."""


class Transfer(ABC):
    """Copy files / directories between locations."""

    @abstractmethod
    def copy(self, src: Path | str, dst: Path | str, *, recursive: bool = False) -> None:
        """Copy ``src`` to ``dst``. If ``recursive`` and src is a directory, copy the tree."""


transfer_registry: Registry[Transfer] = Registry("transfer")


def make_transfer(spec: dict[str, Any] | None) -> Transfer:
    """Build a Transfer from ``{"type": "...", "options": {...}}`` (default: local)."""
    if spec is None:
        from polarisopt.transfer.local import LocalTransfer

        return LocalTransfer()
    if "type" not in spec:
        raise ValueError(f"transfer spec missing 'type': {spec!r}")
    cls = transfer_registry.get(spec["type"])
    options = spec.get("options", {}) or {}
    return cls(**options)
