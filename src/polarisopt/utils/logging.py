"""Logging setup. Library code uses ``logging.getLogger(__name__)``; applications call ``configure``."""

from __future__ import annotations

import logging
import sys
from typing import Literal

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

_CONFIGURED = False


def configure(level: LogLevel = "INFO", *, force: bool = False) -> None:
    """Initialize root logging for the polarisopt CLI / scripts.

    Library modules should never call this — they call ``getLogger(__name__)``
    and let the application configure handlers. This helper exists for the
    CLI and for tests.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s :: %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger. Library convention: ``get_logger(__name__)``."""
    return logging.getLogger(name)
