"""Python 3.10 backport shims.

The codebase targets ``>=3.10`` but uses ``datetime.UTC`` and
``enum.StrEnum``, both of which are 3.11+. Importing from this module
lets the same code run on 3.10 without ``# noqa`` clutter at each
import site.
"""

from __future__ import annotations

import sys
from datetime import timezone

# Avoid ``from datetime import UTC`` — that's the 3.11+ form we're
# compensating for. ``timezone.utc`` works everywhere and equals
# ``datetime.UTC`` on 3.11+.
UTC = timezone.utc  # noqa: UP017

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):  # type: ignore[no-redef]  # noqa: UP042
        """3.10 backport of :class:`enum.StrEnum`."""

        def __str__(self) -> str:
            return str.__str__(self)


__all__ = ["StrEnum", "UTC"]
