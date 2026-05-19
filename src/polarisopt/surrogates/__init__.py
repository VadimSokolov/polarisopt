"""Surrogate models — predict simulator response from a small training set."""

from polarisopt.surrogates.base import (
    Surrogate,
    SurrogateError,
    make_surrogate,
    surrogate_registry,
)

__all__ = ["Surrogate", "SurrogateError", "make_surrogate", "surrogate_registry"]


def _autoload_optional() -> None:
    """Best-effort import of optional surrogates so their @register runs."""
    import contextlib

    with contextlib.suppress(ImportError):  # BoTorch GP
        from polarisopt.surrogates import gp  # noqa: F401
    with contextlib.suppress(ImportError):  # BoTorch multi-task GP
        from polarisopt.surrogates import mtgp  # noqa: F401


_autoload_optional()
