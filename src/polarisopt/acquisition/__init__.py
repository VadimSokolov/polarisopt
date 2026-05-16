"""Acquisition functions — pick the next batch of points using a Surrogate."""

from polarisopt.acquisition.base import (
    AcquisitionError,
    AcquisitionFunction,
    acquisition_registry,
    make_acquisition,
)

__all__ = [
    "AcquisitionError",
    "AcquisitionFunction",
    "acquisition_registry",
    "make_acquisition",
]


def _autoload_optional() -> None:
    import contextlib

    with contextlib.suppress(ImportError):
        from polarisopt.acquisition import ei, qehvi, qei  # noqa: F401


_autoload_optional()
