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
        from polarisopt.acquisition import ei, qehvi, qei, qlognei  # noqa: F401

    # tolerance_ball/heaviside need only scipy (no torch), so they load
    # independently of the BoTorch-backed acquisitions above.
    with contextlib.suppress(ImportError):
        from polarisopt.acquisition import tolerance_ball  # noqa: F401


_autoload_optional()
