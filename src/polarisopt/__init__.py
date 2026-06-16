"""polarisopt — modular design-of-experiments and Bayesian optimization for POLARIS."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("polarisopt")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
