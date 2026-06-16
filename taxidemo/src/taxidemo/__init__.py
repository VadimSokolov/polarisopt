"""taxidemo — emukit-playground taxi simulator as a polarisopt demo problem.

A pure-Python, seedable port of the Amazon emukit-playground taxi
simulator (https://github.com/amzn/emukit-playground), packaged as a
polarisopt ``Simulator`` plugin so the full design-of-experiments and
Bayesian-optimization toolchain can be demonstrated without a POLARIS
install.
"""

from taxidemo.simulator import OUTPUT_KEYS, TaxiParams, TaxiSimulation, default_params, run_taxi_simulation

__version__ = "0.1.0"

__all__ = [
    "OUTPUT_KEYS",
    "TaxiParams",
    "TaxiSimulation",
    "default_params",
    "run_taxi_simulation",
    "__version__",
]
