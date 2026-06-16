"""Stopping criteria for sequential studies."""

from polarisopt.stop.base import (
    StoppingCriterion,
    StoppingState,
    make_stop,
    stop_registry,
)
from polarisopt.stop.combinators import AllStop, AnyStop
from polarisopt.stop.epsilon import EpsilonStop
from polarisopt.stop.hypervolume import HypervolumeStop
from polarisopt.stop.max_iter import MaxIterStop
from polarisopt.stop.plateau import PlateauStop

__all__ = [
    "AllStop",
    "AnyStop",
    "EpsilonStop",
    "HypervolumeStop",
    "MaxIterStop",
    "PlateauStop",
    "StoppingCriterion",
    "StoppingState",
    "make_stop",
    "stop_registry",
]
