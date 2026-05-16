"""Simulator backends — the bridge between Sample and Runner."""

from polarisopt.simulator.base import (
    Simulator,
    SimulatorError,
    make_simulator,
    simulator_registry,
)
from polarisopt.simulator.benchmarks import BENCHMARKS, BenchmarkFn
from polarisopt.simulator.mock import MockSimulator

__all__ = [
    "BENCHMARKS",
    "BenchmarkFn",
    "MockSimulator",
    "Simulator",
    "SimulatorError",
    "make_simulator",
    "simulator_registry",
]
