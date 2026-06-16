"""Simulator backends — the bridge between Sample and Runner."""

from polarisopt.simulator.base import (
    Simulator,
    SimulatorError,
    make_simulator,
    simulator_registry,
)
from polarisopt.simulator.benchmarks import BENCHMARKS, BenchmarkFn
from polarisopt.simulator.mock import MockSimulator
from polarisopt.simulator.polaris import PolarisSimulator
from polarisopt.simulator.polaris_convergence import PolarisConvergenceSimulator

__all__ = [
    "BENCHMARKS",
    "BenchmarkFn",
    "MockSimulator",
    "PolarisConvergenceSimulator",
    "PolarisSimulator",
    "Simulator",
    "SimulatorError",
    "make_simulator",
    "simulator_registry",
]
