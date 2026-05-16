"""Simulator ABC — bridges Sample → JobSpec → output dict.

A :class:`Simulator` knows how to stage a sample (writing input files into
the per-sample workspace), construct the shell command that runs it, and
read its outputs back. The Study orchestrator hands the JobSpec to a
:class:`~polarisopt.runners.base.Runner` and never executes simulator code
itself — this is the master/slave boundary.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from polarisopt.parameters import ParameterSpace
from polarisopt.runners.base import JobSpec
from polarisopt.samples.sample import Sample
from polarisopt.utils.registry import Registry


class SimulatorError(RuntimeError):
    """Raised when the simulator cannot stage or collect output for a sample."""


class Simulator(ABC):
    """Per-sample staging and output collection.

    Implementations must be safe to call from multiple threads — the master
    may stage a batch of samples concurrently.
    """

    @abstractmethod
    def prepare(self, sample: Sample, space: ParameterSpace, workspace: Path) -> JobSpec:
        """Stage files for ``sample`` and return the JobSpec to execute it.

        ``workspace`` is the per-sample directory; the simulator owns it.
        The returned JobSpec's ``cwd`` typically equals ``workspace``.
        """

    @abstractmethod
    def collect_output(self, sample: Sample) -> dict[str, Any]:
        """Read sample outputs from disk into a JSON-safe dict.

        Called only after the runner reports the job finished. The dict is
        consumed by :class:`~polarisopt.metrics.base.Metric.compute`.
        """


simulator_registry: Registry[Simulator] = Registry("simulator")


def make_simulator(spec: dict[str, Any]) -> Simulator:
    """Build a Simulator from a YAML-style spec ``{"type": "...", "options": {...}}``."""
    if "type" not in spec:
        raise ValueError(f"simulator spec missing 'type': {spec!r}")
    cls = simulator_registry.get(spec["type"])
    options = spec.get("options", {}) or {}
    return cls(**options)
