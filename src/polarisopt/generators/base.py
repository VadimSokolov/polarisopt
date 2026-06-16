"""SampleGenerator ABC — choose next batch of inputs in a sequential study.

Generators are *batch-first*: a single-point generator just returns a batch
of size 1. The orchestrator passes them ``GeneratorContext`` containing the
observed history; they return ``(q, ndim)`` of next inputs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from polarisopt.parameters import ParameterSpace
from polarisopt.utils.registry import Registry


@dataclass
class GeneratorContext:
    """All the information a generator needs to pick the next batch."""

    space: ParameterSpace
    X: np.ndarray  # (n, d) — finished sample inputs
    Y: np.ndarray  # (n, m) — corresponding metric values
    iteration: int
    rng: np.random.Generator


class SampleGeneratorError(RuntimeError):
    """Generator-level failure (e.g. acquisition can't optimize, surrogate won't fit)."""


class SampleGenerator(ABC):
    """Pick the next batch of input points."""

    @abstractmethod
    def next(self, ctx: GeneratorContext, *, q: int) -> np.ndarray:
        """Return ``(q, ndim)`` of next-sample inputs, clipped to ``ctx.space``."""


generator_registry: Registry[SampleGenerator] = Registry("generator")


def make_generator(spec: dict[str, Any]) -> SampleGenerator:
    """Build a SampleGenerator from ``{"type": "...", "options": {...}}``."""
    if "type" not in spec:
        raise ValueError(f"generator spec missing 'type': {spec!r}")
    cls = generator_registry.get(spec["type"])
    options = spec.get("options", {}) or {}
    return cls(**options)
