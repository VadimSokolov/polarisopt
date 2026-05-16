"""StoppingCriterion ABC and shared state object."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from polarisopt.utils.registry import Registry


@dataclass
class StoppingState:
    """The state a stopping criterion inspects.

    Built fresh at the top of each sequential iteration so criteria can be
    stateless (the orchestrator owns the history).
    """

    iteration: int
    X: np.ndarray  # (n, d) finished sample inputs
    Y: np.ndarray  # (n, m) metric values
    history: list[StoppingState] = field(default_factory=list)
    start_time_s: float = field(default_factory=time.time)
    minimize: bool = True


class StoppingCriterion(ABC):
    """``should_stop`` is called once per iteration with fresh state."""

    @abstractmethod
    def should_stop(self, state: StoppingState) -> bool: ...


stop_registry: Registry[StoppingCriterion] = Registry("stop")


def make_stop(spec: dict[str, Any]) -> StoppingCriterion:
    """Build a stopping criterion (or combinator) from a YAML-style spec.

    Recursive — combinators ``any``/``all`` expect a ``criteria`` list of
    nested specs, each of which is itself parsed via :func:`make_stop`.
    """
    if "type" not in spec:
        raise ValueError(f"stop spec missing 'type': {spec!r}")
    cls = stop_registry.get(spec["type"])
    options = spec.get("options", {}) or {}
    criteria_specs = spec.get("criteria") or None  # treat empty list / explicit None alike
    if criteria_specs:
        children = [make_stop(c) for c in criteria_specs]
        return cls(criteria=children, **options)
    return cls(**options)
