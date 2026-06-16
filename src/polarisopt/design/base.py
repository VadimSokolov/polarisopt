"""Design ABC — one-shot sample generators (LHS, Morris, Sobol, manual)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from polarisopt.parameters import ParameterSpace
from polarisopt.utils.registry import Registry


class Design(ABC):
    """Generate a static set of sample points over a ``ParameterSpace``.

    Subclasses are constructed with their options (n, levels, ...) and then
    :meth:`generate` is called once with the parameter space and an RNG. The
    output is a ``(n_points, ndim)`` array of inputs already clipped to the
    space bounds.
    """

    @abstractmethod
    def generate(self, space: ParameterSpace, *, rng: np.random.Generator) -> np.ndarray:
        """Return the sample matrix for this design over ``space``."""


design_registry: Registry[Design] = Registry("design")


def make_design(spec: dict[str, Any]) -> Design:
    """Build a Design from a YAML-style spec ``{"type": "...", "options": {...}}``."""
    if "type" not in spec:
        raise ValueError(f"design spec missing 'type': {spec!r}")
    cls = design_registry.get(spec["type"])
    options = spec.get("options", {}) or {}
    return cls(**options)
