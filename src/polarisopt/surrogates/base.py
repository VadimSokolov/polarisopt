"""Surrogate ABC — fit / predict on the (X, Y) history kept in the SampleStore.

The interface is vector-valued: ``Y`` is always shape ``(n, m)`` with
``m == n_objectives``. Single-objective callers use ``m=1``. Concrete
surrogates may delegate per-output to a list of single-output models
(e.g. BoTorch's ``ModelListGP``) — that's a private implementation choice.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from polarisopt.utils.registry import Registry


class SurrogateError(RuntimeError):
    """Surrogate-level failure (insufficient data, NaN inputs, fit divergence)."""


class Surrogate(ABC):
    """Probabilistic surrogate over the parameter space."""

    @property
    @abstractmethod
    def n_objectives(self) -> int:
        """Length of ``m`` after :meth:`fit`. Raises before fit."""

    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Train on observations.

        Parameters
        ----------
        X : (n, d) array of inputs
        Y : (n, m) array of objective values (m=1 for single-objective)
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(mean, variance)`` each shaped ``(n, m)``."""

    def is_fitted(self) -> bool:  # pragma: no cover - default; subclasses override if needed
        return False


surrogate_registry: Registry[Surrogate] = Registry("surrogate")


def make_surrogate(spec: dict[str, Any]) -> Surrogate:
    """Build a Surrogate from ``{"type": "...", "options": {...}}``."""
    if "type" not in spec:
        raise ValueError(f"surrogate spec missing 'type': {spec!r}")
    cls = surrogate_registry.get(spec["type"])
    options = spec.get("options", {}) or {}
    return cls(**options)
