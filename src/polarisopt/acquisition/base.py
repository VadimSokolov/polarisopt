"""Acquisition ABC — given a Surrogate, propose the next batch.

The acquisition layer owns both *building* the acquisition function and
*optimizing* it. Returning ``ndarray`` of candidate inputs lets callers
stay backend-agnostic.

The orchestrator picks the convention: by default we **minimize** metrics
(POLARIS calibration is minimization). Acquisition implementations honor
``minimize=True``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from polarisopt.parameters import ParameterSpace
from polarisopt.surrogates.base import Surrogate
from polarisopt.utils.registry import Registry


class AcquisitionError(RuntimeError):
    """Acquisition-level failure (e.g. infeasible reference point for qEHVI)."""


class AcquisitionFunction(ABC):
    """Pick the next ``q`` candidate input vectors using a fitted Surrogate.

    Concrete subclasses construct themselves from a Surrogate snapshot and
    any user options, then :meth:`optimize` returns a ``(q, ndim)`` matrix.
    """

    def __init__(self, surrogate: Surrogate, *, minimize: bool = True) -> None:
        self.surrogate = surrogate
        self.minimize = bool(minimize)

    @abstractmethod
    def optimize(
        self,
        space: ParameterSpace,
        *,
        q: int,
        observed_Y: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return the next ``q`` input vectors, shape ``(q, ndim)``.

        ``observed_Y`` is the (n, m) matrix of objective values seen so far,
        used to set acquisition baselines (e.g. ``best_f`` for EI, reference
        point for qEHVI). Acquisition implementations are free to ignore it.
        """


acquisition_registry: Registry[AcquisitionFunction] = Registry("acquisition")


def make_acquisition(spec: dict[str, Any], surrogate: Surrogate, *, minimize: bool = True) -> AcquisitionFunction:
    """Build an acquisition from ``{"type": "...", "options": {...}}``."""
    if "type" not in spec:
        raise ValueError(f"acquisition spec missing 'type': {spec!r}")
    cls = acquisition_registry.get(spec["type"])
    options = spec.get("options", {}) or {}
    return cls(surrogate=surrogate, minimize=minimize, **options)
