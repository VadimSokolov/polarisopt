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
        observed_X: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the next ``q`` input vectors, shape ``(q, ndim)``.

        Parameters
        ----------
        space
            Parameter space defining bounds and (for mixed-type spaces)
            categorical/integer structure that the optimizer must respect.
        q
            Batch size — number of candidate input vectors to return.
        observed_Y
            ``(n, m)`` matrix of objective values seen so far. Used to
            set acquisition baselines (e.g. ``best_f`` for EI, reference
            point for qEHVI). Acquisition implementations are free to
            ignore it.
        rng
            Numpy random generator. Acquisition implementations that
            perform stochastic optimization (multi-start, MC sampling)
            should route their entropy through this generator so runs
            are reproducible from the study seed.
        observed_X
            ``(n, ndim)`` matrix of input vectors matching ``observed_Y``
            (v0.18+). Required by noise-aware acquisitions like
            ``qlognei`` (which uses it as ``X_baseline`` for
            posterior-based incumbent inference). Ignored by
            acquisitions that compute their baseline from ``observed_Y``
            alone (e.g. ``qei`` / ``qehvi``). Passed as a keyword with a
            ``None`` default for backwards compatibility with pre-v0.18
            acquisition plugins that don't accept it.

        Returns
        -------
        numpy.ndarray
            ``(q, ndim)`` matrix of proposed input vectors, in the same
            column ordering as ``observed_X`` / ``space``.

        Raises
        ------
        AcquisitionError
            Concrete subclasses may raise this when the caller supplies
            an unusable input — e.g. missing ``observed_X`` for a
            noise-aware acquisition, a multi-objective surrogate for a
            single-objective acquisition, or an infeasible reference
            point for qEHVI.

        Notes
        -----
        The orchestrator's convention is minimization when
        ``self.minimize`` is True. Implementations that wrap a
        maximization-native BoTorch acquisition must sign-flip via
        ``GenericMCObjective`` or equivalent.
        """


acquisition_registry: Registry[AcquisitionFunction] = Registry("acquisition")


def make_acquisition(spec: dict[str, Any], surrogate: Surrogate, *, minimize: bool = True) -> AcquisitionFunction:
    """Build an acquisition from ``{"type": "...", "options": {...}}``."""
    if "type" not in spec:
        raise ValueError(f"acquisition spec missing 'type': {spec!r}")
    cls = acquisition_registry.get(spec["type"])
    options = spec.get("options", {}) or {}
    return cls(surrogate=surrogate, minimize=minimize, **options)
