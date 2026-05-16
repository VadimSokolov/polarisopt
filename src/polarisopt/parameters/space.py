"""ParameterSpace — the search space over POLARIS calibration variables.

A ``Parameter`` is a single tunable knob. It has a name, bounds, a type
(``float`` or ``int``), and the relative path of the POLARIS JSON file
that should receive the value. A ``ParameterSpace`` is a collection.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np


class ParameterType(StrEnum):
    FLOAT = "float"
    INT = "int"


@dataclass(frozen=True)
class Parameter:
    """A single calibration parameter.

    Attributes:
        name: variable name as it appears in the POLARIS JSON.
        file: relative path of the POLARIS JSON that owns this variable.
        low: lower bound (inclusive).
        high: upper bound (inclusive).
        ptype: ``float`` or ``int``.
    """

    name: str
    file: str
    low: float
    high: float
    ptype: ParameterType = ParameterType.FLOAT

    def __post_init__(self) -> None:
        if self.high <= self.low:
            raise ValueError(f"Parameter '{self.name}': high ({self.high}) must exceed low ({self.low}).")

    def clip(self, value: float) -> float | int:
        """Clip ``value`` into [low, high] and coerce to the declared type."""
        v = float(np.clip(value, self.low, self.high))
        if self.ptype is ParameterType.INT:
            return int(round(v))
        return v


@dataclass(frozen=True)
class ParameterSpace:
    """Ordered collection of parameters defining the search space."""

    parameters: tuple[Parameter, ...]

    @classmethod
    def from_iterable(cls, items: list[Parameter] | tuple[Parameter, ...]) -> ParameterSpace:
        params = tuple(items)
        seen: set[str] = set()
        for p in params:
            if p.name in seen:
                raise ValueError(f"Duplicate parameter name: '{p.name}'")
            seen.add(p.name)
        return cls(parameters=params)

    @property
    def ndim(self) -> int:
        return len(self.parameters)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.parameters)

    @property
    def bounds(self) -> np.ndarray:
        """``(ndim, 2)`` array of [low, high] per parameter."""
        return np.array([[p.low, p.high] for p in self.parameters], dtype=float)

    def clip(self, values: np.ndarray) -> np.ndarray:
        """Project a sample vector (or batch) onto the space, respecting int types.

        Accepts shape ``(ndim,)`` or ``(n, ndim)`` and returns the same shape.
        """
        if values.ndim == 1:
            if values.shape[0] != self.ndim:
                raise ValueError(f"Expected vector of length {self.ndim}, got {values.shape}")
            return np.array([p.clip(v) for p, v in zip(self.parameters, values, strict=True)])
        if values.ndim == 2:
            if values.shape[1] != self.ndim:
                raise ValueError(f"Expected (n, {self.ndim}) array, got {values.shape}")
            return np.array([[p.clip(v) for p, v in zip(self.parameters, row, strict=True)] for row in values])
        raise ValueError(f"values must be 1- or 2-dimensional, got {values.ndim}")

    def values_dict(self, values: np.ndarray) -> dict[str, float | int]:
        """Map a single sample vector to ``{parameter_name: clipped_value}``."""
        if values.ndim != 1 or values.shape[0] != self.ndim:
            raise ValueError(f"values must have shape ({self.ndim},), got {values.shape}")
        clipped = self.clip(values)
        return {p.name: clipped[i] for i, p in enumerate(self.parameters)}

    def by_file(self) -> dict[str, list[Parameter]]:
        """Group parameters by their target POLARIS JSON file."""
        out: dict[str, list[Parameter]] = {}
        for p in self.parameters:
            out.setdefault(p.file, []).append(p)
        return out


def _coerce_ptype(raw: Any) -> ParameterType:
    if isinstance(raw, ParameterType):
        return raw
    if raw is None:
        return ParameterType.FLOAT
    s = str(raw).strip().lower()
    if s in {"float", "real", "continuous"}:
        return ParameterType.FLOAT
    if s in {"int", "integer", "discrete"}:
        return ParameterType.INT
    raise ValueError(f"Unknown parameter type: {raw!r}")


def parameter_space_from_records(records: list[dict[str, Any]]) -> ParameterSpace:
    """Build a ``ParameterSpace`` from a list of dicts.

    Each dict requires ``name``, ``file``, ``min``, ``max``; ``type`` defaults to float.
    """
    params: list[Parameter] = []
    for r in records:
        try:
            params.append(
                Parameter(
                    name=str(r["name"]),
                    file=str(r["file"]),
                    low=float(r["min"]),
                    high=float(r["max"]),
                    ptype=_coerce_ptype(r.get("type")),
                )
            )
        except KeyError as exc:
            raise ValueError(f"Parameter record missing required key: {exc}; record={r!r}") from exc
    return ParameterSpace.from_iterable(params)
