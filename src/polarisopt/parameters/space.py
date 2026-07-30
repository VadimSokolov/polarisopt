"""ParameterSpace — the search space over POLARIS calibration variables.

A ``Parameter`` is a single tunable knob. It has a name, bounds, a type
(``float`` or ``int``), and the relative path of the POLARIS JSON file
that should receive the value. A ``ParameterSpace`` is a collection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from polarisopt.parameters.prior import Prior, prior_from_dict
from polarisopt.utils._compat import StrEnum


class ParameterType(StrEnum):
    FLOAT = "float"
    INT = "int"


@dataclass(frozen=True)
class Parameter:
    """A single calibration parameter.

    Parameters
    ----------
    name : str
        Variable name as it appears in the POLARIS JSON file.
    file : str
        Path of the POLARIS JSON that owns this variable, **relative to
        the staged model directory**. Subdirectories are supported:
        e.g. ``config/choice_models/DestinationChoice.json``. Use forward
        slashes; the path is resolved against each sample's workspace.
        ``ParameterSpace.by_file`` groups parameters by this field.
    low : float
        Lower bound, inclusive.
    high : float
        Upper bound, inclusive. Must be strictly greater than ``low``.
    ptype : ParameterType
        ``ParameterType.FLOAT`` (default) or ``ParameterType.INT``.
        Integer parameters are rounded by :meth:`clip`.
    prior : Prior, optional
        v0.27+ (P2). Optional prior distribution on this parameter,
        as a :class:`polarisopt.parameters.Prior` instance
        (``GaussianPrior`` / ``LogNormalPrior`` / ``TruncatedNormalPrior``
        / ``UniformPrior`` / ``BetaPrior``). Consumed by:

        - :class:`polarisopt.design.LHSDesign` (v0.27+ P11) as the
          per-parameter anchor value when
          ``include_prior_mean_anchor=True``.
        - :func:`polarisopt.studies.identifiability.run_identifiability_analysis`
          (v0.29+ P5) as the pin value when the parameter is flagged
          as unidentified AND ``hold_at_prior_mean_if_unidentified``
          is True.
        - v0.31+ history-matching phases as a virtual moment
          contributing ``((θ - prior.mean) / prior.std)²`` to the
          max-implausibility.

        A prior whose ``.mean`` falls outside ``[low, high]`` is
        rejected at construction — almost always a config bug that
        would silently pull LHS anchors and BO gradients out of the
        search region.
    hold_at_prior_mean_if_unidentified : bool, optional
        v0.29+ (P5). When True AND a ``prior`` is set, the
        ``polarisopt identifiability`` CLI's ``--auto-drop-to-prior-mean``
        flag pins this parameter at ``prior.mean`` (rewriting the
        study YAML) if the identifiability report classifies it as
        unidentified (max first-order Sobol < threshold). Default
        False — unidentified parameters are noted in the report but
        left in the search space for the operator to decide.

    Raises
    ------
    ValueError
        If ``high <= low``, or if ``prior.mean`` falls outside
        ``[low, high]``.

    Examples
    --------
    >>> p = Parameter("trip_threshold", "DestinationChoice.json", 0.0, 1.0)
    >>> p.clip(0.5)
    0.5
    >>> p.clip(-1.0)
    0.0
    >>> q = Parameter("top_k", "DestinationChoice.json", 1, 10, ParameterType.INT)
    >>> q.clip(3.7)
    4

    With a meta-analytic prior for a β-calibration study:

    >>> from polarisopt.parameters import GaussianPrior
    >>> p = Parameter(                                        # doctest: +SKIP
    ...     "b_IVTT_Auto_Mean", "ModeChoiceModel.json",
    ...     low=-0.30, high=-0.001,
    ...     prior=GaussianPrior(mean=-0.10, std=0.035),
    ...     hold_at_prior_mean_if_unidentified=True,
    ... )
    """

    name: str
    file: str
    low: float
    high: float
    ptype: ParameterType = ParameterType.FLOAT
    prior: Prior | None = None
    hold_at_prior_mean_if_unidentified: bool = False

    def __post_init__(self) -> None:
        if self.high <= self.low:
            raise ValueError(f"Parameter '{self.name}': high ({self.high}) must exceed low ({self.low}).")
        # If a prior was supplied but its declared mean is well outside
        # this parameter's box, flag it — this is almost always a
        # config bug that would otherwise silently pull LHS anchors
        # and BO gradients outside the search region.
        if self.prior is not None:
            m = self.prior.mean
            if not math.isfinite(m):
                raise ValueError(
                    f"Parameter '{self.name}': prior.mean is not finite ({m!r})"
                )
            if m < self.low or m > self.high:
                raise ValueError(
                    f"Parameter '{self.name}': prior.mean {m!r} is outside "
                    f"the parameter box [{self.low}, {self.high}]"
                )

    def clip(self, value: float) -> float | int:
        """Clip ``value`` into ``[low, high]`` and coerce to the declared type.

        Parameters
        ----------
        value : float

        Returns
        -------
        float or int
            Float for FLOAT parameters; rounded int for INT parameters.
        """
        v = float(np.clip(value, self.low, self.high))
        if self.ptype is ParameterType.INT:
            return int(round(v))
        return v


@dataclass(frozen=True)
class ParameterSpace:
    """Ordered collection of :class:`Parameter` defining the search space.

    Use :meth:`from_iterable` for construction so duplicate-name detection
    runs. The space is immutable once constructed.

    Examples
    --------
    >>> space = ParameterSpace.from_iterable([
    ...     Parameter("x1", "a.json", -5.0, 10.0),
    ...     Parameter("x2", "a.json",  0.0, 15.0),
    ... ])
    >>> space.ndim
    2
    >>> space.names
    ('x1', 'x2')
    >>> space.bounds.shape
    (2, 2)
    """

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
            prior_spec = r.get("prior")
            prior_obj = prior_from_dict(prior_spec) if prior_spec is not None else None
            params.append(
                Parameter(
                    name=str(r["name"]),
                    file=str(r["file"]),
                    low=float(r["min"]),
                    high=float(r["max"]),
                    ptype=_coerce_ptype(r.get("type")),
                    prior=prior_obj,
                    hold_at_prior_mean_if_unidentified=bool(
                        r.get("hold_at_prior_mean_if_unidentified", False)
                    ),
                )
            )
        except KeyError as exc:
            raise ValueError(f"Parameter record missing required key: {exc}; record={r!r}") from exc
    return ParameterSpace.from_iterable(params)
