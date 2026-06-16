"""Manual design — caller supplies the exact points."""

from __future__ import annotations

import numpy as np

from polarisopt.design.base import Design, design_registry
from polarisopt.parameters import ParameterSpace


@design_registry.register("manual")
class ManualDesign(Design):
    """Pass-through design: returns the points given at construction.

    Useful for replaying a saved design or running a one-off study at
    specific points.

    Parameters
    ----------
    points:
        Either a ``(n, ndim)`` array or a list of lists. Validated against
        the space at :meth:`generate` time.
    """

    def __init__(self, points: list[list[float]] | np.ndarray) -> None:
        arr = np.asarray(points, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"ManualDesign.points must be 2-D, got shape {arr.shape}")
        self._points = arr

    def generate(self, space: ParameterSpace, *, rng: np.random.Generator) -> np.ndarray:
        if self._points.shape[1] != space.ndim:
            raise ValueError(
                f"ManualDesign points have {self._points.shape[1]} cols but space has {space.ndim} parameters"
            )
        return space.clip(self._points)
