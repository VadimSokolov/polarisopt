"""Stop when the best metric hasn't improved by more than ``tol`` in ``window`` iters."""

from __future__ import annotations

import numpy as np

from polarisopt.stop.base import StoppingCriterion, StoppingState, stop_registry


@stop_registry.register("plateau")
class PlateauStop(StoppingCriterion):
    """Single-objective plateau detection.

    Looks at the trailing ``window`` iterations in ``state.history`` and stops
    when the spread of best-so-far across the window is below ``tol``.
    """

    def __init__(self, tol: float, window: int = 5, objective_index: int = 0) -> None:
        if tol <= 0:
            raise ValueError(f"tol must be > 0, got {tol}")
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        self.tol = float(tol)
        self.window = int(window)
        self.objective_index = int(objective_index)

    def should_stop(self, state: StoppingState) -> bool:
        history = state.history
        if len(history) < self.window:
            return False
        bests = []
        for h in history[-self.window :]:
            if h.Y.size == 0:
                return False
            col = h.Y[:, self.objective_index]
            bests.append(float(np.min(col) if h.minimize else np.max(col)))
        return (max(bests) - min(bests)) < self.tol
