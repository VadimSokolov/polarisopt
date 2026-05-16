"""Stop once the best metric is within ``epsilon`` of a target value."""

from __future__ import annotations

import numpy as np

from polarisopt.stop.base import StoppingCriterion, StoppingState, stop_registry


@stop_registry.register("epsilon")
class EpsilonStop(StoppingCriterion):
    """Single-objective stop: ``|best - target| < epsilon``.

    Default target is 0.0 (typical for calibration RMSE). For maximization,
    pass ``minimize=False`` via the orchestrator and a positive target.
    """

    def __init__(self, epsilon: float, *, target: float = 0.0, objective_index: int = 0) -> None:
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        self.epsilon = float(epsilon)
        self.target = float(target)
        self.objective_index = int(objective_index)

    def should_stop(self, state: StoppingState) -> bool:
        if state.Y.size == 0:
            return False
        col = state.Y[:, self.objective_index]
        best = float(np.min(col) if state.minimize else np.max(col))
        return abs(best - self.target) < self.epsilon
