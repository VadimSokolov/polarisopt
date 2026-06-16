"""Stop after a fixed number of iterations."""

from __future__ import annotations

from polarisopt.stop.base import StoppingCriterion, StoppingState, stop_registry


@stop_registry.register("max_iter")
class MaxIterStop(StoppingCriterion):
    """Stop once ``state.iteration >= n``."""

    def __init__(self, n: int) -> None:
        if n <= 0:
            raise ValueError(f"MaxIterStop.n must be > 0, got {n}")
        self.n = int(n)

    def should_stop(self, state: StoppingState) -> bool:
        return state.iteration >= self.n
