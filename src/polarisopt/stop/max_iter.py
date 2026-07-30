"""Stop after a fixed number of iterations."""

from __future__ import annotations

from polarisopt.stop.base import StoppingCriterion, StoppingState, stop_registry


@stop_registry.register("max_iter")
class MaxIterStop(StoppingCriterion):
    """Stop once ``state.iteration >= n``.

    ``n = 0`` is legal (v0.26+) and means "no BO iterations after
    warm-up" — the sequential loop's stop check fires immediately at
    ``iteration = 0`` and the generator is never called. Use this for
    validation-run studies (POLARIS at pinned parameters, score, exit)
    where you don't have the ≥2 finished samples the acquisition
    generator needs. Negative values are still rejected as
    almost-certainly-a-bug.
    """

    def __init__(self, n: int) -> None:
        if n < 0:
            raise ValueError(f"MaxIterStop.n must be >= 0, got {n}")
        self.n = int(n)

    def should_stop(self, state: StoppingState) -> bool:
        return state.iteration >= self.n
