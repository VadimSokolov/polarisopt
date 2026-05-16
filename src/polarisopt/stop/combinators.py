"""Compose stopping criteria with logical any/all."""

from __future__ import annotations

from collections.abc import Sequence

from polarisopt.stop.base import StoppingCriterion, StoppingState, stop_registry


@stop_registry.register("any")
class AnyStop(StoppingCriterion):
    """Stop if any child criterion fires (logical OR)."""

    def __init__(self, criteria: Sequence[StoppingCriterion]) -> None:
        if not criteria:
            raise ValueError("AnyStop needs at least one child criterion")
        self.criteria = list(criteria)

    def should_stop(self, state: StoppingState) -> bool:
        return any(c.should_stop(state) for c in self.criteria)


@stop_registry.register("all")
class AllStop(StoppingCriterion):
    """Stop only when *all* child criteria fire (logical AND)."""

    def __init__(self, criteria: Sequence[StoppingCriterion]) -> None:
        if not criteria:
            raise ValueError("AllStop needs at least one child criterion")
        self.criteria = list(criteria)

    def should_stop(self, state: StoppingState) -> bool:
        return all(c.should_stop(state) for c in self.criteria)
