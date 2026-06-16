"""SequentialDesignStudy — warm-up + surrogate-driven loop.

Phases:
  1. **Warm-up** (optional Design). If the phase has no FINISHED rows yet,
     run the warm-up design. Otherwise skip.
  2. **Iteration loop**.
     a. Build :class:`StoppingState` from finished history.
     b. Check stop criterion — break if fired.
     c. Build :class:`GeneratorContext`; call ``generator.next(ctx, q=batch_size)``.
     d. Persist new Sample rows, evaluate via the shared ``_evaluate_batch``.
     e. Checkpoint phase state to the store (RNG + iteration counter).
     f. Increment iteration.

Restart: ``load_phase_state(phase_name)`` returns the last checkpointed
iteration + RNG state; we restore both before re-entering the loop.
"""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from polarisopt.design.base import Design
from polarisopt.generators.base import GeneratorContext, SampleGenerator
from polarisopt.samples.sample import Sample, SampleStatus
from polarisopt.stop.base import StoppingCriterion, StoppingState
from polarisopt.studies.base import Study, StudyContext, StudyError
from polarisopt.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class SequentialPhase:
    """Per-phase config + plugin instances (already constructed from YAML)."""

    name: str
    generator: SampleGenerator
    stop: StoppingCriterion
    warm_up: Design | None = None
    batch_size: int = 1
    minimize: bool = True
    history: list[StoppingState] = field(default_factory=list)


class SequentialDesignStudy(Study):
    """Master loop for a sequential phase."""

    def __init__(self, ctx: StudyContext, phase: SequentialPhase) -> None:
        super().__init__(ctx)
        self.phase = phase
        if phase.batch_size < 1:
            raise StudyError(f"batch_size must be >= 1, got {phase.batch_size}")

    def run(self) -> list[Sample]:
        # Restore RNG / iteration from any saved phase state (restart support).
        iteration = self._restore_state()

        finished_count_at_phase_start = self.ctx.store.count(
            phase=self.phase.name, status=SampleStatus.FINISHED
        )

        # Warm-up: if no finished sample exists yet AND a warm-up design is set, generate it.
        if finished_count_at_phase_start == 0 and self.phase.warm_up is not None:
            log.info("Phase %r: running warm-up", self.phase.name)
            points = self.phase.warm_up.generate(self.ctx.space, rng=self.ctx.rng)
            warm_samples = [
                Sample(phase=self.phase.name, iteration=0, inputs=row) for row in points
            ]
            warm_samples = self.ctx.store.add_many(warm_samples)
            self._evaluate_batch(warm_samples)

        # If the warm-up itself was already mid-flight (some PENDING rows), pick those up
        pending = self.ctx.store.list(phase=self.phase.name, status=SampleStatus.PENDING)
        if pending:
            log.info("Phase %r: evaluating %d pending sample(s)", self.phase.name, len(pending))
            self._evaluate_batch(pending)

        all_phase_samples: list[Sample] = self.ctx.store.list(phase=self.phase.name)

        # Iteration loop
        while True:
            finished = [
                s for s in self.ctx.store.list(phase=self.phase.name, status=SampleStatus.FINISHED)
                if s.metric is not None
            ]
            X, Y = _stack_finished(finished)
            state = StoppingState(
                iteration=iteration,
                X=X,
                Y=Y,
                history=list(self.phase.history),
                minimize=self.phase.minimize,
            )
            if self.phase.stop.should_stop(state):
                log.info(
                    "Phase %r: stop criterion fired at iteration %d (n=%d, best=%s)",
                    self.phase.name,
                    iteration,
                    Y.shape[0],
                    _best_str(Y, self.phase.minimize) if Y.size else "n/a",
                )
                break
            self.phase.history.append(state)

            # Generate next batch
            try:
                inputs = self.phase.generator.next(
                    GeneratorContext(
                        space=self.ctx.space,
                        X=X,
                        Y=Y,
                        iteration=iteration,
                        rng=self.ctx.rng,
                    ),
                    q=self.phase.batch_size,
                )
            except Exception as exc:
                log.exception(
                    "Phase %r: generator failed at iteration %d", self.phase.name, iteration
                )
                raise StudyError(f"generator failed: {exc}") from exc

            iteration += 1
            new_samples = [
                Sample(phase=self.phase.name, iteration=iteration, inputs=row) for row in inputs
            ]
            new_samples = self.ctx.store.add_many(new_samples)
            self._evaluate_batch(new_samples)
            all_phase_samples.extend(new_samples)

            # Checkpoint: RNG + iteration. Surrogate state is refit from store on resume.
            self._save_state(iteration)

        return self.ctx.store.list(phase=self.phase.name)

    # ----- restart -----

    def _restore_state(self) -> int:
        record = self.ctx.store.load_phase_state(self.phase.name)
        if record is None:
            return 0
        rng_blob = record.get("rng_state")
        if rng_blob is not None:
            try:
                state = pickle.loads(rng_blob)
                self.ctx.rng.bit_generator.state = state
            except Exception:
                log.warning("Phase %r: failed to restore RNG state; using fresh", self.phase.name)
        return int(record.get("iteration") or 0)

    def _save_state(self, iteration: int) -> None:
        rng_blob = pickle.dumps(self.ctx.rng.bit_generator.state)
        self.ctx.store.save_phase_state(
            self.phase.name, iteration=iteration, rng_state=rng_blob
        )


def _stack_finished(samples: Sequence[Sample]) -> tuple[np.ndarray, np.ndarray]:
    if not samples:
        return np.empty((0, 0)), np.empty((0, 0))
    X = np.stack([s.inputs for s in samples])
    Y = np.stack([s.metric for s in samples])  # type: ignore[arg-type]
    return X, Y


def _best_str(Y: np.ndarray, minimize: bool) -> str:
    if Y.size == 0:
        return "n/a"
    if Y.shape[1] == 1:
        v = float(np.min(Y) if minimize else np.max(Y))
        return f"{v:.6g}"
    return f"pareto(n={Y.shape[0]})"
