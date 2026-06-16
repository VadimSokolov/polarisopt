"""Static design-of-experiments phase: one Design → many Samples → evaluate."""

from __future__ import annotations

from polarisopt.design.base import Design
from polarisopt.samples.sample import Sample, SampleStatus
from polarisopt.studies.base import Study, StudyContext


class StaticDesignStudy(Study):
    """Run a static design phase end-to-end.

    Parameters
    ----------
    ctx:
        Study context (shared with other phases in a pipeline).
    design:
        The fully-constructed :class:`~polarisopt.design.base.Design`.
    phase_name:
        Name written into ``Sample.phase`` for samples this phase produces.
    """

    def __init__(self, ctx: StudyContext, design: Design, *, phase_name: str = "static") -> None:
        super().__init__(ctx)
        self.design = design
        self.phase_name = phase_name

    def run(self) -> list[Sample]:
        # If the phase already has pending samples (e.g. from a prior interrupted
        # run), evaluate those; otherwise generate fresh ones.
        existing = self.ctx.store.list(phase=self.phase_name, status=SampleStatus.PENDING)
        if existing:
            samples = existing
        else:
            points = self.design.generate(self.ctx.space, rng=self.ctx.rng)
            samples = [
                Sample(phase=self.phase_name, iteration=0, inputs=row) for row in points
            ]
            samples = self.ctx.store.add_many(samples)

        return self._evaluate_batch(samples)
