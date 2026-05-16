"""Glue: turn a validated StudyConfig + workspace into a chain of Study phases."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from polarisopt.config.schema import (
    ParametersConfig,
    StaticPhaseConfig,
    StudyConfig,
)
from polarisopt.design.base import make_design
from polarisopt.metrics.base import make_metric
from polarisopt.parameters import ParameterSpace, load_parameter_file
from polarisopt.parameters.space import parameter_space_from_records
from polarisopt.runners.factory import make_runner
from polarisopt.samples.sample import Sample
from polarisopt.samples.store import SampleStore
from polarisopt.simulator.base import make_simulator
from polarisopt.studies.base import StudyContext, StudyError
from polarisopt.studies.static import StaticDesignStudy
from polarisopt.utils.logging import get_logger
from polarisopt.utils.paths import workspace_layout

log = get_logger(__name__)


def _build_space(p: ParametersConfig) -> ParameterSpace:
    if p.source is not None:
        return load_parameter_file(p.source)
    assert p.inline is not None
    return parameter_space_from_records(p.inline)


class StudyRunner:
    """Build all components from a StudyConfig and run the phases in order."""

    def __init__(self, config: StudyConfig, *, store: SampleStore | None = None) -> None:
        self.config = config
        self.layout = workspace_layout(config.workspace)
        self.layout["root"].mkdir(parents=True, exist_ok=True)
        self.layout["experiments"].mkdir(parents=True, exist_ok=True)
        self.layout["logs"].mkdir(parents=True, exist_ok=True)
        self.layout["scripts"].mkdir(parents=True, exist_ok=True)

        self.store = store or SampleStore.open(self.layout["db"], config.name)
        self.space = _build_space(config.parameters)
        self.runner = make_runner({"type": config.runner.type, "options": config.runner.options})
        self.simulator = make_simulator({"type": config.simulator.type, "options": config.simulator.options})
        self.metric = make_metric({"type": config.metric.type, "options": config.metric.options})

        seed = config.seed if config.seed is not None else int(np.random.SeedSequence().entropy)
        self.rng = np.random.default_rng(seed)

    def run(self) -> list[Sample]:
        """Execute phases in order. Returns the concatenated sample list."""
        all_samples: list[Sample] = []
        for phase in self.config.phases:
            ctx = StudyContext(
                name=phase.name,
                space=self.space,
                workspace=self.layout["root"],
                store=self.store,
                runner=self.runner,
                simulator=self.simulator,
                metric=self.metric,
                rng=self.rng,
            )
            if isinstance(phase, StaticPhaseConfig):
                design = make_design({"type": phase.design.type, "options": phase.design.options})
                study = StaticDesignStudy(ctx, design, phase_name=phase.name)
            else:
                raise StudyError(
                    f"phase type {type(phase).__name__} not yet implemented in v0.1.0 — "
                    "sequential phases land in Week 3"
                )
            log.info("Running phase %r", phase.name)
            all_samples.extend(study.run())
        return all_samples


def run_study(config_path: Path | str) -> list[Sample]:
    """One-shot convenience: load a YAML config and execute it."""
    from polarisopt.config import load_study_config

    config = load_study_config(config_path)
    return StudyRunner(config).run()
