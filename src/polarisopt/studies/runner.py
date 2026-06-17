"""Glue: turn a validated StudyConfig + workspace into a chain of Study phases."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from polarisopt.config.schema import (
    ParametersConfig,
    SequentialPhaseConfig,
    StaticPhaseConfig,
    StudyConfig,
)
from polarisopt.design.base import make_design
from polarisopt.generators.base import make_generator
from polarisopt.metrics.base import make_metric
from polarisopt.parameters import ParameterSpace, load_parameter_file
from polarisopt.parameters.space import parameter_space_from_records
from polarisopt.runners.factory import make_runner
from polarisopt.samples.sample import Sample
from polarisopt.samples.store import SampleStore
from polarisopt.simulator.base import make_simulator
from polarisopt.stop.base import make_stop
from polarisopt.studies.base import StudyContext, StudyError
from polarisopt.studies.ops import simulator_config_fingerprint
from polarisopt.studies.sequential import SequentialDesignStudy, SequentialPhase
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
        # logs/ and scripts/ are *available* paths (see workspace_layout)
        # for callers that need them; we don't pre-create them since
        # polarisopt's own per-sample logs live inside experiments/sim-NNN/.

        self.store = store or SampleStore.open(self.layout["db"], config.name)
        self.space = _build_space(config.parameters)
        # Pluck Study-level poll/orphan knobs out of runner.options before
        # building the runner itself — they belong to the orchestrator loop,
        # not to the runner backend.
        runner_options = dict(config.runner.options)
        self.poll_interval: float = float(runner_options.pop("poll_interval", 5.0))
        self.orphan_threshold: int = int(runner_options.pop("orphan_threshold", 3))
        self.heartbeat_interval: float = float(runner_options.pop("heartbeat_interval", 300.0))
        self.max_retries: int = int(runner_options.pop("max_retries", 0))
        if self.max_retries < 0:
            raise StudyError(f"max_retries must be >= 0, got {self.max_retries}")
        self.runner = make_runner({"type": config.runner.type, "options": runner_options})
        self.config_fingerprint: str = simulator_config_fingerprint(config)
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
                poll_interval=self.poll_interval,
                orphan_threshold=self.orphan_threshold,
                heartbeat_interval=self.heartbeat_interval,
                config_fingerprint=self.config_fingerprint,
                max_retries=self.max_retries,
            )
            if isinstance(phase, StaticPhaseConfig):
                design = make_design({"type": phase.design.type, "options": phase.design.options})
                study = StaticDesignStudy(ctx, design, phase_name=phase.name)
            elif isinstance(phase, SequentialPhaseConfig):
                warm = (
                    make_design({"type": phase.warm_up.type, "options": phase.warm_up.options})
                    if phase.warm_up is not None
                    else None
                )
                generator = make_generator(phase.generator)
                stop = make_stop(phase.stop.model_dump())
                seq_phase = SequentialPhase(
                    name=phase.name,
                    generator=generator,
                    stop=stop,
                    warm_up=warm,
                    batch_size=phase.batch_size,
                    minimize=phase.minimize,
                )
                study = SequentialDesignStudy(ctx, seq_phase)
            else:
                raise StudyError(f"unknown phase config type: {type(phase).__name__}")
            log.info("Running phase %r", phase.name)
            all_samples.extend(study.run())
        return all_samples


def run_study(config_path: Path | str) -> list[Sample]:
    """One-shot convenience: load a YAML config and execute it."""
    from polarisopt.config import load_study_config

    config = load_study_config(config_path)
    return StudyRunner(config).run()
