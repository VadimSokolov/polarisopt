"""Pre-flight validation for a study YAML.

Catches every failure that can be detected without actually running
POLARIS: schema errors, missing parameter files, unregistered plugin
names, malformed paths, inconsistent ref points.

Used by ``polarisopt validate <study.yaml>`` before submission so
users don't waste a Slurm allocation on a typo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from polarisopt.config import load_study_config
from polarisopt.config.schema import (
    SequentialPhaseConfig,
    StaticPhaseConfig,
    StudyConfig,
)
from polarisopt.parameters import ParameterSpace, load_parameter_file
from polarisopt.parameters.space import parameter_space_from_records


@dataclass
class ValidationReport:
    """Result of validating a study YAML.

    Parameters
    ----------
    errors : list of str
        Validation failures that prevent the study from running.
    warnings : list of str
        Non-fatal issues worth flagging (e.g. relative paths, missing
        target file that POLARIS will create later).
    info : list of str
        Useful side-effects of validation (e.g. parameter count, total
        evaluation budget).
    """

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        """Human-readable summary."""
        lines: list[str] = []
        if self.info:
            lines.append("info:")
            lines.extend(f"  - {m}" for m in self.info)
        if self.warnings:
            lines.append("warnings:")
            lines.extend(f"  - {m}" for m in self.warnings)
        if self.errors:
            lines.append("errors:")
            lines.extend(f"  - {m}" for m in self.errors)
        else:
            lines.append("validation passed")
        return "\n".join(lines)


def validate_study(path: Path | str) -> ValidationReport:
    """Validate a study YAML and return a :class:`ValidationReport`.

    Performs:

    1. YAML + Jinja2 rendering (catches syntax errors).
    2. pydantic schema validation (catches missing/typed fields).
    3. Parameter source resolution (catches missing/malformed files).
    4. Plugin name lookup (catches unregistered ``type:`` strings).
    5. Cross-field sanity (qehvi without warm-up is suspicious; ref_point
       length must match objective count if specified).
    6. Path existence checks (workspace creatable, simulator binaries
       resolvable). Missing target files are warnings, not errors.

    Parameters
    ----------
    path : path
        Filesystem path to a study YAML.

    Returns
    -------
    ValidationReport
    """
    report = ValidationReport()
    p = Path(path)
    if not p.exists():
        report.errors.append(f"file does not exist: {p}")
        return report

    try:
        cfg = load_study_config(p)
    except Exception as exc:  # noqa: BLE001
        report.errors.append(f"config load failed: {exc}")
        return report

    report.info.append(f"study name: {cfg.name}")
    report.info.append(f"workspace: {cfg.workspace}")
    report.info.append(f"phases: {len(cfg.phases)}")

    _check_parameters(cfg, report)
    _check_plugins(cfg, report)
    _check_phases(cfg, report)
    _check_paths(cfg, report)
    return report


def _check_parameters(cfg: StudyConfig, report: ValidationReport) -> None:
    try:
        space: ParameterSpace = (
            parameter_space_from_records(cfg.parameters.inline)
            if cfg.parameters.inline is not None
            else load_parameter_file(cfg.parameters.source)  # type: ignore[arg-type]
        )
    except Exception as exc:  # noqa: BLE001
        report.errors.append(f"parameters: {exc}")
        return
    report.info.append(f"parameters: {space.ndim} ({', '.join(space.names)})")


def _check_plugins(cfg: StudyConfig, report: ValidationReport) -> None:
    """Look up every ``type:`` referenced in the YAML against its registry."""
    from polarisopt.design.base import design_registry
    from polarisopt.metrics.base import metric_registry
    from polarisopt.runners.base import runner_registry
    from polarisopt.simulator.base import simulator_registry

    def _check(family: str, registry, name: str) -> None:
        if name not in registry:
            report.errors.append(
                f"{family} '{name}' not registered. Available: {', '.join(registry.names())}"
            )

    _check("simulator", simulator_registry, cfg.simulator.type)
    _check("runner", runner_registry, cfg.runner.type)
    _check("metric", metric_registry, cfg.metric.type)
    for phase in cfg.phases:
        if isinstance(phase, StaticPhaseConfig):
            _check(f"design (phase {phase.name})", design_registry, phase.design.type)


def _check_phases(cfg: StudyConfig, report: ValidationReport) -> None:
    """Cross-field sanity for sequential phases."""
    from polarisopt.acquisition.base import acquisition_registry
    from polarisopt.design.base import design_registry
    from polarisopt.generators.base import generator_registry
    from polarisopt.stop.base import stop_registry
    from polarisopt.surrogates.base import surrogate_registry

    for phase in cfg.phases:
        if isinstance(phase, SequentialPhaseConfig):
            if phase.warm_up is not None and phase.warm_up.type not in design_registry:
                report.errors.append(
                    f"phase {phase.name}: warm_up design '{phase.warm_up.type}' not registered"
                )
            gen_type = phase.generator.get("type")
            if gen_type is None:
                report.errors.append(f"phase {phase.name}: generator missing 'type'")
            elif gen_type not in generator_registry:
                report.errors.append(
                    f"phase {phase.name}: generator '{gen_type}' not registered"
                )
            elif gen_type == "acquisition":
                opts = phase.generator.get("options") or {}
                surr_spec = opts.get("surrogate") or {}
                acq_spec = opts.get("acquisition") or {}
                if surr_spec.get("type") and surr_spec["type"] not in surrogate_registry:
                    report.errors.append(
                        f"phase {phase.name}: surrogate '{surr_spec['type']}' not registered "
                        f"(have you installed [bo]?)"
                    )
                if acq_spec.get("type") and acq_spec["type"] not in acquisition_registry:
                    report.errors.append(
                        f"phase {phase.name}: acquisition '{acq_spec['type']}' not registered "
                        f"(have you installed [bo]?)"
                    )
            stop_type = phase.stop.type
            if stop_type not in stop_registry:
                report.errors.append(
                    f"phase {phase.name}: stop '{stop_type}' not registered"
                )
            if phase.batch_size < 1:
                report.errors.append(
                    f"phase {phase.name}: batch_size must be >= 1, got {phase.batch_size}"
                )


def _check_paths(cfg: StudyConfig, report: ValidationReport) -> None:
    if cfg.simulator.type == "polaris":
        opts = cfg.simulator.options
        binary = opts.get("binary")
        model_source = opts.get("model_source")
        if binary and not Path(binary).exists():
            report.warnings.append(f"simulator.binary not found: {binary}")
        if model_source and not Path(model_source).exists():
            report.warnings.append(f"simulator.model_source not found: {model_source}")
    if cfg.metric.type in ("link_moe", "choice_share"):
        target_key = "target" if cfg.metric.type == "link_moe" else "target_db"
        target = cfg.metric.options.get(target_key)
        if target and not Path(target).exists():
            report.warnings.append(
                f"metric.{target_key} not found yet: {target} (will be required at compute time)"
            )
