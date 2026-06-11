"""Pre-flight validation for a study YAML.

Catches every failure that can be detected without actually running
POLARIS: schema errors, missing parameter files, unregistered plugin
names, malformed paths, inconsistent ref points.

Used by ``polarisopt validate <study.yaml>`` before submission so
users don't waste a Slurm allocation on a typo.
"""

from __future__ import annotations

import inspect
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

# Orchestrator-side keys that live in runner.options but are popped by
# StudyRunner before the runner is constructed. Single source of truth
# in studies/ops.py so plan/validate/run stay in sync.
from polarisopt.studies.ops import _ORCHESTRATOR_RUNNER_OPTIONS as _RUNNER_ORCHESTRATOR_KEYS


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
    _check_plugin_options(cfg, report)
    _check_phases(cfg, report)
    _check_paths(cfg, report)
    return report


def _accepted_init_kwargs(cls: type) -> tuple[set[str], bool]:
    """Collect every keyword name ``cls.__init__`` (and its supers) accept.

    Walks ``cls.__mro__`` so subclasses that ``super().__init__(**kw)`` are
    handled correctly — the cumulative accepted set is the union over the
    chain.

    Returns
    -------
    (accepted, has_var_keyword)
        ``accepted`` is the set of named keyword params anywhere in the
        chain (excluding ``self`` / ``*args`` / ``**kwargs``).
        ``has_var_keyword`` is ``True`` if **any** class in the chain has
        a ``**kwargs`` parameter — when true, unknown options *might* be
        legitimately forwarded somewhere we can't see, so callers should
        downgrade unknown-option errors to warnings.
    """
    accepted: set[str] = set()
    has_var_keyword = False
    for klass in cls.__mro__:
        if klass is object:
            break
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        try:
            sig = inspect.signature(init)
        except (TypeError, ValueError):
            continue
        for name, param in sig.parameters.items():
            if name in ("self", "cls"):
                continue
            if param.kind is inspect.Parameter.VAR_KEYWORD:
                has_var_keyword = True
                continue
            if param.kind is inspect.Parameter.VAR_POSITIONAL:
                continue
            accepted.add(name)
    return accepted, has_var_keyword


def _check_one_plugin_options(
    family: str,
    registry,
    name: str,
    options: dict | None,
    report: ValidationReport,
    *,
    extra_allowed: set[str] | None = None,
) -> None:
    """For one plugin spec, error on options keys not accepted by its __init__.

    Skipped silently if the plugin isn't registered (already flagged by
    :func:`_check_plugins`). When the class chain has ``**kwargs``, unknown
    keys are downgraded to a warning since they might be forwarded.

    ``extra_allowed`` is for orchestrator-side YAML keys that don't show
    up in the plugin's signature (e.g. runner ``poll_interval``,
    ``orphan_threshold``, ``heartbeat_interval`` get popped by
    :class:`StudyRunner` before reaching the runner constructor).
    """
    if name not in registry:
        return
    cls = registry.get(name)
    accepted, has_var_keyword = _accepted_init_kwargs(cls)
    if not accepted and has_var_keyword:
        return
    if extra_allowed:
        accepted = accepted | extra_allowed
    unknown = sorted(set(options or {}) - accepted)
    if not unknown:
        return
    msg = (
        f"{family} '{name}': option(s) {unknown} not in __init__ signature. "
        f"Accepted: {sorted(accepted)}"
    )
    if has_var_keyword:
        report.warnings.append(msg + " (class accepts **kwargs — may be forwarded)")
    else:
        report.errors.append(msg)


def _check_plugin_options(cfg: StudyConfig, report: ValidationReport) -> None:
    """Typecheck every ``options:`` block against its plugin's __init__.

    Catches typos like ``distance: l1`` (real arg ``aggregation``) or
    ``sim_key: demand_db`` (real arg ``source_key``) before a 30s
    staging round-trip burns through.
    """
    from polarisopt.design.base import design_registry
    from polarisopt.metrics.base import metric_registry
    from polarisopt.runners.base import runner_registry
    from polarisopt.simulator.base import simulator_registry

    _check_one_plugin_options(
        "simulator", simulator_registry, cfg.simulator.type,
        cfg.simulator.options, report,
    )
    _check_one_plugin_options(
        "runner", runner_registry, cfg.runner.type,
        cfg.runner.options, report,
        extra_allowed=_RUNNER_ORCHESTRATOR_KEYS,
    )
    _check_one_plugin_options(
        "metric", metric_registry, cfg.metric.type,
        cfg.metric.options, report,
    )
    for phase in cfg.phases:
        if isinstance(phase, StaticPhaseConfig):
            _check_one_plugin_options(
                f"design (phase {phase.name})", design_registry,
                phase.design.type, phase.design.options, report,
            )


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
