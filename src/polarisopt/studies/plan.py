"""Dry-run pre-flight: stage one sample, render its JobSpec, stop before sbatch.

Used by ``polarisopt plan`` and ``polarisopt validate --deep`` to catch
operational failures (missing modules, missing apptainer, wrong scenario
output key, bad parameter file path, ...) without actually submitting
the Slurm job.

Output is a structured dict with everything a human would want to see:

- the rendered JobSpec command
- the per-sample workspace path (so the user can poke around)
- if Slurm: the script that *would* have been sbatched
- if PolarisSimulator: where the result file *would* be written

Returns a :class:`PlanReport` you can ``render()`` to a string.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from polarisopt.config import load_study_config
from polarisopt.config.schema import StaticPhaseConfig, StudyConfig
from polarisopt.design.base import make_design
from polarisopt.parameters import ParameterSpace
from polarisopt.parameters.injection import load_parameter_file
from polarisopt.parameters.space import parameter_space_from_records
from polarisopt.runners.base import JobSpec
from polarisopt.runners.factory import make_runner
from polarisopt.runners.slurm import SlurmRunner
from polarisopt.samples.sample import Sample
from polarisopt.simulator.base import make_simulator
from polarisopt.utils.logging import get_logger
from polarisopt.utils.paths import workspace_layout

log = get_logger(__name__)


@dataclass
class PlanReport:
    """What ``polarisopt plan`` produced.

    Attributes
    ----------
    sample_inputs : np.ndarray
        The actual input vector that would be fed to the first sample.
    workspace : Path
        Per-sample workspace where the dry run staged files.
    job_spec : JobSpec
        The :class:`JobSpec` ``simulator.prepare()`` returned.
    script_path : Path or None
        For Slurm runs, the path to the generated sbatch script (not
        submitted).
    errors : list of str
        Anything fatal that prevented full planning.
    warnings : list of str
        Non-fatal issues (e.g. parameter not found in target JSON).
    info : list of str
        Side observations (parameter count, simulator type, ...).
    """

    sample_inputs: np.ndarray = field(default_factory=lambda: np.empty(0))
    workspace: Path | None = None
    job_spec: JobSpec | None = None
    script_path: Path | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        lines: list[str] = []
        if self.info:
            lines.append("info:")
            lines.extend(f"  - {m}" for m in self.info)
        if self.workspace:
            lines.append(f"sample workspace: {self.workspace}")
        if self.script_path:
            lines.append(f"slurm script:    {self.script_path}")
        if self.job_spec is not None:
            lines.append("job command:")
            for ln in self.job_spec.command.splitlines() or [""]:
                lines.append(f"  {ln}")
            if self.job_spec.env:
                lines.append("environment:")
                for k, v in self.job_spec.env.items():
                    lines.append(f"  {k}={v}")
        if self.warnings:
            lines.append("warnings:")
            lines.extend(f"  - {m}" for m in self.warnings)
        if self.errors:
            lines.append("errors:")
            lines.extend(f"  - {m}" for m in self.errors)
        else:
            lines.append("plan ok")
        return "\n".join(lines)


def _build_space(config: StudyConfig) -> ParameterSpace:
    if config.parameters.inline is not None:
        return parameter_space_from_records(config.parameters.inline)
    return load_parameter_file(config.parameters.source)  # type: ignore[arg-type]


def _first_sample_inputs(config: StudyConfig, space: ParameterSpace) -> np.ndarray:
    """Produce a single representative sample-input vector for the dry run.

    For static phases we instantiate the first phase's design and pick
    its first row. For sequential phases without a warm-up, we synthesize
    the midpoint of the space.
    """
    phase = config.phases[0]
    if isinstance(phase, StaticPhaseConfig):
        design = make_design({"type": phase.design.type, "options": phase.design.options})
        rng = np.random.default_rng(config.seed if config.seed is not None else 0)
        pts = design.generate(space, rng=rng)
        return np.asarray(pts[0])
    # Sequential phase — use warm-up if set, else the bounds midpoint.
    if getattr(phase, "warm_up", None) is not None:
        design = make_design(
            {"type": phase.warm_up.type, "options": phase.warm_up.options}  # type: ignore[union-attr]
        )
        rng = np.random.default_rng(config.seed if config.seed is not None else 0)
        return np.asarray(design.generate(space, rng=rng)[0])
    bounds = space.bounds
    return (bounds[:, 0] + bounds[:, 1]) / 2.0


def plan_study(
    config: Path | str | StudyConfig,
    *,
    sample_index: int = 0,
    workspace_override: Path | None = None,
) -> PlanReport:
    """Build a :class:`PlanReport` by staging one sample and rendering its JobSpec.

    Does **not** submit the job. The per-sample workspace is left intact
    so the user can inspect it.

    Parameters
    ----------
    config : path or StudyConfig
        Study YAML path or pre-loaded config.
    sample_index : int, optional
        Which sample id to use for the staged folder name. Default 0.
    workspace_override : path, optional
        Use this directory instead of ``<config.workspace>/experiments/``
        for the dry run. Recommended for "I don't want to clutter the
        real run dir" workflows.

    Returns
    -------
    PlanReport
    """
    report = PlanReport()
    cfg = load_study_config(config) if isinstance(config, (str, Path)) else config
    report.info.append(f"study: {cfg.name}")
    report.info.append(f"simulator: {cfg.simulator.type}")
    report.info.append(f"runner: {cfg.runner.type}")

    try:
        space = _build_space(cfg)
    except Exception as exc:  # noqa: BLE001
        report.errors.append(f"parameters: {exc}")
        return report
    report.info.append(f"parameters: {space.ndim} ({', '.join(space.names)})")

    try:
        simulator = make_simulator(
            {"type": cfg.simulator.type, "options": cfg.simulator.options}
        )
    except Exception as exc:  # noqa: BLE001
        report.errors.append(f"simulator construction: {exc}")
        return report

    try:
        inputs = _first_sample_inputs(cfg, space)
    except Exception as exc:  # noqa: BLE001
        report.errors.append(f"design.generate (or warm_up): {exc}")
        return report
    report.sample_inputs = inputs

    workspace_root = workspace_override or (
        workspace_layout(cfg.workspace)["experiments"] / "plan-sample"
    )
    workspace_root.mkdir(parents=True, exist_ok=True)
    sample = Sample(id=sample_index, phase="plan", inputs=inputs)
    try:
        spec = simulator.prepare(sample, space, workspace_root)
    except Exception as exc:  # noqa: BLE001
        report.errors.append(f"simulator.prepare: {exc}")
        report.workspace = workspace_root
        return report
    report.workspace = workspace_root
    report.job_spec = spec

    # If the runner is Slurm, render the sbatch script (don't submit).
    if cfg.runner.type == "slurm":
        try:
            runner = make_runner(
                {"type": cfg.runner.type, "options": cfg.runner.options}
            )
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"runner construction: {exc}")
            return report
        if not isinstance(runner, SlurmRunner):
            return report
        try:
            resources = spec.extra.get("resources", runner._default)  # noqa: SLF001
            script_text = runner._render_script(spec, resources)  # noqa: SLF001
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"slurm script render: {exc}")
            return report
        script_path = workspace_root / "plan.slurm"
        script_path.write_text(script_text)
        report.script_path = script_path

    return report
