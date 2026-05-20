"""PolarisConvergenceSimulator — drive a sample through polarislib's convergence loop.

Wraps :class:`PolarisSimulator` but replaces the single-binary invocation
with a call to a user-supplied Python *runner* script. That runner is
responsible for driving :class:`polarislib.Polaris.run()` so iteration
semantics — population synthesis, ABM init, optional DTA passes — are
governed by polarislib's ``ConvergenceConfig`` rather than by polarisopt.

Use this simulator when:

- you need polarislib to manage iteration flags / ``scenario_mods.py``
  rather than cooking up the right ``scenario_abm.json`` yourself; or
- you need ABM-only at non-default ``population_scale_factor`` (a
  ``ConvergenceConfig`` knob that doesn't have a direct scenario JSON
  equivalent).

The master process **does not** import polarislib. Only the user-supplied
runner does, on the slave side. Master/slave separation is preserved.

Custom polarislib knobs
-----------------------

Anything you'd set on ``Polaris.run_config`` can be forwarded to the
runner via ``runner_options``. They're passed on the command line as
``--<dashified-key>=<value>``. Example::

    runner_options:
      population_scale_factor: 0.05
      num_abm_runs: 1
      num_dta_runs: 0
      do_skim: false

A canonical runner script lives at ``run_scripts/polarisopt_runner.py``
in the calibration project; copy it as a starting point.
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path
from typing import Any

from polarisopt.parameters import ParameterSpace
from polarisopt.parameters.injection import inject_values
from polarisopt.runners.base import JobSpec
from polarisopt.samples.sample import Sample
from polarisopt.simulator.base import SimulatorError, simulator_registry
from polarisopt.simulator.polaris import PolarisSimulator
from polarisopt.utils.logging import get_logger

# polarislib's iteration types — see polaris/runs/convergence/convergence_iteration.py
ITER_TYPE_TO_BASE = {
    "abm_init": "01_abm_init_iteration",
    "skim": "00_skim_iteration",
    "pop_synth": "02_pop_synth_iteration",
    "dta": "dta_iteration",
}

log = get_logger(__name__)


@simulator_registry.register("polaris_convergence")
class PolarisConvergenceSimulator(PolarisSimulator):
    """Drive a sample through polarislib's convergence loop via an external runner.

    The Slurm job for each sample runs::

        <setup_commands>
        <python_interpreter> <runner_script> <workspace> [--key=val ...]

    The runner is expected to call ``Polaris.from_dir(workspace).run()``
    after setting any ``run_config`` knobs. ``collect_output`` is inherited
    from :class:`PolarisSimulator` but :meth:`_resolve_output_dir` is
    overridden to find polarislib's ``<db_name>_<iter_str>`` directory
    naming.

    Parameters
    ----------
    runner_script : path
        Absolute path to the Python runner the slave will invoke. Must
        accept the workspace path as its first positional argument; any
        additional ``runner_options`` get forwarded as ``--key=value``.
    python_interpreter : str, optional
        Path to the Python interpreter that has polarislib + deps
        installed. Defaults to :data:`sys.executable`.
    iteration_type : {"abm_init", "skim", "pop_synth", "dta"}
        Which polarislib iteration's output directory to read in
        :meth:`collect_output`. Default ``"abm_init"``.
    runner_options : dict, optional
        Forwarded to the runner script as ``--<dashified-key>=<value>``
        command-line flags. Use for ``population_scale_factor``,
        ``num_abm_runs``, etc. Booleans become ``true``/``false``.
    setup_commands : list of str, optional
        Shell commands to prepend to the JobSpec command before the
        runner invocation. Use for ``module load`` lines on shared HPC
        clusters. Each entry is one shell line (joined with newlines).
    env : dict[str, str], optional
        Extra environment variables for the JobSpec. Stacked with
        ``POLARIS_NUM_THREADS``.
    Other parameters inherited from :class:`PolarisSimulator`.

    Notes
    -----
    The ``binary`` parameter from :class:`PolarisSimulator` is still
    required (the runner uses it to locate the SIF / executable) but
    polarisopt does not invoke it directly — polarislib does, from
    inside the runner.

    Examples
    --------
    YAML:

    .. code-block:: yaml

        simulator:
          type: polaris_convergence
          options:
            runner_script: /lcrc/.../run_scripts/polarisopt_runner.py
            python_interpreter: /home/me/.conda/envs/polaris/bin/python
            iteration_type: abm_init
            runner_options:
              population_scale_factor: 0.05
              num_abm_runs: 1
              do_skim: false
            setup_commands:
              - "module purge"
              - "module load gcc/10.4 hdf5/1.12 libspatialite singularity"
            binary: /lcrc/.../polaris.sif
            model_source: /lcrc/.../DFW_2050_20251028
            scenario_file: scenario_abm.json
            output_db_filename: DFW-Demand.sqlite
            output_dir_key: ["Output controls", "output_dir_name"]
            num_threads: "16"
    """

    def __init__(
        self,
        *,
        runner_script: str,
        python_interpreter: str | None = None,
        iteration_type: str = "abm_init",
        runner_options: dict[str, Any] | None = None,
        setup_commands: list[str] | None = None,
        env: dict[str, str] | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self.runner_script = Path(runner_script)
        if not self.runner_script.exists():
            raise SimulatorError(f"runner_script not found: {self.runner_script}")
        self.python_interpreter = python_interpreter or sys.executable
        if iteration_type not in ITER_TYPE_TO_BASE:
            raise SimulatorError(
                f"iteration_type must be one of {sorted(ITER_TYPE_TO_BASE)}, got {iteration_type!r}"
            )
        self.iteration_type = iteration_type
        self.runner_options: dict[str, Any] = dict(runner_options or {})
        self.setup_commands: list[str] = list(setup_commands or [])
        self.extra_env: dict[str, str] = dict(env or {})

    def prepare(self, sample: Sample, space: ParameterSpace, workspace: Path) -> JobSpec:
        if sample.inputs.shape != (space.ndim,):
            raise SimulatorError(
                f"sample.inputs shape {sample.inputs.shape} != space.ndim={space.ndim}"
            )
        if not self.model_source.exists():
            raise SimulatorError(f"model_source does not exist: {self.model_source}")
        workspace.mkdir(parents=True, exist_ok=True)
        log.info("PolarisConvergenceSimulator: staging model into %s", workspace)
        self._transfer.copy(self.model_source, workspace, recursive=True)
        missing = inject_values(sample.inputs, space, workspace)
        if missing:
            log.warning(
                "PolarisConvergenceSimulator: parameters not found in JSONs: %s",
                missing,
            )

        runner_argv = [
            shlex.quote(self.python_interpreter),
            shlex.quote(str(self.runner_script)),
            shlex.quote(str(workspace)),
            f"--threads={shlex.quote(self.num_threads)}",
        ]
        for k, v in self.runner_options.items():
            flag = "--" + str(k).replace("_", "-")
            # ``--flag=value`` is one shell token; quote the value to
            # survive spaces / shell metacharacters in user-supplied
            # ``runner_options``.
            runner_argv.append(f"{flag}={shlex.quote(_arg_value(v))}")

        command_lines = list(self.setup_commands) + [" ".join(runner_argv)]
        command = "\n".join(command_lines)

        job_env: dict[str, str] = {"POLARIS_NUM_THREADS": self.num_threads, **self.extra_env}

        return JobSpec(
            name=f"polaris-conv-{sample.id or 'unsaved'}",
            command=command,
            cwd=workspace,
            stdout=workspace / "polaris.stdout.log",
            stderr=workspace / "polaris.stderr.log",
            env=job_env,
        )

    def collect_output(self, sample: Sample) -> dict[str, Any]:
        """Extend base class output dict with a ``demand_db`` alias.

        Choice-share-style metrics expect ``demand_db``; expose it
        alongside the base class's ``result_path``.
        """
        out = super().collect_output(sample)
        out["demand_db"] = out["result_path"]
        return out

    def _resolve_output_dir(self, workspace: Path, output_dirname: str) -> Path:
        """Find polarislib's iteration directory under ``workspace``.

        polarislib writes output to ``<db_name>_<iter_str>[_<N>]``:

        - ``<db_name>_<iter_str>_<N>``  when ``iteration_number`` is set
          (e.g. ``DFW_01_abm_init_iteration_0``)
        - ``<db_name>_<iter_str>``      when ``iteration_number is None``
          (e.g. ``DFW_01_abm_init_iteration``)

        ``output_dirname`` here is polarislib's ``db_name`` (the
        ``Output controls.output_dir_name`` value in ``scenario_abm.json``).

        Prefers the highest-numbered ``_<N>`` if any exist, otherwise
        falls back to the no-suffix directory.
        """
        iter_base = ITER_TYPE_TO_BASE[self.iteration_type]
        numbered_pattern = f"{output_dirname}_{iter_base}_*"
        unnumbered = workspace / f"{output_dirname}_{iter_base}"

        candidates: list[tuple[int, Path]] = []
        for d in workspace.glob(numbered_pattern):
            if not d.is_dir():
                continue
            suffix = d.name.rsplit("_", 1)[-1]
            try:
                n = int(suffix)
            except ValueError:
                continue
            candidates.append((n, d))

        if candidates:
            _, best = max(candidates, key=lambda kv: kv[0])
            return best

        if unnumbered.is_dir():
            return unnumbered

        raise SimulatorError(
            f"no output directory matching {numbered_pattern!r} "
            f"or {unnumbered.name!r} under {workspace}"
        )


def _arg_value(value: Any) -> str:
    """Render a runner_options value as a CLI arg."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
