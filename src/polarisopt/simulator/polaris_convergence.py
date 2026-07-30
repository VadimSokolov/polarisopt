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

Runtime budget — important
--------------------------

polarislib's ``abm_init`` iteration runs a **full 24-hour network
simulation** regardless of ``num_dta_runs``. ``num_dta_runs=0`` does
**not** mean "ABM only / no traffic" — it means "no additional DTA
re-runs after the abm_init pass."

For choice-model calibration you usually want the cheapest credible
budget. Two ways to cap it:

- ``population_scale_factor: 0.01`` (1%) — full network, scaled
  population. ~5–20 min for DFW depending on hardware.
- ``do_skim: false`` — skip the LOS skimming pass if your scenario
  already has fresh skims.

If you truly want "choice models only, no traffic at all," ask whether
polarislib's ``pop_synth`` iteration_type fits — it stops after
population synthesis and choice-model evaluation without dispatching to
the C++ traffic simulator. That's a different ``iteration_type``, not
a knob.
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
from polarisopt.simulator.polaris import PolarisSimulator, _arg_value
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
    single_iteration : bool, optional
        Sugar for the choice-model calibration use case: forces
        ``num_abm_runs=0`` and ``num_dta_runs=0`` into ``runner_options``
        so polarislib only runs the configured ``iteration_type`` once,
        with no follow-up ``normal_iteration``. Roughly halves wall
        time. Raises if these keys are already set to non-zero values.
        :meth:`collect_output` enforces by failing if any other
        iteration_type's directory turns up. Default ``False``.
    disable_async_callback : bool, optional
        Forwarded as ``--disable-async-callback=true|false``. When true,
        the runner script is expected to pass a no-op for polarislib's
        ``async_end_of_loop_fn`` so per-iteration DBs are not tarballed
        out from under metrics that need to read them. Default
        ``True`` — "preserve artifacts" is the right stance for the
        calibration use case. Explicit
        ``runner_options.disable_async_callback`` overrides.
    seed_per_sample : bool, optional
        First-class support for POLARIS RNG-noise studies (v0.23+).
        When ``True``, the effective seed passed to the runner as
        ``--seed=<n>`` is ``runner_options["seed"] + sample.id``
        (base seed defaults to 0 if not set). Each sample at the
        same X gets a distinct POLARIS seed, enabling the
        "N repeats at one point" pattern that's the standard first
        diagnostic for any noisy-simulator BO (POLARIS Phase 3B.0 /
        3D). ``--seed-per-sample`` itself is **not** forwarded to the
        runner — polarisopt computes the effective seed before
        submission, so downstream shims that also honor
        ``seed_per_sample`` don't double-add. Default ``False``.
    nested_asc_contraction : dict, optional
        v0.28+ (P4). Configures BLP-1995 nested-ASC contraction
        (β outer / ASC inner) around every POLARIS run. polarisopt
        validates the config shape here and forwards each field to
        the runner script as ``--nested-asc-<key>=<value>`` flags;
        the runner is expected to invoke polarislib's
        ``runs.calibrate.mode_choice.calibrate`` iteratively after
        each simulation. polarisopt does NOT import polarislib and
        does NOT invoke the calibrator itself.

        Shape (all fields except ``enabled`` are optional; validated
        only when ``enabled=True``):

        - ``enabled`` (bool, required): master switch.
        - ``calibrator`` (str): dotted path to the polarislib
          calibrator (documentation only — polarisopt doesn't call it).
        - ``num_planned_activity_iterations`` (int, default 3):
          contraction iterations per sample.
        - ``step_size`` (float, default 2.0): ASC update step.
        - ``target_csv_dir`` (str): where the runner reads target
          shares from.
        - ``cache_post_contraction`` (bool, default True): keep the
          post-contraction ASCs for reproducibility.
        - ``timeout_minutes`` (float, default 30).
        - ``on_convergence_failure`` (str): ``use_last`` (default) or
          ``mark_sample_failed``.

        Unknown keys are rejected at construction — catches typos
        like ``timout_minutes`` that would otherwise silently mean
        "no timeout". Default ``None`` (feature off; no flags
        forwarded, no config validation).

        **v0.32 interaction with output-dir resolution**: when
        enabled, polarislib names each contraction iteration
        ``<db>_<iter>_calib_<N>``. :meth:`_resolve_output_dir`
        skips those ``_calib_*`` dirs and returns the real DTA
        iteration dir (fixes the DFW Phase 7 "all samples produce
        empty Trip table" bug — see CHANGELOG v0.32.0).
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

    YAML with nested-ASC contraction on (v0.28+ / β-calibration):

    .. code-block:: yaml

        simulator:
          type: polaris_convergence
          options:
            runner_script: /lcrc/.../run_scripts/polarisopt_runner.py
            iteration_type: abm_init
            single_iteration: true
            runner_options:
              population_scale_factor: 0.01
            binary: /lcrc/.../polaris.sif
            model_source: /lcrc/.../DFW_2050_20251028
            scenario_file: scenario_abm.json
            output_db_filename: DFW-Demand.sqlite
            num_threads: "16"
            nested_asc_contraction:
              enabled: true
              calibrator: polarislib.runs.calibrate.mode_choice.calibrate
              num_planned_activity_iterations: 3
              step_size: 2.0
              target_csv_dir: calibration_targets
              cache_post_contraction: true
              timeout_minutes: 30
              on_convergence_failure: use_last
    """

    # polarislib scenarios use ``output_dir_name``, not the base class's
    # ``output_directory``. Override so YAML doesn't need to spell it out
    # for every polaris_convergence study.
    DEFAULT_OUTPUT_DIR_KEY: tuple[str, str] = ("Output controls", "output_dir_name")

    # Soft whitelist of ``runner_options`` keys polarisopt knows polarislib
    # understands. Used by ``polarisopt plan`` to warn (not error) on
    # likely typos like ``population_scal_factor``. Add to this set when
    # polarislib gains new ``ConvergenceConfig`` fields. Branch-specific
    # knobs that aren't on this list still pass through fine.
    KNOWN_RUNNER_OPTIONS: frozenset[str] = frozenset({
        "population_scale_factor",
        "num_abm_runs",
        "num_dta_runs",
        "do_skim",
        "do_warm_start",
        "do_calibration",
        "do_dta",
        "do_abm",
        "do_pop_synth",
        "do_init",
        "current_iteration",
        "start_iteration_from",
        "archive_dir",
        "db_name",
        "output_dir_name",
        "polaris_exe",
        "fixed_demand",
        "fixed_supply",
        "max_concurrent",
        "disable_async_callback",
        "seed",
    })

    def unknown_runner_options(self) -> list[str]:
        """Return ``runner_options`` keys not in :attr:`KNOWN_RUNNER_OPTIONS`.

        Used by ``polarisopt plan`` to surface likely typos before a
        compute allocation is burned. The whitelist is *soft* — branch-
        specific polarislib knobs that aren't listed still pass through.
        """
        return sorted(set(self.runner_options) - self.KNOWN_RUNNER_OPTIONS)

    def __init__(
        self,
        *,
        runner_script: str,
        python_interpreter: str | None = None,
        iteration_type: str = "abm_init",
        runner_options: dict[str, Any] | None = None,
        setup_commands: list[str] | None = None,
        env: dict[str, str] | None = None,
        single_iteration: bool = False,
        disable_async_callback: bool = True,
        seed_per_sample: bool = False,
        nested_asc_contraction: dict[str, Any] | None = None,
        **kw: Any,
    ) -> None:
        # polarislib workloads stage 1.5–3 GB per sample; if 100 FAILED
        # samples accumulate their workspaces nobody asked to preserve,
        # the filesystem fills. Default cleanup_on_failure=True for
        # polaris_convergence is the safer call. The base PolarisSimulator
        # keeps the default False because forensic preservation is the
        # more common use case for hand-rolled simulations. Explicit
        # ``cleanup_on_failure`` in YAML wins.
        kw.setdefault("cleanup_on_failure", True)
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
        self.single_iteration: bool = bool(single_iteration)
        self.seed_per_sample: bool = bool(seed_per_sample)
        # If the caller also set ``seed_per_sample`` inside runner_options
        # (older YAMLs targeting the project-local shim), the top-level
        # kwarg takes precedence and we strip the runner_options entry so
        # the shim doesn't ALSO compute the offset — polarisopt owns the
        # arithmetic now.
        if self.seed_per_sample:
            self.runner_options.pop("seed_per_sample", None)
        if self.single_iteration:
            # The "ABM-only / choice-models-only" mode for calibration:
            # polarislib runs abm_init then stops, no follow-up
            # normal_iteration that would double wall-time. Reject conflicts
            # so users don't silently get the un-shortcut'd run.
            for forced_key in ("num_abm_runs", "num_dta_runs"):
                if self.runner_options.get(forced_key) not in (None, 0):
                    raise SimulatorError(
                        f"single_iteration=True forces {forced_key}=0; "
                        f"runner_options conflict: {forced_key}="
                        f"{self.runner_options[forced_key]!r}. Pick one."
                    )
                self.runner_options[forced_key] = 0
        # Default to preserving per-iteration artifacts. polarislib's stock
        # async_end_of_loop_fn tarballs the iteration DBs, which breaks
        # any metric that needs to open them. Runner scripts are expected
        # to honor ``--disable-async-callback=true`` by passing a no-op
        # for ``async_end_of_loop_fn`` to ``Polaris.run()``. Explicit
        # ``runner_options.disable_async_callback`` wins.
        self.disable_async_callback: bool = bool(disable_async_callback)
        if "disable_async_callback" not in self.runner_options:
            self.runner_options["disable_async_callback"] = self.disable_async_callback

        # v0.28 P4: nested-ASC contraction config (BLP-1995 style, β outer,
        # ASC inner). polarisopt validates the config shape and forwards
        # the flags to the runner script; the runner is expected to
        # invoke ``polarislib.runs.calibrate.mode_choice.calibrate`` (or
        # the configured ``calibrator``) after each POLARIS run at
        # candidate β. Off by default; must set ``enabled: true``.
        self.nested_asc_contraction: dict[str, Any] | None = (
            _validate_nested_asc(nested_asc_contraction)
            if nested_asc_contraction is not None
            else None
        )

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

        # Per-sample runner_options: start from the shared config, apply
        # any per-sample overrides (currently just seed_per_sample) into
        # a fresh copy so we don't mutate the constructor's dict.
        per_sample_options: dict[str, Any] = dict(self.runner_options)
        if self.seed_per_sample:
            base_seed = int(per_sample_options.get("seed", 0) or 0)
            offset = int(sample.id) if sample.id is not None else 0
            per_sample_options["seed"] = base_seed + offset
            log.info(
                "seed_per_sample: base=%d + sim_id=%d -> POLARIS seed=%d",
                base_seed, offset, per_sample_options["seed"],
            )

        runner_argv = [
            shlex.quote(self.python_interpreter),
            shlex.quote(str(self.runner_script)),
            shlex.quote(str(workspace)),
            f"--threads={shlex.quote(self.num_threads)}",
        ]
        # v0.28 nested-ASC forwarding — flags carry a namespaced
        # ``--nested-asc-<key>`` prefix so the runner can distinguish
        # them from ordinary runner_options and skip them silently if
        # it doesn't implement contraction. Only forwarded when the
        # contraction is enabled AND the config validated cleanly.
        if self.nested_asc_contraction is not None and self.nested_asc_contraction.get("enabled"):
            for k, v in self.nested_asc_contraction.items():
                per_sample_options[f"nested_asc_{k}"] = v

        for k, v in per_sample_options.items():
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
        """Extend base class output dict with polarislib-specific paths.

        Adds:

        - ``demand_db`` — alias for ``result_path``; choice-share metrics
          expect this key.
        - ``progress_log_path`` — absolute path to the POLARIS binary's
          per-iteration ``log/polaris_progress.log``, if present. This
          is what you tail to see "are we at sim-hour 10 or sim-hour
          20" inside a running iteration. The polarisopt-side wrapper
          log (``polaris.stdout.log``) is a different, coarser thing.

        Normalizes ``iteration``: the base class returns ``None`` for
        polarislib's unsuffixed ``<db>_<iter_str>`` directory
        (``iteration_number is None`` case), which crashes downstream
        metrics that expect an integer. We map "unsuffixed = baseline"
        to ``iteration: 0`` so a single number-line works uniformly.
        """
        out = super().collect_output(sample)
        out["demand_db"] = out["result_path"]
        progress = Path(out["output_dir"]) / "log" / "polaris_progress.log"
        out["progress_log_path"] = str(progress) if progress.exists() else None
        if out.get("iteration") is None:
            out["iteration"] = 0
        if self.single_iteration:
            self._assert_no_extra_iteration_dirs(sample)
        return out

    def _assert_no_extra_iteration_dirs(self, sample: Sample) -> None:
        """When ``single_iteration=True``, fail loudly if other iteration_type
        dirs slipped past the polarislib config — that means the runner
        script didn't honor the forced ``num_abm_runs=0/num_dta_runs=0``
        and the wall-time savings didn't actually happen.
        """
        if sample.folder is None:
            return
        expected_base = ITER_TYPE_TO_BASE[self.iteration_type]
        unexpected: list[str] = []
        for d in sample.folder.iterdir():
            if not d.is_dir():
                continue
            for other_iter, other_base in ITER_TYPE_TO_BASE.items():
                if other_iter == self.iteration_type:
                    continue
                # Match polarislib's <db>_<iter_str>[_<N>] convention.
                if other_base in d.name and expected_base not in d.name:
                    unexpected.append(d.name)
                    break
        if unexpected:
            raise SimulatorError(
                f"single_iteration=True but found extra iteration dir(s) "
                f"in {sample.folder}: {unexpected}. The runner script likely "
                f"ignored --num-abm-runs=0 / --num-dta-runs=0."
            )

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

        v0.32 fix (DFW report): when ``nested_asc_contraction`` is
        enabled, ``polarislib.runs.calibrate.calibration.get_output_dir``
        writes per-contraction-iteration artifacts named
        ``<db_name>_<iter_str>_calib_<N>``. The old glob picked up
        those `_calib_N` directories (because `rsplit("_", 1)[-1]`
        parses `N` as an int), so the highest-N contraction dir won
        even though its Trip table is empty (contraction is
        activity-generation only). The Phase 7 DFW run lost 33/33
        samples to garbage metric vectors before the workaround.
        Skip `_calib_*`-suffixed dirs on the nested-ASC path — they
        are polarislib contraction artifacts, not real DTA iterations.
        """
        iter_base = ITER_TYPE_TO_BASE[self.iteration_type]
        numbered_pattern = f"{output_dirname}_{iter_base}_*"
        unnumbered = workspace / f"{output_dirname}_{iter_base}"
        nested_asc_on = (
            self.nested_asc_contraction is not None
            and bool(self.nested_asc_contraction.get("enabled"))
        )

        candidates: list[tuple[int, Path]] = []
        for d in workspace.glob(numbered_pattern):
            if not d.is_dir():
                continue
            # v0.32 fix: exclude polarislib contraction artifacts when the
            # feature that creates them is on. Match on the substring
            # `_calib_` (before the trailing numeric suffix), so we still
            # accept the real DTA iteration dirs `_<N>`.
            if nested_asc_on and "_calib_" in d.name:
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


# v0.28 P4: nested-ASC contraction config validation.
#
# polarisopt owns the config shape; the runner script owns the actual
# calibrator invocation. We only validate here — we don't import
# polarislib (runners on the cluster do that themselves).

_NESTED_ASC_ALLOWED_KEYS = frozenset({
    "enabled",
    "calibrator",
    "num_planned_activity_iterations",
    "step_size",
    "target_csv_dir",
    "cache_post_contraction",
    "timeout_minutes",
    "on_convergence_failure",
})
_NESTED_ASC_ON_FAILURE = ("use_last", "mark_sample_failed")


def _validate_nested_asc(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate ``nested_asc_contraction`` YAML config.

    Enforced:
    - ``enabled`` must be a bool.
    - ``num_planned_activity_iterations`` must be a positive int.
    - ``step_size`` must be a positive finite float.
    - ``timeout_minutes`` must be a positive finite number.
    - ``on_convergence_failure`` must be one of ``use_last``,
      ``mark_sample_failed``.
    - Unknown keys are rejected (early typo detection — otherwise
      a misspelled ``timout_minutes`` would silently mean no timeout).
    """
    if not isinstance(cfg, dict):
        raise SimulatorError(
            f"nested_asc_contraction must be a dict, got {type(cfg).__name__}"
        )
    unknown = set(cfg) - _NESTED_ASC_ALLOWED_KEYS
    if unknown:
        raise SimulatorError(
            f"nested_asc_contraction: unknown key(s) {sorted(unknown)}; "
            f"expected one of {sorted(_NESTED_ASC_ALLOWED_KEYS)}"
        )
    enabled = cfg.get("enabled", False)
    if not isinstance(enabled, bool):
        raise SimulatorError(
            f"nested_asc_contraction.enabled must be bool, got {enabled!r}"
        )
    # If the caller left enabled=False (or unset), everything else is
    # optional — the flags aren't forwarded and no validation catches
    # config drift. But if enabled=True, downstream fields MUST parse.
    if enabled:
        n_iter = cfg.get("num_planned_activity_iterations", 3)
        if not isinstance(n_iter, int) or isinstance(n_iter, bool) or n_iter <= 0:
            raise SimulatorError(
                f"nested_asc_contraction.num_planned_activity_iterations "
                f"must be a positive int, got {n_iter!r}"
            )
        step = cfg.get("step_size", 2.0)
        if not isinstance(step, (int, float)) or isinstance(step, bool):
            raise SimulatorError(
                f"nested_asc_contraction.step_size must be numeric, got {step!r}"
            )
        step_f = float(step)
        import math as _math
        if not _math.isfinite(step_f) or step_f <= 0:
            raise SimulatorError(
                f"nested_asc_contraction.step_size must be positive finite, got {step!r}"
            )
        timeout = cfg.get("timeout_minutes", 30)
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool):
            raise SimulatorError(
                f"nested_asc_contraction.timeout_minutes must be numeric, got {timeout!r}"
            )
        timeout_f = float(timeout)
        if not _math.isfinite(timeout_f) or timeout_f <= 0:
            raise SimulatorError(
                f"nested_asc_contraction.timeout_minutes must be positive finite, "
                f"got {timeout!r}"
            )
        on_failure = cfg.get("on_convergence_failure", "use_last")
        if on_failure not in _NESTED_ASC_ON_FAILURE:
            raise SimulatorError(
                f"nested_asc_contraction.on_convergence_failure must be one of "
                f"{list(_NESTED_ASC_ON_FAILURE)}, got {on_failure!r}"
            )
    # Return a normalized shallow copy so the caller's dict can't be
    # mutated from under us.
    return dict(cfg)


