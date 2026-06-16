"""PolarisSimulator — runs the POLARIS C++ engine on a per-sample workspace.

Per-sample workflow
-------------------
``prepare``:
  1. Copy the model directory from ``model_source`` to the sample workspace
     via :class:`~polarisopt.transfer.base.Transfer` (default :class:`LocalTransfer`).
  2. Inject the sample's parameter values into POLARIS JSON files via
     :func:`polarisopt.parameters.injection.inject_values`.
  3. Build a :class:`JobSpec` that invokes the POLARIS binary on the staged
     scenario file with the configured thread count.

``collect_output`` returns metadata only (paths). Metrics open the
POLARIS output files (HDF5, SQLite) directly with the paths in the
returned dict.

The simulator never imports polarislib in core code paths — only optional
features (Globus transfer) flow through that dependency.
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
from polarisopt.simulator.base import Simulator, SimulatorError, simulator_registry
from polarisopt.transfer.base import Transfer, make_transfer
from polarisopt.utils.logging import get_logger

log = get_logger(__name__)


@simulator_registry.register("polaris")
class PolarisSimulator(Simulator):
    """Run the POLARIS C++ engine on a per-sample copy of the model.

    The master process never imports polarislib or executes POLARIS;
    this class only stages files and constructs a :class:`JobSpec`. The
    POLARIS binary runs on a slave compute node via the configured
    :class:`Runner` (typically ``slurm``).

    Parameters
    ----------
    binary : path
        Absolute path to the POLARIS executable or Apptainer ``.sif`` image.
    model_source : path
        Absolute path to the source model directory. Copied recursively
        into each sample's workspace by the configured :class:`Transfer`.
    scenario_file : str
        Name of the scenario JSON inside the model directory (e.g.
        ``"scenario_abm.json"``).
    output_db_filename : str
        Name of the result file written by POLARIS inside the output
        directory (e.g. ``"DFW-Result.h5"``).
    num_threads : str, optional
        Thread count passed to POLARIS. Default ``"1"``. Numeric strings
        match POLARIS's CLI convention.
    output_dir_key : tuple of (str, str) or None
        Where in the scenario JSON to find the relative output directory.
        Default ``("Output controls", "output_directory")``.
    transfer : dict or None
        Optional ``{"type": ..., "options": {...}}`` configuring a
        :class:`~polarisopt.transfer.base.Transfer` for staging. Defaults
        to ``{"type": "local"}``. Use ``{"type": "anl"}`` for Globus on
        VMS-backed paths (requires ``polarisopt[anl]``).
    pre_script : path or None
        Optional Python script to run **before** the POLARIS binary.
        Invoked as
        ``<pre_script_interpreter> <pre_script> <workspace>
        --<param-name-dashified>=<value> …`` with every parameter in
        the sample's input vector forwarded as a CLI flag (dashified:
        ``am_sigma`` → ``--am-sigma``). Booleans render as
        ``true``/``false``. Use for parameter-driven pre-processing
        that can't be expressed as JSON field injection — e.g.
        regenerating a demand DB, materializing skim tables,
        transforming model files in shape-dependent ways. The script
        runs in the staged workspace; failures abort the sample
        (``set -e`` is emitted in the rendered command). Default
        ``None`` (no pre-step).
    pre_script_interpreter : str or None
        Path to the Python interpreter for ``pre_script``. Default
        ``sys.executable``. Ignored when ``pre_script`` is ``None``.
    apptainer_binary : str, optional
        Runtime to use when ``binary`` ends in ``.sif``. Default
        ``"apptainer"``. Override to ``"singularity"`` on older clusters.
        Ignored when the binary is native (no ``.sif`` extension).
    singularity_binds : list of str, optional
        Extra ``-B`` bind specs forwarded to ``apptainer run`` when the
        binary is a SIF. The per-sample workspace and the SIF's parent
        directory are bound automatically; add extras here for paths
        the scenario JSON references outside those trees (e.g. shared
        skim caches, scratch dirs). Each entry is a host path (``/lcrc/...``)
        or a ``host:container`` mapping. Default: empty list.
    sif_entrypoint : str or None
        For the newer POLARIS SIF format where the runscript dispatches
        by executable name, this string is inserted as the first
        positional arg (e.g. ``"Integrated_Model"`` produces
        ``apptainer run ... polaris.sif Integrated_Model <scenario> <threads>``).
        Default ``None`` (historical bare invocation).

    Raises
    ------
    SimulatorError
        If ``output_dir_key`` isn't 2-element, or if the model source
        path is missing at ``prepare`` time, or if the result file is
        missing at ``collect_output`` time.

    Examples
    --------
    >>> from polarisopt.simulator import PolarisSimulator
    >>> sim = PolarisSimulator(                                         # doctest: +SKIP
    ...     binary="/lcrc/.../polaris_exe/Integrated_Model.sif",
    ...     model_source="/lcrc/.../DFW_2050_20251028",
    ...     scenario_file="scenario_abm.json",
    ...     output_db_filename="DFW-Result.h5",
    ...     num_threads="16",
    ... )
    """

    DEFAULT_OUTPUT_DIR_KEY: tuple[str, str] = ("Output controls", "output_directory")

    def __init__(
        self,
        *,
        binary: str,
        model_source: str,
        scenario_file: str,
        output_db_filename: str,
        num_threads: str = "1",
        num_iterations: int = 1,
        output_dir_key: tuple[str, str] | list[str] | None = None,
        transfer: dict[str, Any] | None = None,
        apptainer_binary: str = "apptainer",
        singularity_binds: list[str] | None = None,
        sif_entrypoint: str | None = None,
        pre_script: str | None = None,
        pre_script_interpreter: str | None = None,
    ) -> None:
        self.binary = Path(binary)
        self.model_source = Path(model_source)
        self.scenario_file = scenario_file
        self.output_db_filename = output_db_filename
        self.num_threads = str(num_threads)
        if num_iterations < 1:
            raise SimulatorError(
                f"num_iterations must be >= 1, got {num_iterations}"
            )
        self.num_iterations = int(num_iterations)
        self.output_dir_key: tuple[str, str] = (
            tuple(output_dir_key)  # type: ignore[assignment]
            if output_dir_key is not None
            else self.DEFAULT_OUTPUT_DIR_KEY
        )
        if len(self.output_dir_key) != 2:
            raise SimulatorError(
                f"output_dir_key must be (category, key); got {self.output_dir_key!r}"
            )
        self._transfer: Transfer = make_transfer(transfer)
        self.apptainer_binary: str = apptainer_binary
        self.singularity_binds: list[str] = list(singularity_binds or [])
        self.sif_entrypoint: str | None = sif_entrypoint
        self.pre_script: Path | None = Path(pre_script) if pre_script else None
        if self.pre_script is not None and not self.pre_script.exists():
            raise SimulatorError(f"pre_script not found: {self.pre_script}")
        # Resolved lazily — sys.executable inside __init__ might differ
        # from what's available at runtime in some envs; keep the string.
        self.pre_script_interpreter: str = pre_script_interpreter or sys.executable

    # ----- staging -----

    def prepare(self, sample: Sample, space: ParameterSpace, workspace: Path) -> JobSpec:
        if sample.inputs.shape != (space.ndim,):
            raise SimulatorError(
                f"sample.inputs shape {sample.inputs.shape} != space.ndim={space.ndim}"
            )
        if not self.model_source.exists():
            raise SimulatorError(f"model_source does not exist: {self.model_source}")
        workspace.mkdir(parents=True, exist_ok=True)
        log.info("PolarisSimulator: staging model into %s", workspace)
        self._transfer.copy(self.model_source, workspace, recursive=True)
        missing = inject_values(sample.inputs, space, workspace)
        if missing:
            log.warning(
                "PolarisSimulator: parameters not found in JSONs: %s", missing
            )

        scenario_path = workspace / self.scenario_file
        if not scenario_path.exists():
            raise SimulatorError(
                f"scenario file missing after staging: {scenario_path}"
            )

        single_invocation = self._build_invocation(scenario_path, workspace)
        pre_invocation = self._build_pre_script_invocation(sample, space, workspace)
        if self.num_iterations == 1 and pre_invocation is None:
            command = single_invocation
        else:
            # Bash sequence. ``set -e`` ensures we bail on pre_script
            # failure before the binary runs (otherwise stale demand DB
            # would silently feed POLARIS). The for-loop wraps multi-
            # iteration native-POLARIS runs; pre_script only runs once.
            lines = ["set -e"]
            if pre_invocation is not None:
                lines.append(pre_invocation)
            if self.num_iterations == 1:
                lines.append(single_invocation)
            else:
                # POLARIS handles iteration numbering itself — the second
                # run reads the first run's <output>_iteration_1 outputs
                # as warm-starts and writes <output>_iteration_2.
                lines.append(f"for i in $(seq 1 {self.num_iterations}); do")
                lines.append(
                    f'    echo "[polarisopt] iteration $i of {self.num_iterations}"'
                )
                lines.append(f"    {single_invocation}")
                lines.append("done")
            command = "\n".join(lines) + "\n"
        return JobSpec(
            name=f"polaris-sample-{sample.id or 'unsaved'}",
            command=command,
            cwd=workspace,
            stdout=workspace / "polaris.stdout.log",
            stderr=workspace / "polaris.stderr.log",
            env={"POLARIS_NUM_THREADS": self.num_threads},
        )

    # ----- command construction -----

    def _is_sif(self) -> bool:
        """True when ``binary`` is a SIF (Apptainer/Singularity image)."""
        return self.binary.suffix.lower() == ".sif"

    def _auto_bind_paths(self, workspace: Path) -> list[str]:
        """Bind defaults: per-sample workspace + the SIF's parent dir.

        Workspace bind is the critical one — every staged file (scenario
        JSON, model SQLite copies, output paths) lives under it, and the
        Apptainer default mount namespace doesn't include ``/lcrc/`` on
        ANL clusters. SIF's parent is bound so things like sibling
        ``.params`` files alongside the image are visible if POLARIS
        expects them.
        """
        return [str(workspace.resolve()), str(self.binary.parent.resolve())]

    def _build_pre_script_invocation(
        self, sample: Sample, space: ParameterSpace, workspace: Path
    ) -> str | None:
        """Render the pre-binary Python invocation, or None if disabled.

        Forwards every sample parameter as
        ``--<dashified-name>=<value>``, matching the convention used by
        :class:`PolarisConvergenceSimulator` for ``runner_options``.
        Booleans render as ``true``/``false``; everything else falls
        through to ``str()``. Values are :func:`shlex.quote`-d so
        spaces or shell metacharacters in numeric formatting can't
        break the command.
        """
        if self.pre_script is None:
            return None
        argv = [
            shlex.quote(self.pre_script_interpreter),
            shlex.quote(str(self.pre_script)),
            shlex.quote(str(workspace)),
        ]
        for name, value in zip(space.names, sample.inputs, strict=True):
            flag = "--" + name.replace("_", "-")
            argv.append(f"{flag}={shlex.quote(_arg_value(value))}")
        return " ".join(argv)

    def _build_invocation(self, scenario_path: Path, workspace: Path) -> str:
        """Render the full single-invocation command.

        Native binary: ``<binary> <scenario> <threads>``.

        SIF binary: ``<apptainer> run -B <auto-binds> [-B <user-binds>]
        <binary> [<sif_entrypoint>] <scenario> <threads>``.
        """
        if not self._is_sif():
            return " ".join([
                shlex.quote(str(self.binary)),
                shlex.quote(str(scenario_path)),
                shlex.quote(self.num_threads),
            ])
        bind_specs: list[str] = []
        seen: set[str] = set()
        for spec in [*self._auto_bind_paths(workspace), *self.singularity_binds]:
            if spec not in seen:
                seen.add(spec)
                bind_specs.append(spec)
        bind_args: list[str] = []
        for spec in bind_specs:
            bind_args.extend(("-B", shlex.quote(spec)))
        parts = [
            shlex.quote(self.apptainer_binary),
            "run",
            *bind_args,
            shlex.quote(str(self.binary)),
        ]
        if self.sif_entrypoint:
            parts.append(shlex.quote(self.sif_entrypoint))
        parts.append(shlex.quote(str(scenario_path)))
        parts.append(shlex.quote(self.num_threads))
        return " ".join(parts)

    # ----- collection -----

    def collect_output(self, sample: Sample) -> dict[str, Any]:
        if sample.folder is None:
            raise SimulatorError(f"sample {sample.id} has no folder set")
        scenario_path = sample.folder / self.scenario_file
        if not scenario_path.exists():
            raise SimulatorError(f"scenario file missing in {sample.folder}")

        import json

        scenario = json.loads(scenario_path.read_text())
        category, key = self.output_dir_key
        try:
            output_dirname = scenario[category][key]
        except KeyError as exc:
            raise SimulatorError(
                f"output dir key {self.output_dir_key!r} not in scenario JSON"
            ) from exc

        output_dir = self._resolve_output_dir(sample.folder, output_dirname)
        result_path = output_dir / self.output_db_filename
        if not result_path.exists():
            raise SimulatorError(f"POLARIS result file missing: {result_path}")
        return {
            "result_path": str(result_path),
            "output_dir": str(output_dir),
            "scenario_path": str(scenario_path),
            "iteration": _iteration_of(output_dir),
        }

    def _resolve_output_dir(self, workspace: Path, output_dirname: str) -> Path:
        """Pick the right output directory.

        For ``num_iterations == 1`` POLARIS may use either
        ``<output_dirname>`` or ``<output_dirname>_iteration_1`` (deployment-
        dependent). For ``num_iterations > 1``, the highest-numbered
        ``_iteration_N`` directory is the final one. We look for both
        patterns and pick the most recent existing one.
        """
        candidates: list[tuple[int, Path]] = []
        base = workspace / output_dirname
        if base.exists():
            candidates.append((0, base))
        for it_dir in workspace.glob(f"{output_dirname}_iteration_*"):
            if not it_dir.is_dir():
                continue
            try:
                n = int(it_dir.name.rsplit("_iteration_", 1)[-1])
            except ValueError:
                continue
            candidates.append((n, it_dir))
        if not candidates:
            raise SimulatorError(
                f"no output directory found for {output_dirname!r} under {workspace}"
            )
        # Highest iteration wins; ties fall back to the non-iteration base.
        _, best = max(candidates, key=lambda kv: kv[0])
        return best


def _iteration_of(output_dir: Path) -> int | None:
    """Extract the iteration number from a path like ``foo_iteration_3``."""
    name = output_dir.name
    if "_iteration_" not in name:
        return None
    try:
        return int(name.rsplit("_iteration_", 1)[-1])
    except ValueError:
        return None


def _arg_value(value: Any) -> str:
    """Render a Python value as a single CLI arg token.

    Booleans become ``"true"`` / ``"false"`` (POLARIS / polarislib
    convention); everything else falls through to ``str()``. Used both
    by ``PolarisSimulator.pre_script`` arg forwarding and by
    ``PolarisConvergenceSimulator`` ``runner_options`` forwarding.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
