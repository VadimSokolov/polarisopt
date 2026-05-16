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
        output_dir_key: tuple[str, str] | list[str] | None = None,
        transfer: dict[str, Any] | None = None,
    ) -> None:
        self.binary = Path(binary)
        self.model_source = Path(model_source)
        self.scenario_file = scenario_file
        self.output_db_filename = output_db_filename
        self.num_threads = str(num_threads)
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

        command = " ".join(
            [
                shlex.quote(str(self.binary)),
                shlex.quote(str(scenario_path)),
                self.num_threads,
            ]
        )
        return JobSpec(
            name=f"polaris-sample-{sample.id or 'unsaved'}",
            command=command,
            cwd=workspace,
            stdout=workspace / "polaris.stdout.log",
            stderr=workspace / "polaris.stderr.log",
            env={"POLARIS_NUM_THREADS": self.num_threads},
        )

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

        output_dir = sample.folder / output_dirname
        result_path = output_dir / self.output_db_filename
        if not result_path.exists():
            raise SimulatorError(f"POLARIS result file missing: {result_path}")
        return {
            "result_path": str(result_path),
            "output_dir": str(output_dir),
            "scenario_path": str(scenario_path),
        }
