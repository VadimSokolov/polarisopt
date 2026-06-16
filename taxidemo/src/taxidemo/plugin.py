"""polarisopt simulator plugin: registers the ``taxi`` simulator type.

Discovered automatically through the ``polarisopt.simulators`` entry point
declared in this package's ``pyproject.toml`` — ``pip install`` the package
and ``simulator: {type: taxi}`` becomes available in any study YAML.

Example study snippet::

    simulator:
      type: taxi
      options:
        n_repeats: 3          # average this many seeded runs per sample
        base_seed: 42
        max_steps: 1000       # any TaxiParams field can be pinned here...
        grid_size: 5
        journey_frequency: 50

    parameters:
      inline:                  # ...and the rest searched over
        - { name: taxi_count,     file: inputs.json, min: 1, max: 100, type: int }
        - { name: base_fare,      file: inputs.json, min: 0, max: 15 }
        - { name: cost_per_tile,  file: inputs.json, min: 1, max: 20 }
        - { name: max_multiplier, file: inputs.json, min: 1, max: 3 }

Each sample is evaluated in a slave subprocess
(``python -m taxidemo.runner``), mirroring the master/slave pattern used
for real POLARIS runs, so the same study works under both the local and
the Slurm runner.
"""

from __future__ import annotations

import dataclasses
import json
import shlex
import sys
from pathlib import Path
from typing import Any

from polarisopt.parameters import ParameterSpace
from polarisopt.runners.base import JobSpec
from polarisopt.samples.sample import Sample
from polarisopt.simulator.base import Simulator, SimulatorError, simulator_registry

from taxidemo.simulator import TaxiParams

_PARAM_NAMES = {f.name for f in dataclasses.fields(TaxiParams)}


@simulator_registry.register("taxi")
class TaxiSimulator(Simulator):
    """Evaluate the emukit taxi simulator for one sample.

    Parameters
    ----------
    n_repeats : int
        Number of seeded simulator runs to average per sample (the
        simulator is stochastic; averaging tames the noise the surrogate
        has to model).
    base_seed : int
        Per-sample seeds are derived deterministically from this and the
        sample id, so a retried sample reproduces its result exactly.
    python_executable : str or None
        Interpreter for the slave subprocess. Defaults to
        :data:`sys.executable`; on a cluster, point this at the Python of
        an environment that has ``taxidemo`` installed (or activate that
        environment via the runner's ``setup_commands``).
    **fixed
        Any :class:`~taxidemo.simulator.TaxiParams` field to pin for the
        whole study (e.g. ``grid_size: 5``, ``max_steps: 1000``). Fields
        not pinned here and not searched over via ``parameters`` keep the
        playground defaults.
    """

    INPUT_FILE = "inputs.json"
    OUTPUT_FILE = "outputs.json"

    def __init__(
        self,
        *,
        n_repeats: int = 1,
        base_seed: int = 42,
        python_executable: str | None = None,
        **fixed: float,
    ) -> None:
        unknown = set(fixed) - _PARAM_NAMES
        if unknown:
            raise SimulatorError(
                f"unknown taxi parameter(s) {sorted(unknown)}; available: {sorted(_PARAM_NAMES)}"
            )
        if n_repeats < 1:
            raise SimulatorError(f"n_repeats must be >= 1, got {n_repeats}")
        self.n_repeats = int(n_repeats)
        self.base_seed = int(base_seed)
        self.python_executable = python_executable or sys.executable
        self.fixed = dict(fixed)

    def prepare(self, sample: Sample, space: ParameterSpace, workspace: Path) -> JobSpec:
        unknown = set(space.names) - _PARAM_NAMES
        if unknown:
            raise SimulatorError(
                f"unknown taxi parameter(s) in search space {sorted(unknown)}; available: {sorted(_PARAM_NAMES)}"
            )
        overlap = set(space.names) & set(self.fixed)
        if overlap:
            raise SimulatorError(f"parameter(s) {sorted(overlap)} are both searched and pinned in simulator options")

        params = {**self.fixed, **space.values_dict(sample.inputs)}
        # One distinct, reproducible seed block per sample.
        seed = self.base_seed + self.n_repeats * (sample.id or 0)

        workspace.mkdir(parents=True, exist_ok=True)
        input_path = workspace / self.INPUT_FILE
        output_path = workspace / self.OUTPUT_FILE
        input_path.write_text(
            json.dumps({"params": params, "seed": seed, "n_repeats": self.n_repeats}, indent=2)
        )

        cmd = " ".join(
            [
                shlex.quote(self.python_executable),
                "-m",
                "taxidemo.runner",
                shlex.quote(str(input_path)),
                shlex.quote(str(output_path)),
            ]
        )
        return JobSpec(
            name=f"taxi-sample-{sample.id or 'unsaved'}",
            command=cmd,
            cwd=workspace,
            stdout=workspace / "stdout.log",
            stderr=workspace / "stderr.log",
        )

    def collect_output(self, sample: Sample) -> dict[str, Any]:
        if sample.folder is None:
            raise SimulatorError(f"sample {sample.id} has no folder set")
        output_path = sample.folder / self.OUTPUT_FILE
        if not output_path.exists():
            raise SimulatorError(f"output file missing: {output_path}")
        return dict(json.loads(output_path.read_text()))
