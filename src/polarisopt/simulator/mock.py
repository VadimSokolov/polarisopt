"""MockSimulator — runs a benchmark function in a slave subprocess via the Runner.

Exercises the full master/slave loop without POLARIS. Picks one of the
:data:`~polarisopt.simulator.benchmarks.BENCHMARKS` by name.
"""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path
from typing import Any

from polarisopt.parameters import ParameterSpace
from polarisopt.runners.base import JobSpec
from polarisopt.samples.sample import Sample
from polarisopt.simulator.base import Simulator, SimulatorError, simulator_registry
from polarisopt.simulator.benchmarks import BENCHMARKS


@simulator_registry.register("mock")
class MockSimulator(Simulator):
    """Evaluate a benchmark function in a slave subprocess.

    Useful for tutorials, smoke tests, and CI runs that don't require a
    POLARIS install. Picks one of the named SFU benchmark functions in
    :data:`polarisopt.simulator.benchmarks.BENCHMARKS`. Each sample is
    evaluated by forking ``python -m polarisopt.simulator._mock_runner``
    against an ``inputs.json``, exactly mirroring the master/slave
    pattern used for real POLARIS runs.

    Parameters
    ----------
    function : str
        One of ``"branin"``, ``"rosenbrock"``, ``"hartmann6"``, ``"quadratic"``.
    python_executable : str or None
        Python interpreter to use in the slave subprocess. Defaults to
        :data:`sys.executable`.

    Raises
    ------
    SimulatorError
        If ``function`` isn't a known benchmark.

    Examples
    --------
    >>> from polarisopt.simulator import MockSimulator
    >>> sim = MockSimulator(function="branin")
    >>> sim.function
    'branin'
    """

    INPUT_FILE = "inputs.json"
    OUTPUT_FILE = "outputs.json"

    def __init__(self, function: str, *, python_executable: str | None = None) -> None:
        if function not in BENCHMARKS:
            raise SimulatorError(
                f"unknown benchmark function {function!r}; available: {sorted(BENCHMARKS)}"
            )
        self.function = function
        self.python_executable = python_executable or sys.executable

    def prepare(self, sample: Sample, space: ParameterSpace, workspace: Path) -> JobSpec:
        if sample.inputs.shape != (space.ndim,):
            raise SimulatorError(
                f"sample.inputs has shape {sample.inputs.shape}, expected ({space.ndim},)"
            )
        workspace.mkdir(parents=True, exist_ok=True)
        input_path = workspace / self.INPUT_FILE
        output_path = workspace / self.OUTPUT_FILE
        input_path.write_text(json.dumps({"inputs": sample.inputs.tolist()}))

        cmd = " ".join(
            [
                shlex.quote(self.python_executable),
                "-m",
                "polarisopt.simulator._mock_runner",
                shlex.quote(self.function),
                shlex.quote(str(input_path)),
                shlex.quote(str(output_path)),
            ]
        )
        return JobSpec(
            name=f"mock-{self.function}-sample-{sample.id or 'unsaved'}",
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
