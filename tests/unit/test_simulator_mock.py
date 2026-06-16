from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from polarisopt.parameters import Parameter, ParameterSpace
from polarisopt.runners.local import LocalRunner
from polarisopt.samples.sample import Sample
from polarisopt.simulator import BENCHMARKS, MockSimulator, SimulatorError, make_simulator


@pytest.fixture
def space() -> ParameterSpace:
    return ParameterSpace.from_iterable(
        [Parameter("x1", "a.json", -5.0, 10.0), Parameter("x2", "a.json", 0.0, 15.0)]
    )


def _wait(runner: LocalRunner, job, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        runner.status(job)
        if job.status.is_terminal():
            return
        time.sleep(0.05)
    pytest.fail(f"Job did not terminate; last status={job.status}")


def test_mock_runner_module_directly(tmp_path: Path) -> None:
    inp = tmp_path / "in.json"
    out = tmp_path / "out.json"
    inp.write_text(json.dumps({"inputs": [0.0, 0.0]}))
    rc = subprocess.run(
        [sys.executable, "-m", "polarisopt.simulator._mock_runner", "branin", str(inp), str(out)],
        check=False,
    ).returncode
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["function"] == "branin"
    # branin at (0, 0) is a known value
    assert payload["value"] == pytest.approx(BENCHMARKS["branin"](np.array([0.0, 0.0])))


def test_mock_simulator_unknown_function() -> None:
    with pytest.raises(SimulatorError):
        MockSimulator(function="not_a_function")


def test_mock_simulator_prepare_creates_input(tmp_path: Path, space: ParameterSpace) -> None:
    sim = MockSimulator(function="branin")
    sample = Sample(id=1, phase="t", inputs=np.array([1.0, 2.0]))
    spec = sim.prepare(sample, space, tmp_path / "sim-1")
    assert (tmp_path / "sim-1" / MockSimulator.INPUT_FILE).exists()
    assert "polarisopt.simulator._mock_runner" in spec.command
    assert "branin" in spec.command


def test_mock_simulator_end_to_end_via_local_runner(tmp_path: Path, space: ParameterSpace) -> None:
    sim = MockSimulator(function="branin")
    runner = LocalRunner()
    sample = Sample(id=7, phase="t", inputs=np.array([np.pi, 2.275]))  # branin global min
    folder = tmp_path / "sim-7"
    spec = sim.prepare(sample, space, folder)
    sample.folder = folder
    job = runner.submit(spec)
    _wait(runner, job)
    assert job.status.value == "finished"
    output = sim.collect_output(sample)
    # Branin global min ≈ 0.397887
    assert output["value"] == pytest.approx(0.397887, abs=1e-4)


def test_mock_simulator_input_shape_mismatch(tmp_path: Path, space: ParameterSpace) -> None:
    sim = MockSimulator(function="branin")
    sample = Sample(id=1, inputs=np.array([1.0]))  # wrong shape
    with pytest.raises(SimulatorError):
        sim.prepare(sample, space, tmp_path)


def test_make_simulator_factory(space: ParameterSpace) -> None:
    sim = make_simulator({"type": "mock", "options": {"function": "quadratic"}})
    assert isinstance(sim, MockSimulator)
    assert sim.function == "quadratic"
