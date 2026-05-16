from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from polarisopt.parameters import Parameter, ParameterSpace
from polarisopt.samples.sample import Sample
from polarisopt.simulator import PolarisSimulator
from polarisopt.simulator.base import SimulatorError, make_simulator


def _build_fake_model(root: Path, scenario_name: str = "scenario_abm.json") -> Path:
    """Construct a POLARIS-shaped model directory with two parameter JSONs and a scenario."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "DestinationChoice.json").write_text(
        json.dumps({"Hard constraints": {"trip_threshold": 0.0, "min_distance": 0.0}})
    )
    (root / "ActivityChoice.json").write_text(
        json.dumps({"Weights": {"alpha": 0.0}})
    )
    (root / scenario_name).write_text(
        json.dumps(
            {
                "Output controls": {"output_directory": "out"},
                "General simulation controls": {"database_name": "TestModel"},
            }
        )
    )
    return root


@pytest.fixture
def space() -> ParameterSpace:
    return ParameterSpace.from_iterable(
        [
            Parameter("trip_threshold", "DestinationChoice.json", 0.0, 1.0),
            Parameter("alpha", "ActivityChoice.json", -1.0, 1.0),
        ]
    )


def _make_sim(tmp_path: Path) -> PolarisSimulator:
    model = _build_fake_model(tmp_path / "src_model")
    return PolarisSimulator(
        binary="/usr/bin/echo",  # any executable for tests
        model_source=str(model),
        scenario_file="scenario_abm.json",
        output_db_filename="TestModel-Result.h5",
        num_threads="4",
    )


def test_prepare_stages_model_and_injects(tmp_path: Path, space: ParameterSpace) -> None:
    sim = _make_sim(tmp_path)
    sample = Sample(id=1, phase="test", inputs=np.array([0.42, 0.7]))
    workspace = tmp_path / "experiments" / "sim-1"
    spec = sim.prepare(sample, space, workspace)
    # files staged
    dest_json = workspace / "DestinationChoice.json"
    assert dest_json.exists()
    payload = json.loads(dest_json.read_text())
    assert payload["Hard constraints"]["trip_threshold"] == pytest.approx(0.42)
    activity = json.loads((workspace / "ActivityChoice.json").read_text())
    assert activity["Weights"]["alpha"] == pytest.approx(0.7)
    # job spec sane
    assert spec.cwd == workspace
    assert "scenario_abm.json" in spec.command
    assert spec.env["POLARIS_NUM_THREADS"] == "4"


def test_prepare_rejects_wrong_input_shape(tmp_path: Path, space: ParameterSpace) -> None:
    sim = _make_sim(tmp_path)
    sample = Sample(id=1, inputs=np.array([0.5]))  # too few
    with pytest.raises(SimulatorError):
        sim.prepare(sample, space, tmp_path / "x")


def test_prepare_rejects_missing_model_source(tmp_path: Path, space: ParameterSpace) -> None:
    sim = PolarisSimulator(
        binary="/usr/bin/echo",
        model_source=str(tmp_path / "does_not_exist"),
        scenario_file="scenario_abm.json",
        output_db_filename="R.h5",
    )
    sample = Sample(id=1, inputs=np.array([0.5, 0.5]))
    with pytest.raises(SimulatorError, match="model_source"):
        sim.prepare(sample, space, tmp_path / "x")


def test_collect_output_returns_result_paths(tmp_path: Path, space: ParameterSpace) -> None:
    sim = _make_sim(tmp_path)
    sample = Sample(id=2, inputs=np.array([0.3, 0.5]))
    workspace = tmp_path / "experiments" / "sim-2"
    sim.prepare(sample, space, workspace)
    # simulate POLARIS writing its result
    (workspace / "out").mkdir(parents=True, exist_ok=True)
    result_path = workspace / "out" / "TestModel-Result.h5"
    with h5py.File(result_path, "w") as f:
        g = f.create_group("link_moe")
        g.create_dataset("link_travel_time", data=np.ones((3, 2)))
        g.create_dataset("link_in_volume", data=np.ones((3, 2)))
    sample.folder = workspace
    out = sim.collect_output(sample)
    assert out["result_path"] == str(result_path)


def test_collect_output_missing_result_raises(tmp_path: Path, space: ParameterSpace) -> None:
    sim = _make_sim(tmp_path)
    sample = Sample(id=3, inputs=np.array([0.0, 0.0]))
    workspace = tmp_path / "experiments" / "sim-3"
    sim.prepare(sample, space, workspace)
    sample.folder = workspace
    # do NOT write result file
    with pytest.raises(SimulatorError, match="result file missing"):
        sim.collect_output(sample)


def test_make_simulator_polaris(tmp_path: Path) -> None:
    model = _build_fake_model(tmp_path / "m")
    sim = make_simulator(
        {
            "type": "polaris",
            "options": {
                "binary": "/usr/bin/echo",
                "model_source": str(model),
                "scenario_file": "scenario_abm.json",
                "output_db_filename": "R.h5",
            },
        }
    )
    assert isinstance(sim, PolarisSimulator)
