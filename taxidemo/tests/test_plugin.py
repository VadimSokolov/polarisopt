"""Tests for the polarisopt simulator plugin and the slave runner."""

import json
import subprocess
import sys

import numpy as np
import pytest

from polarisopt.parameters.space import parameter_space_from_records
from polarisopt.samples.sample import Sample
from polarisopt.simulator.base import SimulatorError, simulator_registry
from polarisopt.utils.plugins import load_external_plugins

from taxidemo.plugin import TaxiSimulator
from taxidemo.runner import evaluate
from taxidemo.simulator import OUTPUT_KEYS


def _space():
    return parameter_space_from_records(
        [
            {"name": "taxi_count", "file": "inputs.json", "min": 1, "max": 100, "type": "int"},
            {"name": "base_fare", "file": "inputs.json", "min": 0, "max": 15},
        ]
    )


def test_registered_via_entry_point():
    load_external_plugins()
    assert simulator_registry.get("taxi") is TaxiSimulator


def test_rejects_unknown_fixed_parameter():
    with pytest.raises(SimulatorError, match="unknown taxi parameter"):
        TaxiSimulator(grid_sze=5)


def test_rejects_searched_and_pinned_overlap(tmp_path):
    sim = TaxiSimulator(taxi_count=10)
    sample = Sample(id=1, inputs=np.array([20.0, 5.0]))
    with pytest.raises(SimulatorError, match="both searched and pinned"):
        sim.prepare(sample, _space(), tmp_path)


def test_prepare_and_collect_roundtrip(tmp_path):
    sim = TaxiSimulator(n_repeats=2, base_seed=7, max_steps=200)
    sample = Sample(id=3, inputs=np.array([15.0, 6.0]), folder=tmp_path)
    spec = sim.prepare(sample, _space(), tmp_path)

    staged = json.loads((tmp_path / "inputs.json").read_text())
    assert staged["params"] == {"max_steps": 200, "taxi_count": 15, "base_fare": 6.0}
    assert staged["seed"] == 7 + 2 * 3
    assert staged["n_repeats"] == 2

    # Execute the JobSpec command exactly as a runner would.
    result = subprocess.run(spec.command, shell=True, cwd=spec.cwd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    out = sim.collect_output(sample)
    assert set(OUTPUT_KEYS) <= set(out)
    assert len(out["repeats"]) == 2


def test_evaluate_averages_repeats():
    one = evaluate({"max_steps": 200}, seed=5, n_repeats=1)
    two = evaluate({"max_steps": 200}, seed=5, n_repeats=2)
    assert one["repeats"][0] == two["repeats"][0]
    for key in OUTPUT_KEYS:
        expected = (two["repeats"][0][key] + two["repeats"][1][key]) / 2
        assert two[key] == pytest.approx(expected)


def test_slave_uses_default_python(tmp_path):
    sim = TaxiSimulator()
    sample = Sample(id=1, inputs=np.array([10.0, 5.0]))
    spec = sim.prepare(sample, _space(), tmp_path)
    assert sys.executable in spec.command
    assert "taxidemo.runner" in spec.command
