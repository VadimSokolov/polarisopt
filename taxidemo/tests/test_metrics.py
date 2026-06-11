"""Tests for the output_match calibration metric."""

import json

import pytest

from polarisopt.metrics.base import MetricError, metric_registry
from polarisopt.utils.plugins import load_external_plugins

from taxidemo.metrics import OutputMatchMetric


def test_registered_via_entry_point():
    load_external_plugins()
    assert metric_registry.get("output_match") is OutputMatchMetric


def test_zero_at_exact_match():
    m = OutputMatchMetric(targets={"journeys_completed": 300.0, "pick_up_time": 90.0})
    assert m.compute({"journeys_completed": 300, "pick_up_time": 90.0}) == pytest.approx([0.0])


def test_mean_squared_relative_error():
    m = OutputMatchMetric(targets={"a": 100.0, "b": 10.0})
    # ((150-100)/100)^2 = 0.25 ; ((15-10)/10)^2 = 0.25 ; mean = 0.25
    assert m.compute({"a": 150.0, "b": 15.0}) == pytest.approx([0.25])


def test_custom_scales_and_small_targets():
    # |target| < 1 falls back to scale 1 unless overridden.
    m = OutputMatchMetric(targets={"a": 0.5}, scales={"a": 0.5})
    assert m.compute({"a": 1.0}) == pytest.approx([1.0])
    m_default = OutputMatchMetric(targets={"a": 0.5})
    assert m_default.compute({"a": 1.0}) == pytest.approx([0.25])


def test_targets_file(tmp_path):
    path = tmp_path / "targets.json"
    path.write_text(json.dumps({"missed": 80.0}))
    m = OutputMatchMetric(targets_file=str(path))
    assert m.targets == {"missed": 80.0}
    assert m.n_objectives == 1


def test_missing_output_key():
    m = OutputMatchMetric(targets={"nope": 1.0})
    with pytest.raises(MetricError, match="not in simulator output"):
        m.compute({"profit": 1.0})


def test_validation_errors(tmp_path):
    with pytest.raises(MetricError, match="exactly one"):
        OutputMatchMetric()
    with pytest.raises(MetricError, match="exactly one"):
        OutputMatchMetric(targets={"a": 1.0}, targets_file="x.json")
    with pytest.raises(MetricError, match="not found"):
        OutputMatchMetric(targets_file=str(tmp_path / "missing.json"))
    with pytest.raises(MetricError, match="non-empty"):
        OutputMatchMetric(targets={})
    with pytest.raises(MetricError, match="positive"):
        OutputMatchMetric(targets={"a": 1.0}, scales={"a": 0.0})
    with pytest.raises(MetricError, match="not finite"):
        OutputMatchMetric(targets={"a": 1.0}).compute({"a": float("nan")})
