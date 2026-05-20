"""Tests for ConstantMetric — artifact-only studies."""

from __future__ import annotations

import numpy as np
import pytest

from polarisopt.metrics import ConstantMetric, make_metric, metric_registry


def test_default_returns_single_zero() -> None:
    m = ConstantMetric()
    assert m.n_objectives == 1
    np.testing.assert_array_equal(m.compute({}), [0.0])


def test_scalar_value() -> None:
    m = ConstantMetric(value=42.0)
    np.testing.assert_array_equal(m.compute({"anything": "ignored"}), [42.0])


def test_multi_objective_value() -> None:
    m = ConstantMetric(value=[1.0, 2.0, 3.0])
    assert m.n_objectives == 3
    np.testing.assert_array_equal(m.compute({}), [1.0, 2.0, 3.0])


def test_empty_list_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ConstantMetric(value=[])


def test_compute_returns_fresh_copy() -> None:
    """Calling compute() twice should not return the same mutable array."""
    m = ConstantMetric(value=[1.0, 2.0])
    a = m.compute({})
    a[0] = 999.0
    b = m.compute({})
    assert b[0] == 1.0


def test_registered_under_both_names() -> None:
    assert "constant" in metric_registry
    assert "null_metric" in metric_registry
    assert metric_registry.get("constant") is metric_registry.get("null_metric")


def test_factory_round_trip() -> None:
    m = make_metric({"type": "constant", "options": {"value": [0.0, 0.0]}})
    assert isinstance(m, ConstantMetric)
    assert m.n_objectives == 2

    # null_metric alias also works
    m2 = make_metric({"type": "null_metric", "options": {}})
    assert isinstance(m2, ConstantMetric)
