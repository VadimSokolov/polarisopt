from __future__ import annotations

import math

import numpy as np
import pytest

from polarisopt.metrics import IdentityMetric, MetricError, make_metric, metric_registry


def test_identity_single_objective() -> None:
    m = IdentityMetric(keys="value")
    assert m.n_objectives == 1
    np.testing.assert_array_equal(m.compute({"value": 3.14}), [3.14])


def test_identity_multi_objective() -> None:
    m = IdentityMetric(keys=["a", "b"])
    assert m.n_objectives == 2
    np.testing.assert_array_equal(m.compute({"a": 1.0, "b": 2.0}), [1.0, 2.0])


def test_identity_missing_key_raises() -> None:
    m = IdentityMetric(keys="value")
    with pytest.raises(MetricError, match="not in simulator output"):
        m.compute({"different": 1})


def test_identity_non_finite_raises() -> None:
    m = IdentityMetric(keys="value")
    with pytest.raises(MetricError, match="not finite"):
        m.compute({"value": math.nan})


def test_identity_non_numeric_raises() -> None:
    m = IdentityMetric(keys="value")
    with pytest.raises(MetricError, match="not numeric"):
        m.compute({"value": "hello"})


def test_identity_empty_keys_rejected() -> None:
    with pytest.raises(ValueError):
        IdentityMetric(keys=[])


def test_make_metric_factory() -> None:
    m = make_metric({"type": "identity", "options": {"keys": "v"}})
    assert isinstance(m, IdentityMetric)


def test_metric_registry_has_identity() -> None:
    assert "identity" in metric_registry
