"""Tests for the multi-task GP surrogate."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from polarisopt.surrogates import make_surrogate
from polarisopt.surrogates.base import SurrogateError
from polarisopt.surrogates.mtgp import MultiTaskGPSurrogate


def test_mtgp_single_obj_works(tmp_path) -> None:
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(8, 2))
    Y = ((X - 0.3) ** 2).sum(axis=1, keepdims=True)
    s = MultiTaskGPSurrogate()
    s.fit(X, Y)
    assert s.n_objectives == 1
    mean, var = s.predict(X[:2])
    assert mean.shape == (2, 1) and var.shape == (2, 1)


def test_mtgp_multi_obj_correlated_outputs() -> None:
    """When outputs are highly correlated, the multi-task GP should fit fine."""
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(12, 3))
    base = (X ** 2).sum(axis=1)
    Y = np.column_stack([base, 2 * base + 0.1 * rng.normal(size=12)])
    s = MultiTaskGPSurrogate(rank=1)
    s.fit(X, Y)
    assert s.n_objectives == 2
    mean, var = s.predict(X[:3])
    assert mean.shape == (3, 2)
    assert var.shape == (3, 2)
    assert (var > 0).all()


def test_mtgp_rejects_unfitted_predict() -> None:
    s = MultiTaskGPSurrogate()
    with pytest.raises(SurrogateError):
        s.predict(np.zeros((1, 2)))


def test_mtgp_rejects_too_few_points() -> None:
    s = MultiTaskGPSurrogate()
    with pytest.raises(SurrogateError):
        s.fit(np.zeros((1, 2)), np.zeros((1, 2)))


def test_mtgp_rejects_non_finite() -> None:
    s = MultiTaskGPSurrogate()
    X = np.array([[0.1, 0.2], [0.3, np.nan]])
    Y = np.array([[1.0, 2.0], [2.0, 3.0]])
    with pytest.raises(SurrogateError):
        s.fit(X, Y)


def test_mtgp_factory() -> None:
    s = make_surrogate({"type": "mtgp", "options": {"rank": 1}})
    assert isinstance(s, MultiTaskGPSurrogate)
