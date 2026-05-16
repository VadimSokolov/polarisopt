from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from polarisopt.acquisition import make_acquisition
from polarisopt.acquisition.base import AcquisitionError
from polarisopt.acquisition.ei import EIAcquisition
from polarisopt.acquisition.qehvi import QEHVIAcquisition
from polarisopt.acquisition.qei import QEIAcquisition
from polarisopt.parameters import Parameter, ParameterSpace
from polarisopt.surrogates.gp import GPSurrogate


def _space2d() -> ParameterSpace:
    return ParameterSpace.from_iterable(
        [Parameter("x1", "a.json", 0.0, 1.0), Parameter("x2", "a.json", 0.0, 1.0)]
    )


def _train_gp(m: int = 1, n: int = 10, seed: int = 0) -> tuple[GPSurrogate, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(n, 2))
    if m == 1:
        Y = np.sum((X - 0.3) ** 2, axis=1, keepdims=True)
    else:
        Y = np.stack([np.sum((X - 0.3) ** 2, axis=1), np.sum((X - 0.7) ** 2, axis=1)], axis=1)
    gp = GPSurrogate()
    gp.fit(X, Y)
    return gp, Y


def test_ei_proposes_in_bounds() -> None:
    gp, Y = _train_gp(m=1)
    acq = EIAcquisition(gp, num_restarts=2, raw_samples=32)
    next_x = acq.optimize(_space2d(), q=1, observed_Y=Y, rng=np.random.default_rng(1))
    assert next_x.shape == (1, 2)
    assert (next_x >= 0).all() and (next_x <= 1).all()


def test_ei_rejects_batch() -> None:
    gp, Y = _train_gp(m=1)
    acq = EIAcquisition(gp, num_restarts=2, raw_samples=32)
    with pytest.raises(AcquisitionError, match="q=1"):
        acq.optimize(_space2d(), q=4, observed_Y=Y, rng=np.random.default_rng(1))


def test_qei_batch_returns_q_points() -> None:
    gp, Y = _train_gp(m=1, n=8)
    acq = QEIAcquisition(gp, mc_samples=32, num_restarts=2, raw_samples=32)
    pts = acq.optimize(_space2d(), q=3, observed_Y=Y, rng=np.random.default_rng(1))
    assert pts.shape == (3, 2)
    assert (pts >= 0).all() and (pts <= 1).all()


def test_qehvi_requires_multi_obj() -> None:
    gp, Y = _train_gp(m=1)
    acq = QEHVIAcquisition(gp, mc_samples=16, num_restarts=2, raw_samples=16)
    with pytest.raises(AcquisitionError):
        acq.optimize(_space2d(), q=2, observed_Y=Y, rng=np.random.default_rng(1))


def test_qehvi_multi_obj_returns_q_points() -> None:
    gp, Y = _train_gp(m=2, n=12)
    acq = QEHVIAcquisition(gp, mc_samples=16, num_restarts=2, raw_samples=16)
    pts = acq.optimize(_space2d(), q=2, observed_Y=Y, rng=np.random.default_rng(1))
    assert pts.shape == (2, 2)


def test_make_acquisition_via_registry() -> None:
    gp, _ = _train_gp(m=1)
    acq = make_acquisition({"type": "qei", "options": {"mc_samples": 16}}, surrogate=gp)
    assert isinstance(acq, QEIAcquisition)
