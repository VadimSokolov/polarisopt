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


# ---------- v0.18: qlognei (noise-aware) ----------

from polarisopt.acquisition.qlognei import QLogNEIAcquisition  # noqa: E402


def _train_gp_with_x(m: int = 1, n: int = 10, seed: int = 0) -> tuple[GPSurrogate, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(n, 2))
    if m == 1:
        Y = np.sum((X - 0.3) ** 2, axis=1, keepdims=True)
    else:
        Y = np.stack(
            [np.sum((X - 0.3) ** 2, axis=1), np.sum((X - 0.7) ** 2, axis=1)], axis=1,
        )
    gp = GPSurrogate()
    gp.fit(X, Y)
    return gp, X, Y


def test_qlognei_returns_q_points_in_bounds() -> None:
    gp, X, Y = _train_gp_with_x(m=1, n=10)
    acq = QLogNEIAcquisition(gp, mc_samples=32, num_restarts=2, raw_samples=32)
    pts = acq.optimize(
        _space2d(), q=3, observed_Y=Y, observed_X=X, rng=np.random.default_rng(1),
    )
    assert pts.shape == (3, 2)
    assert (pts >= 0).all() and (pts <= 1).all()


def test_qlognei_requires_observed_x() -> None:
    """The whole point of NEI is X_baseline. Refusing it in the API
    catches driver-side wiring mistakes at test time rather than at BO
    time on the cluster."""
    gp, _, Y = _train_gp_with_x(m=1, n=10)
    acq = QLogNEIAcquisition(gp, mc_samples=16, num_restarts=2, raw_samples=16)
    with pytest.raises(AcquisitionError, match="observed_X"):
        acq.optimize(_space2d(), q=2, observed_Y=Y, rng=np.random.default_rng(1))


def test_qlognei_rejects_multi_obj() -> None:
    gp, X, Y = _train_gp_with_x(m=2, n=12)
    acq = QLogNEIAcquisition(gp, mc_samples=16, num_restarts=2, raw_samples=16)
    with pytest.raises(AcquisitionError, match="single-objective"):
        acq.optimize(
            _space2d(), q=2, observed_Y=Y, observed_X=X, rng=np.random.default_rng(1),
        )


def test_qlognei_rejects_mismatched_row_counts() -> None:
    gp, X, Y = _train_gp_with_x(m=1, n=10)
    acq = QLogNEIAcquisition(gp, mc_samples=16, num_restarts=2, raw_samples=16)
    with pytest.raises(AcquisitionError, match="row counts disagree"):
        acq.optimize(
            _space2d(), q=1, observed_Y=Y, observed_X=X[:-1],
            rng=np.random.default_rng(1),
        )


def test_qlognei_registered() -> None:
    gp, _, _ = _train_gp_with_x(m=1)
    acq = make_acquisition(
        {"type": "qlognei", "options": {"mc_samples": 16}}, surrogate=gp,
    )
    assert isinstance(acq, QLogNEIAcquisition)


def test_generator_forwards_observed_x_to_qlognei() -> None:
    """AcquisitionGenerator must pass X through so qlognei can use it as
    X_baseline. This is the v0.18 wiring change end-to-end."""
    from polarisopt.generators.acquisition import AcquisitionGenerator
    from polarisopt.generators.base import GeneratorContext

    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, size=(10, 2))
    Y = np.sum((X - 0.3) ** 2, axis=1, keepdims=True) + 0.001 * rng.standard_normal((10, 1))

    gen = AcquisitionGenerator(
        surrogate={"type": "gp", "options": {}},
        acquisition={
            "type": "qlognei",
            "options": {"mc_samples": 16, "num_restarts": 2, "raw_samples": 32},
        },
    )
    ctx = GeneratorContext(space=_space2d(), X=X, Y=Y, rng=rng, iteration=0)
    pts = gen.next(ctx, q=2)
    assert pts.shape == (2, 2)


def test_qei_still_accepts_observed_x_and_ignores_it() -> None:
    """Backwards-compat: existing qei plugin accepts the new kwarg without
    complaining, and ignores it — behavior identical to pre-v0.18."""
    gp, X, Y = _train_gp_with_x(m=1, n=10)
    acq = QEIAcquisition(gp, mc_samples=32, num_restarts=2, raw_samples=32)
    pts = acq.optimize(
        _space2d(), q=2, observed_Y=Y, observed_X=X, rng=np.random.default_rng(1),
    )
    assert pts.shape == (2, 2)
