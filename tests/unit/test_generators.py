from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from polarisopt.generators import (
    AcquisitionGenerator,
    GeneratorContext,
    RandomGenerator,
    make_generator,
)
from polarisopt.generators.base import SampleGeneratorError
from polarisopt.parameters import Parameter, ParameterSpace


def _space() -> ParameterSpace:
    return ParameterSpace.from_iterable(
        [Parameter("x1", "a.json", 0.0, 1.0), Parameter("x2", "a.json", 0.0, 1.0)]
    )


def test_random_generator_batch_in_bounds() -> None:
    g = RandomGenerator()
    ctx = GeneratorContext(
        space=_space(),
        X=np.empty((0, 2)),
        Y=np.empty((0, 1)),
        iteration=0,
        rng=np.random.default_rng(0),
    )
    pts = g.next(ctx, q=5)
    assert pts.shape == (5, 2)
    assert (pts >= 0).all() and (pts <= 1).all()


def test_acquisition_generator_end_to_end() -> None:
    g = AcquisitionGenerator(
        surrogate={"type": "gp", "options": {}},
        acquisition={"type": "qei", "options": {"mc_samples": 16, "num_restarts": 2, "raw_samples": 32}},
    )
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(8, 2))
    Y = np.sum((X - 0.3) ** 2, axis=1, keepdims=True)
    pts = g.next(
        GeneratorContext(space=_space(), X=X, Y=Y, iteration=0, rng=rng),
        q=2,
    )
    assert pts.shape == (2, 2)


def test_acquisition_generator_needs_history() -> None:
    g = AcquisitionGenerator(
        surrogate={"type": "gp", "options": {}},
        acquisition={"type": "qei", "options": {"mc_samples": 16}},
    )
    with pytest.raises(SampleGeneratorError):
        g.next(
            GeneratorContext(
                space=_space(),
                X=np.empty((0, 2)),
                Y=np.empty((0, 1)),
                iteration=0,
                rng=np.random.default_rng(0),
            ),
            q=1,
        )


def test_make_generator_factory() -> None:
    g = make_generator({"type": "random"})
    assert isinstance(g, RandomGenerator)
