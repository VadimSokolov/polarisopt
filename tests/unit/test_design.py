from __future__ import annotations

import numpy as np
import pytest

from polarisopt.design import (
    LHSDesign,
    ManualDesign,
    MorrisDesign,
    SobolDesign,
    design_registry,
    make_design,
)
from polarisopt.parameters import Parameter, ParameterSpace, ParameterType


@pytest.fixture
def space() -> ParameterSpace:
    return ParameterSpace.from_iterable(
        [
            Parameter("a", "a.json", 0.0, 1.0),
            Parameter("b", "b.json", -1.0, 1.0),
            Parameter("c", "c.json", 0, 10, ParameterType.INT),
        ]
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


def test_lhs_shape_and_bounds(space: ParameterSpace, rng: np.random.Generator) -> None:
    pts = LHSDesign(n=8).generate(space, rng=rng)
    assert pts.shape == (8, 3)
    bounds = space.bounds
    assert (pts[:, 0] >= bounds[0, 0]).all() and (pts[:, 0] <= bounds[0, 1]).all()
    assert (pts[:, 1] >= bounds[1, 0]).all() and (pts[:, 1] <= bounds[1, 1]).all()
    # int dim
    assert np.all(pts[:, 2] == pts[:, 2].astype(int))


def test_lhs_rejects_nonpositive_n() -> None:
    with pytest.raises(ValueError):
        LHSDesign(n=0)


def test_lhs_deterministic_with_seed(space: ParameterSpace) -> None:
    a = LHSDesign(n=10).generate(space, rng=np.random.default_rng(7))
    b = LHSDesign(n=10).generate(space, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


def test_sobol_shape_and_bounds(space: ParameterSpace, rng: np.random.Generator) -> None:
    pts = SobolDesign(n=8).generate(space, rng=rng)
    assert pts.shape == (8, 3)
    bounds = space.bounds
    assert (pts[:, 0] >= bounds[0, 0]).all() and (pts[:, 0] <= bounds[0, 1]).all()


def test_morris_shape(space: ParameterSpace, rng: np.random.Generator) -> None:
    # Morris emits N*(d+1) rows
    pts = MorrisDesign(n_trajectories=3, num_levels=4).generate(space, rng=rng)
    assert pts.shape == (3 * (space.ndim + 1), space.ndim)
    # bounds respected
    bounds = space.bounds
    assert np.all(pts >= bounds[:, 0])
    assert np.all(pts <= bounds[:, 1])


def test_morris_invalid_args() -> None:
    with pytest.raises(ValueError):
        MorrisDesign(n_trajectories=0)
    with pytest.raises(ValueError):
        MorrisDesign(n_trajectories=2, num_levels=1)


def test_manual_design(space: ParameterSpace, rng: np.random.Generator) -> None:
    pts = ManualDesign(points=[[0.5, 0.0, 3.7], [-1.0, 99.0, 4.4]]).generate(space, rng=rng)
    assert pts.shape == (2, 3)
    # second row's b=99 clipped to upper bound 1.0, c rounded to 4
    assert pts[1, 1] == 1.0
    assert pts[1, 2] == 4


def test_manual_design_dim_mismatch(space: ParameterSpace, rng: np.random.Generator) -> None:
    with pytest.raises(ValueError):
        ManualDesign(points=[[0.0, 0.0]]).generate(space, rng=rng)


def test_make_design_uses_registry(space: ParameterSpace, rng: np.random.Generator) -> None:
    d = make_design({"type": "lhs", "options": {"n": 4}})
    assert isinstance(d, LHSDesign)
    assert d.generate(space, rng=rng).shape == (4, 3)


def test_make_design_unknown_type() -> None:
    with pytest.raises(KeyError, match="Unknown design"):
        make_design({"type": "not_a_thing"})


def test_make_design_missing_type() -> None:
    with pytest.raises(ValueError, match="missing 'type'"):
        make_design({"options": {}})


def test_design_registry_lists_all_builtins() -> None:
    assert {"lhs", "morris", "sobol", "manual"}.issubset(design_registry.names())
