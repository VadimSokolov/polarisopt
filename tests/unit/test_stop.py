from __future__ import annotations

import numpy as np
import pytest

from polarisopt.stop import (
    AllStop,
    AnyStop,
    EpsilonStop,
    HypervolumeStop,
    MaxIterStop,
    PlateauStop,
    make_stop,
)
from polarisopt.stop.base import StoppingState


def _state(iter_n: int, Y: np.ndarray, history: list[StoppingState] | None = None, minimize: bool = True) -> StoppingState:
    X = np.zeros((Y.shape[0], 2)) if Y.size else np.empty((0, 2))
    return StoppingState(iteration=iter_n, X=X, Y=Y, history=history or [], minimize=minimize)


def test_max_iter_fires_at_n() -> None:
    s = MaxIterStop(n=5)
    assert not s.should_stop(_state(0, np.empty((0, 1))))
    assert not s.should_stop(_state(4, np.empty((0, 1))))
    assert s.should_stop(_state(5, np.empty((0, 1))))


def test_max_iter_rejects_bad_n() -> None:
    with pytest.raises(ValueError):
        MaxIterStop(n=0)


def test_epsilon_minimization() -> None:
    s = EpsilonStop(epsilon=0.01, target=0.0)
    assert not s.should_stop(_state(0, np.array([[1.0]])))
    assert s.should_stop(_state(0, np.array([[0.005]])))


def test_epsilon_handles_empty_y() -> None:
    s = EpsilonStop(epsilon=0.01)
    assert not s.should_stop(_state(0, np.empty((0, 1))))


def test_plateau_fires_when_history_window_flat() -> None:
    s = PlateauStop(tol=1e-3, window=3)
    history = [_state(i, np.array([[1.0 + 1e-4 * i]])) for i in range(3)]
    state = _state(3, np.array([[1.0]]), history=history)
    assert s.should_stop(state)


def test_plateau_does_not_fire_when_improving() -> None:
    s = PlateauStop(tol=1e-3, window=3)
    history = [_state(i, np.array([[1.0 - 0.5 * i]])) for i in range(3)]
    assert not s.should_stop(_state(3, np.array([[0.5]]), history=history))


def test_combinators_any_all() -> None:
    fire = MaxIterStop(n=1)
    never = MaxIterStop(n=999)
    assert AnyStop([fire, never]).should_stop(_state(5, np.empty((0, 1))))
    assert not AllStop([fire, never]).should_stop(_state(5, np.empty((0, 1))))


def test_hypervolume_eventually_stagnates() -> None:
    s = HypervolumeStop(ref_point=[10.0, 10.0], tol=1e-6, patience=2)
    Y = np.array([[1.0, 5.0], [5.0, 1.0]])  # Pareto front of 2 points
    # First call: prev_hv set, no stop
    assert not s.should_stop(_state(0, Y))
    # Second call: same HV → stagnant=1, not yet
    assert not s.should_stop(_state(1, Y))
    # Third call: stagnant=2 == patience → stop
    assert s.should_stop(_state(2, Y))


def test_hypervolume_3d_via_botorch() -> None:
    """Smoke test for the 3-objective code path (requires [bo] extra)."""
    pytest.importorskip("torch")
    s = HypervolumeStop(ref_point=[10.0, 10.0, 10.0], tol=1e-6, patience=2)
    Y = np.array([[1.0, 5.0, 2.0], [5.0, 1.0, 4.0], [3.0, 3.0, 1.0]])
    assert s.n_objectives == 3
    assert not s.should_stop(_state(0, Y))
    assert not s.should_stop(_state(1, Y))
    assert s.should_stop(_state(2, Y))


def test_hypervolume_rejects_1d_ref_point() -> None:
    with pytest.raises(ValueError, match="length >= 2"):
        HypervolumeStop(ref_point=[10.0])


def test_hypervolume_ignores_wrong_shape_y() -> None:
    s = HypervolumeStop(ref_point=[10.0, 10.0])
    # Y has 3 obj but stop expects 2 — silently no-op
    Y3 = np.array([[1.0, 2.0, 3.0]])
    assert not s.should_stop(_state(0, Y3))


def test_make_stop_recursive() -> None:
    spec = {
        "type": "any",
        "criteria": [
            {"type": "max_iter", "options": {"n": 3}},
            {"type": "epsilon", "options": {"epsilon": 0.1}},
        ],
    }
    s = make_stop(spec)
    assert isinstance(s, AnyStop)
    assert len(s.criteria) == 2
