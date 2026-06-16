"""Tests for the SampleStore analysis helpers added in v0.5."""

from __future__ import annotations

import numpy as np
import pytest

from polarisopt.samples.sample import Sample, SampleStatus
from polarisopt.samples.store import SampleStore


def _seed(store: SampleStore, metrics: list[list[float] | None], phase: str = "p") -> list[Sample]:
    rows: list[Sample] = []
    for i, m in enumerate(metrics):
        s = store.add(Sample(phase=phase, iteration=i // 4, inputs=np.array([float(i)])))
        if m is None:
            s.status = SampleStatus.FAILED
            s.message = "boom"
        else:
            s.status = SampleStatus.FINISHED
            s.metric = np.asarray(m, dtype=float)
        store.update(s)
        rows.append(s)
    return rows


def test_finished_samples_filters_status_and_phase() -> None:
    store = SampleStore.open_memory("s")
    _seed(store, [[1.0], [2.0], None, [0.5]])
    finished = store.finished_samples()
    assert [s.metric[0] for s in finished] == [1.0, 2.0, 0.5]
    _seed(store, [[10.0]], phase="other")
    finished_phase = store.finished_samples(phase="p")
    assert len(finished_phase) == 3


def test_finished_samples_skips_finished_without_metric() -> None:
    """A FINISHED sample with metric=None still shouldn't return from finished_samples()."""
    store = SampleStore.open_memory("s")
    s = store.add(Sample(phase="p", inputs=np.array([0.1])))
    s.status = SampleStatus.FINISHED  # but no metric set
    store.update(s)
    assert store.finished_samples() == []


def test_metric_matrix_shape() -> None:
    store = SampleStore.open_memory("s")
    _seed(store, [[1.0, 0.5], [2.0, 0.25], [3.0, 0.125]])
    Y = store.metric_matrix()
    assert Y.shape == (3, 2)
    np.testing.assert_array_equal(Y[:, 0], [1.0, 2.0, 3.0])


def test_metric_matrix_empty() -> None:
    store = SampleStore.open_memory("s")
    assert store.metric_matrix().shape == (0, 0)


def test_best_so_far_minimize() -> None:
    store = SampleStore.open_memory("s")
    rows = _seed(store, [[3.0], [1.5], [0.25], [2.0]])
    result = store.best_so_far()
    assert result is not None
    sample, value = result
    assert sample.id == rows[2].id
    assert value == pytest.approx(0.25)


def test_best_so_far_maximize() -> None:
    store = SampleStore.open_memory("s")
    rows = _seed(store, [[3.0], [1.5], [0.25], [2.0]])
    result = store.best_so_far(minimize=False)
    assert result is not None
    sample, value = result
    assert sample.id == rows[0].id
    assert value == pytest.approx(3.0)


def test_best_so_far_objective_index() -> None:
    store = SampleStore.open_memory("s")
    _seed(store, [[1.0, 5.0], [2.0, 1.0], [3.0, 3.0]])
    # objective 0: best is row 0; objective 1: best is row 1
    a = store.best_so_far(objective_index=0)
    b = store.best_so_far(objective_index=1)
    assert a is not None and a[1] == pytest.approx(1.0)
    assert b is not None and b[1] == pytest.approx(1.0)


def test_best_so_far_empty_returns_none() -> None:
    store = SampleStore.open_memory("s")
    assert store.best_so_far() is None


def test_pareto_front_single_objective_collapses_to_min() -> None:
    store = SampleStore.open_memory("s")
    rows = _seed(store, [[3.0], [1.5], [0.25], [2.0]])
    front = store.pareto_front()
    assert len(front) == 1
    assert front[0].id == rows[2].id


def test_pareto_front_two_objective() -> None:
    store = SampleStore.open_memory("s")
    # Pareto-front should be (1,5), (3,2) — (5,5) is dominated by (1,5), (4,3) dominated by (3,2)
    metrics = [[1.0, 5.0], [3.0, 2.0], [5.0, 5.0], [4.0, 3.0]]
    rows = _seed(store, metrics)
    front_ids = {s.id for s in store.pareto_front()}
    assert front_ids == {rows[0].id, rows[1].id}


def test_pareto_front_maximize() -> None:
    store = SampleStore.open_memory("s")
    # When maximizing, the higher values dominate.
    metrics = [[1.0, 1.0], [2.0, 2.0], [3.0, 0.5]]
    rows = _seed(store, metrics)
    front_ids = {s.id for s in store.pareto_front(minimize=False)}
    assert front_ids == {rows[1].id, rows[2].id}


def test_pareto_front_phase_filter() -> None:
    store = SampleStore.open_memory("s")
    _seed(store, [[1.0, 1.0]], phase="warmup")
    _seed(store, [[0.5, 0.5]], phase="bo")
    front = store.pareto_front(phase="bo")
    assert len(front) == 1
    assert front[0].phase == "bo"
