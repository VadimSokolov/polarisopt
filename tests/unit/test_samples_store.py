from __future__ import annotations

import numpy as np
import pytest

from polarisopt.samples import Sample, SampleStatus, SampleStore


def test_add_and_get_roundtrip() -> None:
    store = SampleStore.open_memory("study1")
    s = Sample(phase="warmup", iteration=0, inputs=np.array([0.1, 0.2, 0.3]))
    store.add(s)
    assert s.id is not None
    assert s.created_at is not None

    fetched = store.get(s.id)
    assert fetched.phase == "warmup"
    np.testing.assert_array_almost_equal(fetched.inputs, [0.1, 0.2, 0.3])
    assert fetched.status is SampleStatus.PENDING
    assert fetched.metric is None


def test_update_persists_metric_and_status() -> None:
    store = SampleStore.open_memory("study2")
    s = store.add(Sample(phase="seq", inputs=np.array([1.0])))
    s.status = SampleStatus.FINISHED
    s.metric = np.array([42.0, 3.14])
    s.runtime_s = 12.5
    store.update(s)

    fetched = store.get(s.id)  # type: ignore[arg-type]
    assert fetched.status is SampleStatus.FINISHED
    np.testing.assert_array_almost_equal(fetched.metric, [42.0, 3.14])
    assert fetched.runtime_s == pytest.approx(12.5)


def test_list_with_filters() -> None:
    store = SampleStore.open_memory("study3")
    for i in range(3):
        store.add(Sample(phase="warmup", iteration=0, inputs=np.array([float(i)])))
    for i in range(2):
        s = store.add(Sample(phase="seq", iteration=1, inputs=np.array([float(i)])))
        s.status = SampleStatus.FINISHED
        s.metric = np.array([float(i)])
        store.update(s)

    assert store.count() == 5
    assert store.count(phase="warmup") == 3
    assert store.count(phase="seq", status=SampleStatus.FINISHED) == 2
    assert len(store.list(phase="seq")) == 2
    assert len(store.list(status=SampleStatus.PENDING)) == 3


def test_phase_state_roundtrip() -> None:
    store = SampleStore.open_memory("study4")
    assert store.load_phase_state("seq") is None
    store.save_phase_state("seq", iteration=0, rng_state=b"rng0")
    store.save_phase_state("seq", iteration=1, rng_state=b"rng1", surrogate_state=b"surr1")

    latest = store.load_phase_state("seq")
    assert latest is not None
    assert latest["iteration"] == 1
    assert latest["rng_state"] == b"rng1"
    assert latest["surrogate_state"] == b"surr1"


def test_reopen_same_study_attaches(tmp_path) -> None:
    db = tmp_path / "x.db"
    store_a = SampleStore.open(db, "shared")
    store_a.add(Sample(phase="p", inputs=np.array([1.0])))
    sid_a = store_a.study_id

    store_b = SampleStore.open(db, "shared")
    assert store_b.study_id == sid_a
    assert store_b.count() == 1


def test_to_dataframe_columns() -> None:
    store = SampleStore.open_memory("study5")
    s = store.add(Sample(phase="p", inputs=np.array([1.0, 2.0])))
    s.status = SampleStatus.FINISHED
    s.metric = np.array([7.0])
    store.update(s)
    df = store.to_dataframe()
    assert list(df.columns) == [
        "id", "phase", "iteration", "inputs", "status", "metric",
        "folder", "runtime_s", "runner_task_id", "message",
        "created_at", "updated_at",
    ]
    assert df.loc[0, "metric"] == [7.0]
