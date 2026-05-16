from __future__ import annotations

import sqlite3
from pathlib import Path

import h5py
import numpy as np
import pytest

from polarisopt.metrics import ChoiceShareMetric, LinkMoeMetric
from polarisopt.metrics.base import MetricError


def _write_linkmoe(path: Path, tt: np.ndarray, vol: np.ndarray) -> None:
    with h5py.File(path, "w") as f:
        g = f.create_group("link_moe")
        g.create_dataset("link_travel_time", data=tt)
        g.create_dataset("link_in_volume", data=vol)


def test_link_moe_rmse_zero_when_identical(tmp_path: Path) -> None:
    tt = np.array([[1.0, 2.0], [3.0, 4.0]])
    vol = np.array([[10.0, 20.0], [30.0, 40.0]])
    target = tmp_path / "target.h5"
    sim = tmp_path / "sim.h5"
    _write_linkmoe(target, tt, vol)
    _write_linkmoe(sim, tt, vol)
    metric = LinkMoeMetric(target=target)
    out = metric.compute({"result_path": str(sim)})
    assert out.shape == (1,)
    assert float(out[0]) == pytest.approx(0.0)


def test_link_moe_rmse_nonzero_when_different(tmp_path: Path) -> None:
    tt = np.array([[1.0, 1.0]])
    vol = np.array([[10.0, 10.0]])
    sim_tt = tt + 0.5
    target = tmp_path / "target.h5"
    sim = tmp_path / "sim.h5"
    _write_linkmoe(target, tt, vol)
    _write_linkmoe(sim, sim_tt, vol)
    metric = LinkMoeMetric(target=target)
    out = metric.compute({"result_path": str(sim)})
    # per-link vehicle-time difference: mean across intervals of (1.5 - 1.0) * 10 = 5
    # so error = 5, RMSE = 5
    assert float(out[0]) == pytest.approx(5.0)


def test_link_moe_aggregation_kinds(tmp_path: Path) -> None:
    tt = np.array([[1.0]])
    vol = np.array([[1.0]])
    target = tmp_path / "t.h5"
    _write_linkmoe(target, tt, vol)
    _write_linkmoe(tmp_path / "s.h5", tt + 1, vol)
    for kind in ("rmse", "mse", "mae"):
        out = LinkMoeMetric(target=target, aggregation=kind).compute(
            {"result_path": str(tmp_path / "s.h5")}
        )
        assert out.shape == (1,) and out[0] > 0


def test_link_moe_rejects_missing_keys(tmp_path: Path) -> None:
    target = tmp_path / "t.h5"
    with h5py.File(target, "w") as f:
        f.create_group("other")  # missing link_moe
    sim = tmp_path / "s.h5"
    _write_linkmoe(sim, np.ones((1, 1)), np.ones((1, 1)))
    metric = LinkMoeMetric(target=target)
    with pytest.raises(MetricError, match="link_moe"):
        metric.compute({"result_path": str(sim)})


def test_link_moe_missing_result_path(tmp_path: Path) -> None:
    target = tmp_path / "t.h5"
    _write_linkmoe(target, np.ones((1, 1)), np.ones((1, 1)))
    with pytest.raises(MetricError, match="result_path"):
        LinkMoeMetric(target=target).compute({})


# ----- choice_share -----


def _make_demand_db(path: Path, rows: list[tuple[str, int]]) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE mode_share (mode TEXT, n INTEGER)")
    conn.executemany("INSERT INTO mode_share VALUES (?, ?)", rows)
    conn.commit()
    conn.close()


def test_choice_share_identical_yields_zero(tmp_path: Path) -> None:
    target = tmp_path / "target.sqlite"
    sim = tmp_path / "sim.sqlite"
    rows = [("auto", 50), ("transit", 30), ("walk", 20)]
    _make_demand_db(target, rows)
    _make_demand_db(sim, rows)
    metric = ChoiceShareMetric(
        target_db=target,
        sql="SELECT mode AS category, n AS count FROM mode_share",
    )
    out = metric.compute({"demand_db": str(sim)})
    assert out.shape == (1,)
    assert float(out[0]) == pytest.approx(0.0)


def test_choice_share_detects_mismatch(tmp_path: Path) -> None:
    target = tmp_path / "target.sqlite"
    sim = tmp_path / "sim.sqlite"
    _make_demand_db(target, [("auto", 50), ("transit", 50)])
    _make_demand_db(sim, [("auto", 60), ("transit", 40)])  # 10% shift
    metric = ChoiceShareMetric(
        target_db=target,
        sql="SELECT mode AS category, n AS count FROM mode_share",
    )
    out = metric.compute({"demand_db": str(sim)})
    assert float(out[0]) == pytest.approx(0.2)  # |0.6-0.5| + |0.4-0.5| = 0.2


def test_choice_share_vector_aggregation(tmp_path: Path) -> None:
    target = tmp_path / "target.sqlite"
    sim = tmp_path / "sim.sqlite"
    _make_demand_db(target, [("auto", 50), ("transit", 50)])
    _make_demand_db(sim, [("auto", 70), ("transit", 30)])
    metric = ChoiceShareMetric(
        target_db=target,
        sql="SELECT mode AS category, n AS count FROM mode_share",
        aggregation="vector",
    )
    assert metric.n_objectives == 2
    out = metric.compute({"demand_db": str(sim)})
    assert out.shape == (2,)
    # both categories shifted by 0.2
    np.testing.assert_allclose(out, [0.2, 0.2])


def test_choice_share_missing_source_key(tmp_path: Path) -> None:
    target = tmp_path / "target.sqlite"
    _make_demand_db(target, [("auto", 1)])
    metric = ChoiceShareMetric(target_db=target, sql="SELECT mode AS category, n AS count FROM mode_share")
    with pytest.raises(MetricError, match="demand_db"):
        metric.compute({})
