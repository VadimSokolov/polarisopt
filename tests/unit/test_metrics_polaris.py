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


def test_choice_share_cross_entropy_identical_is_target_entropy(tmp_path: Path) -> None:
    """CE(p||p) = H(p) — the entropy of the target."""
    target = tmp_path / "target.sqlite"
    sim = tmp_path / "sim.sqlite"
    rows = [("auto", 80), ("walk", 10), ("transit", 10)]
    _make_demand_db(target, rows)
    _make_demand_db(sim, rows)
    metric = ChoiceShareMetric(
        target_db=target,
        sql="SELECT mode AS category, n AS count FROM mode_share",
        aggregation="cross_entropy",
    )
    out = metric.compute({"demand_db": str(sim)})
    assert out.shape == (1,)
    p = np.array([0.8, 0.1, 0.1])
    expected = -float(np.sum(p * np.log(p)))
    assert float(out[0]) == pytest.approx(expected, rel=1e-9)


def test_choice_share_kl_divergence_identical_is_zero(tmp_path: Path) -> None:
    """KL(p||p) = 0 — desirable for a "zero at perfect match" objective."""
    target = tmp_path / "target.sqlite"
    sim = tmp_path / "sim.sqlite"
    rows = [("auto", 80), ("walk", 10), ("transit", 10)]
    _make_demand_db(target, rows)
    _make_demand_db(sim, rows)
    metric = ChoiceShareMetric(
        target_db=target,
        sql="SELECT mode AS category, n AS count FROM mode_share",
        aggregation="kl_divergence",
    )
    out = metric.compute({"demand_db": str(sim)})
    assert out.shape == (1,)
    assert float(out[0]) == pytest.approx(0.0, abs=1e-12)


def test_choice_share_cross_entropy_zero_sim_clipped_at_eps(tmp_path: Path) -> None:
    """A sim missing a target-present category doesn't blow up — clipped to eps."""
    target = tmp_path / "target.sqlite"
    sim = tmp_path / "sim.sqlite"
    _make_demand_db(target, [("auto", 60), ("walk", 20), ("transit", 20)])
    _make_demand_db(sim,    [("auto", 60), ("walk", 20)])  # transit missing
    metric = ChoiceShareMetric(
        target_db=target,
        sql="SELECT mode AS category, n AS count FROM mode_share",
        aggregation="cross_entropy",
        eps=1e-6,
    )
    out = metric.compute({"demand_db": str(sim)})
    # transit contribution: -0.2 * log(1e-6) ≈ 2.764 (large but finite)
    assert np.isfinite(out[0])
    assert float(out[0]) > 2.0
    assert float(out[0]) < 5.0


def test_choice_share_kl_zero_target_category_ignored(tmp_path: Path) -> None:
    """Categories with p_target = 0 must not contribute (0 * log = 0)."""
    target = tmp_path / "target.sqlite"
    sim = tmp_path / "sim.sqlite"
    # Target has no BIKE; sim reports some. KL(p||q) should ignore BIKE.
    _make_demand_db(target, [("auto", 80), ("walk", 20)])
    _make_demand_db(sim,    [("auto", 80), ("walk", 15), ("bike", 5)])
    metric = ChoiceShareMetric(
        target_db=target,
        sql="SELECT mode AS category, n AS count FROM mode_share",
        aggregation="kl_divergence",
    )
    out = metric.compute({"demand_db": str(sim)})
    # KL = 0.8*log(0.8/0.8) + 0.2*log(0.2/0.15) = 0 + 0.2*log(4/3)
    expected = 0.2 * float(np.log(0.2 / 0.15))
    assert float(out[0]) == pytest.approx(expected, rel=1e-6)


def test_choice_share_rejects_bad_aggregation() -> None:
    with pytest.raises(ValueError, match="unknown aggregation"):
        ChoiceShareMetric(target_db="/dev/null", sql="", aggregation="not_a_real_one")


def test_choice_share_rejects_bad_eps() -> None:
    for bad in (0, -1e-9, float("nan"), float("inf"), "1e-6"):
        with pytest.raises(ValueError, match="eps"):
            ChoiceShareMetric(target_db="/dev/null", sql="", eps=bad)  # type: ignore[arg-type]
