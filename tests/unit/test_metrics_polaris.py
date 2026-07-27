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
    """CE(p||p) = H(p) — holds only when smoothing is disabled (opt-out path)."""
    target = tmp_path / "target.sqlite"
    sim = tmp_path / "sim.sqlite"
    rows = [("auto", 80), ("walk", 10), ("transit", 10)]
    _make_demand_db(target, rows)
    _make_demand_db(sim, rows)
    metric = ChoiceShareMetric(
        target_db=target,
        sql="SELECT mode AS category, n AS count FROM mode_share",
        aggregation="cross_entropy",
        laplace_smoothing_alpha=0,  # v0.20 semantics — identity math only
    )
    out = metric.compute({"demand_db": str(sim)})
    assert out.shape == (1,)
    p = np.array([0.8, 0.1, 0.1])
    expected = -float(np.sum(p * np.log(p)))
    assert float(out[0]) == pytest.approx(expected, rel=1e-9)


def test_choice_share_kl_divergence_identical_is_zero(tmp_path: Path) -> None:
    """KL(p||p) = 0 — holds only when smoothing is disabled (opt-out path)."""
    target = tmp_path / "target.sqlite"
    sim = tmp_path / "sim.sqlite"
    rows = [("auto", 80), ("walk", 10), ("transit", 10)]
    _make_demand_db(target, rows)
    _make_demand_db(sim, rows)
    metric = ChoiceShareMetric(
        target_db=target,
        sql="SELECT mode AS category, n AS count FROM mode_share",
        aggregation="kl_divergence",
        laplace_smoothing_alpha=0,
    )
    out = metric.compute({"demand_db": str(sim)})
    assert out.shape == (1,)
    assert float(out[0]) == pytest.approx(0.0, abs=1e-12)


def test_choice_share_cross_entropy_zero_sim_clipped_at_eps(tmp_path: Path) -> None:
    """A sim missing a target-present category doesn't blow up — clipped to eps.

    Only reachable via the smoothing-disabled opt-out (alpha=0). This is the
    v0.20 behavior kept as a compat path, not the default.
    """
    target = tmp_path / "target.sqlite"
    sim = tmp_path / "sim.sqlite"
    _make_demand_db(target, [("auto", 60), ("walk", 20), ("transit", 20)])
    _make_demand_db(sim,    [("auto", 60), ("walk", 20)])  # transit missing
    metric = ChoiceShareMetric(
        target_db=target,
        sql="SELECT mode AS category, n AS count FROM mode_share",
        aggregation="cross_entropy",
        eps=1e-6,
        laplace_smoothing_alpha=0,
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
        laplace_smoothing_alpha=0,
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


# ----- v0.21: Laplace-smoothed CE / KL + jensen_shannon -----


def test_choice_share_cross_entropy_default_uses_laplace_alpha_one(tmp_path: Path) -> None:
    """Default alpha=1 gives (n_k+1)/(N+K) smoothed sim shares, not raw."""
    target = tmp_path / "target.sqlite"
    sim = tmp_path / "sim.sqlite"
    rows = [("auto", 80), ("walk", 10), ("transit", 10)]  # N=100, K=3
    _make_demand_db(target, rows)
    _make_demand_db(sim, rows)
    metric = ChoiceShareMetric(
        target_db=target,
        sql="SELECT mode AS category, n AS count FROM mode_share",
        aggregation="cross_entropy",
    )
    out = metric.compute({"demand_db": str(sim)})
    p_tgt = np.array([0.8, 0.1, 0.1])
    p_sim_smoothed = np.array([81 / 103, 11 / 103, 11 / 103])
    expected = -float(np.sum(p_tgt * np.log(p_sim_smoothed)))
    assert float(out[0]) == pytest.approx(expected, rel=1e-9)


def test_choice_share_cross_entropy_laplace_no_bimodality_when_sim_zero(tmp_path: Path) -> None:
    """The v0.20 bug: sim missing a target-positive mode → CE blows up to ~27.

    With alpha=1 (v0.21 default) the smoothed sim share on the missing mode
    is 1/(N+K) ≈ 1/103 ≈ 0.0097, so the contribution is
    -0.2 * log(1/103) ≈ 0.93 — bounded and comparable to the other terms.
    """
    target = tmp_path / "target.sqlite"
    sim = tmp_path / "sim.sqlite"
    _make_demand_db(target, [("auto", 60), ("walk", 20), ("transit", 20)])
    _make_demand_db(sim,    [("auto", 60), ("walk", 40)])  # transit missing, N=100, K=3
    metric = ChoiceShareMetric(
        target_db=target,
        sql="SELECT mode AS category, n AS count FROM mode_share",
        aggregation="cross_entropy",
    )
    out = metric.compute({"demand_db": str(sim)})
    # Smoothed: auto=(60+1)/103, walk=(40+1)/103, transit=(0+1)/103
    expected = -(
        0.6 * np.log(61 / 103) + 0.2 * np.log(41 / 103) + 0.2 * np.log(1 / 103)
    )
    assert float(out[0]) == pytest.approx(expected, rel=1e-9)
    # Regression against v0.20 blow-up: value MUST be < 2.0. With eps=1e-12
    # the transit term alone would have contributed 0.2 * 27.6 ≈ 5.5.
    assert float(out[0]) < 2.0


def test_choice_share_kl_default_alpha_one_positive_at_identity(tmp_path: Path) -> None:
    """With alpha=1 the smoothed KL is small but nonzero at perfect identity."""
    target = tmp_path / "target.sqlite"
    sim = tmp_path / "sim.sqlite"
    rows = [("auto", 80), ("walk", 10), ("transit", 10)]  # N=100, K=3
    _make_demand_db(target, rows)
    _make_demand_db(sim, rows)
    metric = ChoiceShareMetric(
        target_db=target,
        sql="SELECT mode AS category, n AS count FROM mode_share",
        aggregation="kl_divergence",
    )
    out = metric.compute({"demand_db": str(sim)})
    p_tgt = np.array([0.8, 0.1, 0.1])
    p_sim = np.array([81 / 103, 11 / 103, 11 / 103])
    expected = float(np.sum(p_tgt * (np.log(p_tgt) - np.log(p_sim))))
    assert float(out[0]) == pytest.approx(expected, rel=1e-9)
    assert float(out[0]) > 0  # smoothing pulls sim away from target


def test_choice_share_jensen_shannon_identity_is_zero(tmp_path: Path) -> None:
    """JS(p, p) = 0. No eps or smoothing needed."""
    target = tmp_path / "target.sqlite"
    sim = tmp_path / "sim.sqlite"
    rows = [("auto", 80), ("walk", 10), ("transit", 10)]
    _make_demand_db(target, rows)
    _make_demand_db(sim, rows)
    metric = ChoiceShareMetric(
        target_db=target,
        sql="SELECT mode AS category, n AS count FROM mode_share",
        aggregation="jensen_shannon",
    )
    out = metric.compute({"demand_db": str(sim)})
    assert float(out[0]) == pytest.approx(0.0, abs=1e-12)


def test_choice_share_jensen_shannon_bounded_by_ln2(tmp_path: Path) -> None:
    """Maximally-disjoint distributions saturate JS at ln 2."""
    target = tmp_path / "target.sqlite"
    sim = tmp_path / "sim.sqlite"
    _make_demand_db(target, [("auto", 100), ("walk", 0)])
    _make_demand_db(sim,    [("auto", 0),   ("walk", 100)])
    metric = ChoiceShareMetric(
        target_db=target,
        sql="SELECT mode AS category, n AS count FROM mode_share",
        aggregation="jensen_shannon",
    )
    out = metric.compute({"demand_db": str(sim)})
    assert float(out[0]) == pytest.approx(float(np.log(2)), rel=1e-9)


def test_choice_share_jensen_shannon_symmetric(tmp_path: Path) -> None:
    """JS(p, q) == JS(q, p) — swap roles by swapping the SQLite files."""
    a = tmp_path / "a.sqlite"
    b = tmp_path / "b.sqlite"
    _make_demand_db(a, [("auto", 70), ("walk", 20), ("transit", 10)])
    _make_demand_db(b, [("auto", 40), ("walk", 30), ("transit", 30)])
    forward = ChoiceShareMetric(
        target_db=a,
        sql="SELECT mode AS category, n AS count FROM mode_share",
        aggregation="jensen_shannon",
    ).compute({"demand_db": str(b)})
    reverse = ChoiceShareMetric(
        target_db=b,
        sql="SELECT mode AS category, n AS count FROM mode_share",
        aggregation="jensen_shannon",
    ).compute({"demand_db": str(a)})
    assert float(forward[0]) == pytest.approx(float(reverse[0]), rel=1e-12)


def test_choice_share_jensen_shannon_missing_category(tmp_path: Path) -> None:
    """Categories present in sim but not target (and vice-versa) contribute finitely."""
    target = tmp_path / "target.sqlite"
    sim = tmp_path / "sim.sqlite"
    _make_demand_db(target, [("auto", 60), ("walk", 40)])
    _make_demand_db(sim,    [("auto", 60), ("walk", 20), ("bike", 20)])
    metric = ChoiceShareMetric(
        target_db=target,
        sql="SELECT mode AS category, n AS count FROM mode_share",
        aggregation="jensen_shannon",
    )
    out = metric.compute({"demand_db": str(sim)})
    assert np.isfinite(out[0])
    assert 0.0 < float(out[0]) < float(np.log(2))


def test_choice_share_rejects_bad_laplace_alpha() -> None:
    for bad in (-1e-9, float("nan"), float("inf"), "1", [1.0], True):
        with pytest.raises(ValueError, match="laplace_smoothing_alpha"):
            ChoiceShareMetric(
                target_db="/dev/null",
                sql="",
                laplace_smoothing_alpha=bad,  # type: ignore[arg-type]
            )
    # Zero is a valid opt-out (fall back to eps), not an error.
    ChoiceShareMetric(target_db="/dev/null", sql="", laplace_smoothing_alpha=0)
