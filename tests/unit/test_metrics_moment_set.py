"""Unit tests for the v0.26 moment_set metric."""

from __future__ import annotations

import sqlite3
import warnings
from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest

from polarisopt.metrics import MomentSetMetric, make_metric
from polarisopt.metrics.base import MetricError


def _write_csv(path: Path, header: str, rows: list[str]) -> None:
    path.write_text(header + "\n" + "\n".join(rows) + "\n")


def _write_demand_db(path: Path, rows: list[tuple[str, str, int]]) -> None:
    """Two-key demand DB: (purpose, mode) → count. Trip table with a
    trivial schema the metric SQL can hit."""
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE Trip (purpose TEXT, mode TEXT, n INTEGER)")
    conn.executemany("INSERT INTO Trip VALUES (?, ?, ?)", rows)
    conn.commit()
    conn.close()


def _mode_share_sql() -> str:
    return dedent(
        """
        SELECT purpose, mode,
               SUM(n) * 1.0 / (SELECT SUM(n) FROM Trip WHERE purpose IS NOT NULL) AS share
        FROM Trip
        GROUP BY purpose, mode
        """
    )


def _mode_share_moment(target_path: Path) -> dict:
    return {
        "name": "mode_shares_by_purpose",
        "source_sql": _mode_share_sql(),
        "target": str(target_path),
        "target_key_cols": ["purpose", "mode"],
        "target_value_col": "share",
        "obs_noise_std": 0.005,
        "model_discrepancy_std": 0.02,
    }


def test_identity_target_yields_zero_residuals(tmp_path: Path) -> None:
    """Metric-on-target-DB: residuals must be exactly 0 element-by-element.

    This is the P7-style self-verification property — a metric that fails
    this on its own target is broken."""
    target_csv = tmp_path / "modes.csv"
    _write_csv(
        target_csv,
        "purpose,mode,share",
        ["HBW,auto,0.6", "HBW,transit,0.2", "HBW,walk,0.2"],
    )
    sim = tmp_path / "sim.sqlite"
    _write_demand_db(sim, [("HBW", "auto", 60), ("HBW", "transit", 20), ("HBW", "walk", 20)])
    metric = MomentSetMetric(moments=[_mode_share_moment(target_csv)])
    out = metric.compute({"demand_db": str(sim)})
    assert out.shape == (3,)
    np.testing.assert_allclose(out, np.zeros(3), atol=1e-12)


def test_n_objectives_matches_target_row_count(tmp_path: Path) -> None:
    """n_objectives is fixed at construction and equals the sum of
    per-moment target row counts."""
    t1 = tmp_path / "t1.csv"
    _write_csv(t1, "purpose,mode,share", ["HBW,auto,0.6", "HBW,transit,0.4"])
    t2 = tmp_path / "t2.csv"
    _write_csv(t2, "purpose,mode,share", ["HBO,auto,0.7", "HBO,transit,0.2", "HBO,walk,0.1"])
    metric = MomentSetMetric(
        moments=[
            {**_mode_share_moment(t1), "name": "hbw"},
            {**_mode_share_moment(t2), "name": "hbo"},
        ],
    )
    assert metric.n_objectives == 5
    assert metric.moment_slices == {"hbw": slice(0, 2), "hbo": slice(2, 5)}


def test_residuals_are_sim_minus_target(tmp_path: Path) -> None:
    target_csv = tmp_path / "t.csv"
    _write_csv(
        target_csv,
        "purpose,mode,share",
        ["HBW,auto,0.5", "HBW,transit,0.5"],
    )
    sim = tmp_path / "sim.sqlite"
    _write_demand_db(sim, [("HBW", "auto", 60), ("HBW", "transit", 40)])
    metric = MomentSetMetric(moments=[_mode_share_moment(target_csv)])
    out = metric.compute({"demand_db": str(sim)})
    # sim = [0.6, 0.4], target = [0.5, 0.5] → residual [0.1, -0.1]
    np.testing.assert_allclose(out, [0.1, -0.1], atol=1e-9)


def test_missing_sim_category_treated_as_zero(tmp_path: Path) -> None:
    """A target-present category with no sim rows contributes
    ``0 - target_k`` (not eps-clipped, not blown up)."""
    target_csv = tmp_path / "t.csv"
    _write_csv(
        target_csv,
        "purpose,mode,share",
        ["HBW,auto,0.6", "HBW,transit,0.2", "HBW,walk,0.2"],
    )
    sim = tmp_path / "sim.sqlite"
    _write_demand_db(sim, [("HBW", "auto", 60), ("HBW", "transit", 40)])  # no walk
    metric = MomentSetMetric(moments=[_mode_share_moment(target_csv)])
    out = metric.compute({"demand_db": str(sim)})
    # sim: auto 0.6, transit 0.4, walk missing → 0
    # target: 0.6, 0.2, 0.2 → residual: 0.0, 0.2, -0.2
    np.testing.assert_allclose(out, [0.0, 0.2, -0.2], atol=1e-9)


def test_extra_sim_category_ignored(tmp_path: Path) -> None:
    """A sim row whose key isn't in the target dropps silently — the
    target defines the calibration objective."""
    target_csv = tmp_path / "t.csv"
    _write_csv(target_csv, "purpose,mode,share", ["HBW,auto,1.0"])
    sim = tmp_path / "sim.sqlite"
    _write_demand_db(sim, [("HBW", "auto", 60), ("HBW", "bike", 40)])
    metric = MomentSetMetric(moments=[_mode_share_moment(target_csv)])
    out = metric.compute({"demand_db": str(sim)})
    assert out.shape == (1,)
    # sim auto share = 0.6, target 1.0 → -0.4
    np.testing.assert_allclose(out, [-0.4], atol=1e-9)


def test_weight_per_element_does_not_scale_raw_residuals(tmp_path: Path) -> None:
    """v0.36: the raw residual vector (scalarize='none') is UNWEIGHTED.

    This test previously asserted the opposite. Folding the weight into
    the residual meant history matching and calibrate-md — which both
    consume this vector — saw weighted residuals while their obs/md
    denominators stayed unweighted, so any weight != 1 silently
    rescaled the Vernon 3-sigma cutoff. The weight now applies only to
    scalarize='sum_squared_weighted'.
    """
    target_csv = tmp_path / "t.csv"
    _write_csv(target_csv, "purpose,mode,share", ["HBW,auto,0.5", "HBW,transit,0.5"])
    sim = tmp_path / "sim.sqlite"
    _write_demand_db(sim, [("HBW", "auto", 60), ("HBW", "transit", 40)])
    m = {**_mode_share_moment(target_csv), "weight_per_element": 3.0}
    out = MomentSetMetric(moments=[m]).compute({"demand_db": str(sim)})
    np.testing.assert_allclose(out, [0.1, -0.1], atol=1e-9)


def test_log_ratio_residual_aggregation(tmp_path: Path) -> None:
    """log_ratio_residual returns log((sim + eps) / (target + eps))."""
    target_csv = tmp_path / "boardings.csv"
    _write_csv(
        target_csv, "agency,type,boardings",
        ["CATS,bus,1000", "CATS,rail,200"],
    )
    sim = tmp_path / "sim.sqlite"
    conn = sqlite3.connect(str(sim))
    conn.execute("CREATE TABLE Boarding (agency TEXT, type TEXT, n INTEGER)")
    conn.executemany(
        "INSERT INTO Boarding VALUES (?, ?, ?)",
        [("CATS", "bus", 800), ("CATS", "rail", 250)],
    )
    conn.commit()
    conn.close()
    metric = MomentSetMetric(moments=[{
        "name": "boarding",
        "source_sql": "SELECT agency, type, SUM(n) AS boardings FROM Boarding GROUP BY agency, type",
        "target": str(target_csv),
        "target_key_cols": ["agency", "type"],
        "target_value_col": "boardings",
        "obs_noise_std": 100.0,
        "model_discrepancy_std": 200.0,
        "aggregation": "log_ratio_residual",
    }])
    out = metric.compute({"demand_db": str(sim)})
    eps = 1e-9
    expected = np.array([
        np.log((800 + eps) / (1000 + eps)),
        np.log((250 + eps) / (200 + eps)),
    ])
    np.testing.assert_allclose(out, expected, atol=1e-9)


def test_missing_sim_row_under_log_ratio_uses_epsilon(tmp_path: Path) -> None:
    """Sim missing a target-positive key produces a finite (very negative)
    log ratio, not −inf. The epsilon floor is what does this."""
    target_csv = tmp_path / "t.csv"
    _write_csv(target_csv, "agency,type,boardings",
               ["A,bus,100", "A,rail,50"])
    sim = tmp_path / "sim.sqlite"
    conn = sqlite3.connect(str(sim))
    conn.execute("CREATE TABLE Boarding (agency TEXT, type TEXT, n INTEGER)")
    conn.execute("INSERT INTO Boarding VALUES ('A', 'bus', 100)")
    conn.commit()
    conn.close()
    metric = MomentSetMetric(moments=[{
        "name": "b",
        "source_sql": "SELECT agency, type, SUM(n) AS boardings FROM Boarding GROUP BY agency, type",
        "target": str(target_csv),
        "target_key_cols": ["agency", "type"],
        "target_value_col": "boardings",
        "model_discrepancy_std": 10.0,
        "aggregation": "log_ratio_residual",
        "log_epsilon": 0.01,
    }])
    out = metric.compute({"demand_db": str(sim)})
    assert np.isfinite(out).all()
    # bus: log(100.01/100.01) ≈ 0; rail: log(0.01/50.01) = large negative
    assert abs(out[0]) < 1e-3
    assert out[1] < -5


def test_scalarize_none_is_default_and_returns_vector(tmp_path: Path) -> None:
    target_csv = tmp_path / "t.csv"
    _write_csv(target_csv, "purpose,mode,share", ["HBW,auto,0.5", "HBW,transit,0.5"])
    sim = tmp_path / "sim.sqlite"
    _write_demand_db(sim, [("HBW", "auto", 60), ("HBW", "transit", 40)])
    metric = MomentSetMetric(moments=[_mode_share_moment(target_csv)])
    assert metric.scalarize == "none"
    assert metric.n_objectives == 2


def test_scalarize_sum_squared_weighted(tmp_path: Path) -> None:
    target_csv = tmp_path / "t.csv"
    _write_csv(target_csv, "purpose,mode,share", ["HBW,auto,0.5", "HBW,transit,0.5"])
    sim = tmp_path / "sim.sqlite"
    _write_demand_db(sim, [("HBW", "auto", 60), ("HBW", "transit", 40)])
    m = {**_mode_share_moment(target_csv), "weight_per_element": 2.0}
    metric = MomentSetMetric(moments=[m], scalarize="sum_squared_weighted")
    out = metric.compute({"demand_db": str(sim)})
    assert metric.n_objectives == 1
    assert out.shape == (1,)
    # v0.36: standard WLS is sum(w * r^2), i.e. LINEAR in w.
    # r = +/-0.1, w = 2  ->  2*(0.01) + 2*(0.01) = 0.04.
    # This previously asserted 0.08 because the weight was folded into
    # the residual and then squared, giving sum(w^2 * r^2) — so a user
    # asking for 3x influence silently got 9x.
    assert float(out[0]) == pytest.approx(0.04, abs=1e-12)


def test_scalarize_max_implausibility(tmp_path: Path) -> None:
    target_csv = tmp_path / "t.csv"
    _write_csv(target_csv, "purpose,mode,share", ["HBW,auto,0.5", "HBW,transit,0.5"])
    sim = tmp_path / "sim.sqlite"
    _write_demand_db(sim, [("HBW", "auto", 60), ("HBW", "transit", 40)])
    # obs=0.005, md=0.02 → denom = sqrt(0.005^2 + 0.02^2) ≈ 0.02062
    metric = MomentSetMetric(
        moments=[_mode_share_moment(target_csv)],
        scalarize="max_implausibility",
    )
    out = metric.compute({"demand_db": str(sim)})
    denom = np.sqrt(0.005**2 + 0.02**2)
    expected = 0.1 / denom  # both residuals equal in magnitude
    assert float(out[0]) == pytest.approx(expected, rel=1e-9)


def test_scalarize_mean_implausibility(tmp_path: Path) -> None:
    target_csv = tmp_path / "t.csv"
    _write_csv(target_csv, "purpose,mode,share", ["HBW,auto,0.6", "HBW,transit,0.4"])
    sim = tmp_path / "sim.sqlite"
    _write_demand_db(sim, [("HBW", "auto", 70), ("HBW", "transit", 30)])
    # residuals: 0.1, -0.1; abs both 0.1; mean = 0.1 / denom
    metric = MomentSetMetric(
        moments=[_mode_share_moment(target_csv)],
        scalarize="mean_implausibility",
    )
    out = metric.compute({"demand_db": str(sim)})
    denom = np.sqrt(0.005**2 + 0.02**2)
    assert float(out[0]) == pytest.approx(0.1 / denom, rel=1e-9)


def test_scalarize_implausibility_requires_positive_denom(tmp_path: Path) -> None:
    """Cannot compute implausibility with obs=md=0 — construction should
    reject even though the moment itself is otherwise valid."""
    target_csv = tmp_path / "t.csv"
    _write_csv(target_csv, "purpose,mode,share", ["HBW,auto,0.5", "HBW,transit,0.5"])
    m = {
        "name": "hbw",
        "source_sql": _mode_share_sql(),
        "target": str(target_csv),
        "target_key_cols": ["purpose", "mode"],
        "target_value_col": "share",
        # obs=md=0 → denominator 0 → implausibility undefined
    }
    with (
        pytest.raises(ValueError, match="positive obs_noise_std or model_discrepancy_std"),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore", UserWarning)
        MomentSetMetric(moments=[m], scalarize="max_implausibility")


def test_missing_discrepancy_emits_warning(tmp_path: Path) -> None:
    """Vernon 2010 §3.5: silent md=0 produces empty NROY. Warn loudly."""
    target_csv = tmp_path / "t.csv"
    _write_csv(target_csv, "purpose,mode,share", ["HBW,auto,1.0"])
    m = {
        "name": "hbw",
        "source_sql": _mode_share_sql(),
        "target": str(target_csv),
        "target_key_cols": ["purpose", "mode"],
        "target_value_col": "share",
        "obs_noise_std": 0.01,
        # model_discrepancy_std absent → warn
    }
    with pytest.warns(UserWarning, match="model_discrepancy_std"):
        MomentSetMetric(moments=[m])


def test_no_warning_when_discrepancy_positive_on_all(tmp_path: Path) -> None:
    target_csv = tmp_path / "t.csv"
    _write_csv(target_csv, "purpose,mode,share", ["HBW,auto,1.0"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        MomentSetMetric(moments=[_mode_share_moment(target_csv)])


def test_metadata_vectors_align_with_slices(tmp_path: Path) -> None:
    """History-matching consumers need per-element obs/md/weight aligned
    to moment_slices. Verify the layout."""
    t1 = tmp_path / "t1.csv"
    _write_csv(t1, "purpose,mode,share", ["HBW,auto,0.5", "HBW,transit,0.5"])
    t2 = tmp_path / "t2.csv"
    _write_csv(t2, "purpose,mode,share", ["HBO,auto,0.7", "HBO,bike,0.3"])
    metric = MomentSetMetric(
        moments=[
            {**_mode_share_moment(t1), "name": "hbw",
             "obs_noise_std": 0.005, "model_discrepancy_std": 0.02,
             "weight_per_element": 1.0},
            {**_mode_share_moment(t2), "name": "hbo",
             "obs_noise_std": 0.010, "model_discrepancy_std": 0.05,
             "weight_per_element": 2.0},
        ],
    )
    sl_hbw, sl_hbo = metric.moment_slices["hbw"], metric.moment_slices["hbo"]
    np.testing.assert_allclose(metric.obs_noise_std_vector[sl_hbw], [0.005, 0.005])
    np.testing.assert_allclose(metric.obs_noise_std_vector[sl_hbo], [0.010, 0.010])
    np.testing.assert_allclose(metric.model_discrepancy_std_vector[sl_hbw], [0.02, 0.02])
    np.testing.assert_allclose(metric.model_discrepancy_std_vector[sl_hbo], [0.05, 0.05])
    np.testing.assert_allclose(metric.weight_vector[sl_hbw], [1.0, 1.0])
    np.testing.assert_allclose(metric.weight_vector[sl_hbo], [2.0, 2.0])


def test_rejects_empty_moments_list(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one moment"):
        MomentSetMetric(moments=[])


def test_rejects_unknown_aggregation(tmp_path: Path) -> None:
    target_csv = tmp_path / "t.csv"
    _write_csv(target_csv, "purpose,mode,share", ["HBW,auto,1.0"])
    with pytest.raises(ValueError, match="unknown aggregation"):
        MomentSetMetric(moments=[{
            **_mode_share_moment(target_csv),
            "aggregation": "elementwise_squared",
        }])


def test_rejects_unknown_scalarize(tmp_path: Path) -> None:
    target_csv = tmp_path / "t.csv"
    _write_csv(target_csv, "purpose,mode,share", ["HBW,auto,1.0"])
    with pytest.raises(ValueError, match="unknown scalarize"):
        MomentSetMetric(moments=[_mode_share_moment(target_csv)], scalarize="softmax")


def test_rejects_missing_required_field(tmp_path: Path) -> None:
    target_csv = tmp_path / "t.csv"
    _write_csv(target_csv, "purpose,mode,share", ["HBW,auto,1.0"])
    with pytest.raises(ValueError, match="missing required field"):
        MomentSetMetric(moments=[{"name": "x"}])  # missing source_sql, target, keys, val col


def test_rejects_negative_or_nonfinite_scalars(tmp_path: Path) -> None:
    target_csv = tmp_path / "t.csv"
    _write_csv(target_csv, "purpose,mode,share", ["HBW,auto,1.0"])
    for bad_field, bad_value in (
        ("obs_noise_std", -1e-3),
        ("obs_noise_std", float("nan")),
        ("model_discrepancy_std", float("inf")),
        ("weight_per_element", 0),
        ("weight_per_element", -1.0),
        ("log_epsilon", 0),
    ):
        m = {**_mode_share_moment(target_csv), bad_field: bad_value}
        with pytest.raises(ValueError, match=bad_field):
            MomentSetMetric(moments=[m])


def test_missing_target_csv_raises_metric_error(tmp_path: Path) -> None:
    m = {
        "name": "x",
        "source_sql": _mode_share_sql(),
        "target": str(tmp_path / "does-not-exist.csv"),
        "target_key_cols": ["purpose", "mode"],
        "target_value_col": "share",
        "model_discrepancy_std": 0.01,
    }
    with pytest.raises(MetricError, match="target CSV not found"):
        MomentSetMetric(moments=[m])


def test_target_csv_missing_column_raises(tmp_path: Path) -> None:
    target_csv = tmp_path / "t.csv"
    _write_csv(target_csv, "purpose,mode,not_share_col", ["HBW,auto,1.0"])
    with pytest.raises(MetricError, match="missing column"):
        MomentSetMetric(moments=[_mode_share_moment(target_csv)])


def test_target_csv_duplicate_keys_raise(tmp_path: Path) -> None:
    target_csv = tmp_path / "t.csv"
    _write_csv(target_csv, "purpose,mode,share",
               ["HBW,auto,0.6", "HBW,auto,0.7"])
    with pytest.raises(MetricError, match="duplicate"):
        MomentSetMetric(moments=[_mode_share_moment(target_csv)])


def test_missing_source_key_raises_at_compute(tmp_path: Path) -> None:
    target_csv = tmp_path / "t.csv"
    _write_csv(target_csv, "purpose,mode,share", ["HBW,auto,1.0"])
    metric = MomentSetMetric(moments=[_mode_share_moment(target_csv)])
    with pytest.raises(MetricError, match="demand_db"):
        metric.compute({})


def test_sql_missing_columns_raises_at_compute(tmp_path: Path) -> None:
    target_csv = tmp_path / "t.csv"
    _write_csv(target_csv, "purpose,mode,share", ["HBW,auto,1.0"])
    sim = tmp_path / "sim.sqlite"
    _write_demand_db(sim, [("HBW", "auto", 60)])
    # SQL returns 'mode_col' instead of the declared 'mode' — should fail cleanly
    m = {
        **_mode_share_moment(target_csv),
        "source_sql": "SELECT purpose, mode AS mode_col, 1.0 AS share FROM Trip",
    }
    metric = MomentSetMetric(moments=[m])
    with pytest.raises(MetricError, match="must return columns"):
        metric.compute({"demand_db": str(sim)})


def test_factory_roundtrip(tmp_path: Path) -> None:
    """Round-trip through metric_registry / make_metric."""
    target_csv = tmp_path / "t.csv"
    _write_csv(target_csv, "purpose,mode,share", ["HBW,auto,1.0"])
    metric = make_metric({
        "type": "moment_set",
        "options": {
            "moments": [_mode_share_moment(target_csv)],
            "scalarize": "sum_squared_weighted",
        },
    })
    assert isinstance(metric, MomentSetMetric)
    assert metric.scalarize == "sum_squared_weighted"
    assert metric.n_objectives == 1
