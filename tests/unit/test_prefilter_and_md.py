"""Unit tests for v0.33 P1 (LHS analytical_prefilter) and P2
(model_discrepancy_std='auto' + calibrate-md CLI)."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from click.testing import CliRunner

from polarisopt.cli import cli
from polarisopt.config import load_study_config
from polarisopt.design import LHSDesign
from polarisopt.metrics import MomentSetMetric
from polarisopt.metrics.moment_set import is_md_auto
from polarisopt.parameters import Parameter, ParameterSpace
from polarisopt.samples.sample import Sample, SampleStatus
from polarisopt.samples.store import SampleStore
from polarisopt.studies.md_calibration import (
    MdCalibrationError,
    calibrate_md_from_store,
    calibrated_yaml_snippet,
    format_md_report,
)
from polarisopt.utils.paths import workspace_layout

# ----- helpers -----


def _write_csv(path: Path, header: str, rows: list[str]) -> None:
    path.write_text(header + "\n" + "\n".join(rows) + "\n")


def _space_3d() -> ParameterSpace:
    return ParameterSpace.from_iterable([
        Parameter("x0", "a.json", 0.0, 1.0),
        Parameter("x1", "a.json", 0.0, 1.0),
        Parameter("x2", "a.json", 0.0, 1.0),
    ])


# ----- P1: LHSDesign(analytical_prefilter=...) -----


def _write_prefilter_module(path: Path, body: str) -> Path:
    """Write a Python module containing the callable used by the prefilter."""
    path.write_text(dedent(body))
    return path


def test_prefilter_rejects_soft_via_threshold(tmp_path: Path) -> None:
    """The callable returns a max_share_dev score; polarisopt applies
    the ``reject_if_max_share_dev_gt`` threshold."""
    module = _write_prefilter_module(tmp_path / "pf.py", """
        def score_candidate(theta):
            # x0 controls fitness: small x0 = feasible, large x0 = infeasible.
            return {"max_share_dev": theta["x0"]}
    """)
    design = LHSDesign(
        n=10, scramble=False,
        analytical_prefilter={
            "module": str(module),
            "function": "score_candidate",
            "reject_if_max_share_dev_gt": 0.3,
            "oversample_factor": 10,
        },
    )
    pts = design.generate(_space_3d(), rng=np.random.default_rng(0))
    assert pts.shape == (10, 3)
    # Every retained point has x0 ≤ 0.3.
    assert np.all(pts[:, 0] <= 0.3 + 1e-9)


def test_prefilter_hard_reject_removes_candidate(tmp_path: Path) -> None:
    """The callable can hard-reject via ``{'reject': True, 'reason': ...}``."""
    module = _write_prefilter_module(tmp_path / "pf.py", """
        def score(theta):
            if theta["x1"] > 0.7:
                return {"reject": True, "reason": "x1 too big"}
            return {"max_share_dev": 0.01}
    """)
    design = LHSDesign(
        n=5, scramble=False,
        analytical_prefilter={
            "module": str(module),
            "function": "score",
            "oversample_factor": 20,
        },
    )
    pts = design.generate(_space_3d(), rng=np.random.default_rng(0))
    assert pts.shape[0] <= 5
    assert np.all(pts[:, 1] <= 0.7 + 1e-9)


def test_prefilter_raises_when_all_rejected(tmp_path: Path) -> None:
    """If the callable rejects everything, polarisopt raises with a
    hint to widen the threshold — better than crashing later in POLARIS."""
    module = _write_prefilter_module(tmp_path / "pf.py", """
        def score(theta):
            return {"reject": True, "reason": "always reject"}
    """)
    design = LHSDesign(
        n=5,
        analytical_prefilter={
            "module": str(module), "function": "score", "oversample_factor": 5,
        },
    )
    with pytest.raises(ValueError, match="rejected ALL"):
        design.generate(_space_3d(), rng=np.random.default_rng(0))


def test_prefilter_warns_when_survivors_below_n(tmp_path: Path, caplog) -> None:
    """When survivors < n, polarisopt returns what it has and logs a warning.
    Better than blocking the wave."""
    import logging

    module = _write_prefilter_module(tmp_path / "pf.py", """
        def score(theta):
            if theta["x0"] > 0.1:
                return {"reject": True}
            return {"max_share_dev": 0.0}
    """)
    design = LHSDesign(
        n=50, scramble=False,
        analytical_prefilter={
            "module": str(module), "function": "score", "oversample_factor": 5,
        },
    )
    with caplog.at_level(logging.WARNING, logger="polarisopt.design.lhs"):
        pts = design.generate(_space_3d(), rng=np.random.default_rng(0))
    assert pts.shape[0] < 50
    assert any("only" in r.message and "survivors" in r.message for r in caplog.records)


def test_prefilter_import_error_raises_at_construction(tmp_path: Path) -> None:
    """A missing module surfaces at construction, not at generate() time."""
    with pytest.raises(ValueError, match="module file does not exist"):
        LHSDesign(
            n=5,
            analytical_prefilter={
                "module": str(tmp_path / "does-not-exist.py"),
                "function": "score",
            },
        )


def test_prefilter_missing_function_raises_at_generate(tmp_path: Path) -> None:
    """A module without the named function surfaces at generate() (import time)."""
    module = _write_prefilter_module(tmp_path / "pf.py", "# empty module\n")
    design = LHSDesign(
        n=3,
        analytical_prefilter={
            "module": str(module), "function": "no_such_function",
        },
    )
    with pytest.raises(ValueError, match="not a callable"):
        design.generate(_space_3d(), rng=np.random.default_rng(0))


def test_prefilter_off_by_default_regenerates_full_batch() -> None:
    """No prefilter → all N LHS points come through unfiltered."""
    design = LHSDesign(n=8)
    pts = design.generate(_space_3d(), rng=np.random.default_rng(0))
    assert pts.shape == (8, 3)


def test_prefilter_rejects_bad_oversample_at_construction(tmp_path: Path) -> None:
    module = _write_prefilter_module(tmp_path / "pf.py", "def score(t): return {}")
    with pytest.raises(ValueError, match="oversample_factor"):
        LHSDesign(
            n=3,
            analytical_prefilter={
                "module": str(module), "function": "score", "oversample_factor": 0,
            },
        )


def test_prefilter_rejects_bad_threshold_at_construction(tmp_path: Path) -> None:
    module = _write_prefilter_module(tmp_path / "pf.py", "def score(t): return {}")
    with pytest.raises(ValueError, match="reject_if_max_share_dev_gt"):
        LHSDesign(
            n=3,
            analytical_prefilter={
                "module": str(module),
                "function": "score",
                "reject_if_max_share_dev_gt": -0.01,
            },
        )


# ----- P2 part 1: MomentSetMetric accepts model_discrepancy_std='auto' -----


def _basic_target(path: Path) -> Path:
    _write_csv(path, "purpose,mode,share",
               ["HBW,auto,0.5", "HBW,transit,0.3", "HBW,walk,0.2"])
    return path


def test_moment_set_accepts_md_auto_sentinel(tmp_path: Path) -> None:
    target = _basic_target(tmp_path / "t.csv")
    with warnings.catch_warnings():
        # UserWarning is EXPECTED here — we're using 'auto'.
        warnings.simplefilter("ignore", UserWarning)
        metric = MomentSetMetric(moments=[{
            "name": "shares",
            "source_sql": "SELECT purpose, mode, 1.0 AS share FROM Trip",
            "target": str(target),
            "target_key_cols": ["purpose", "mode"],
            "target_value_col": "share",
            "obs_noise_std": 0.005,
            "model_discrepancy_std": "auto",
        }])
    assert is_md_auto(metric.moments[0])
    # metadata vector holds NaN for auto entries
    assert np.all(np.isnan(metric.model_discrepancy_std_vector))


def test_moment_set_md_auto_case_insensitive(tmp_path: Path) -> None:
    target = _basic_target(tmp_path / "t.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        metric = MomentSetMetric(moments=[{
            "name": "shares",
            "source_sql": "SELECT purpose, mode, 1.0 AS share FROM Trip",
            "target": str(target),
            "target_key_cols": ["purpose", "mode"],
            "target_value_col": "share",
            "model_discrepancy_std": "AUTO",
        }])
    assert is_md_auto(metric.moments[0])


def test_moment_set_md_auto_warns_user(tmp_path: Path) -> None:
    target = _basic_target(tmp_path / "t.csv")
    with pytest.warns(UserWarning, match="calibrate-md"):
        MomentSetMetric(moments=[{
            "name": "shares",
            "source_sql": "SELECT purpose, mode, 1.0 AS share FROM Trip",
            "target": str(target),
            "target_key_cols": ["purpose", "mode"],
            "target_value_col": "share",
            "model_discrepancy_std": "auto",
        }])


def test_moment_set_md_auto_incompatible_with_implausibility(tmp_path: Path) -> None:
    """max_implausibility needs a scalar md — 'auto' means "unknown yet"
    and would produce NaN implausibility. Reject at construction."""
    target = _basic_target(tmp_path / "t.csv")
    with (
        pytest.raises(ValueError, match="incompatible with"),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore", UserWarning)
        MomentSetMetric(
            moments=[{
                "name": "shares",
                "source_sql": "SELECT purpose, mode, 1.0 AS share FROM Trip",
                "target": str(target),
                "target_key_cols": ["purpose", "mode"],
                "target_value_col": "share",
                "obs_noise_std": 0.005,
                "model_discrepancy_std": "auto",
            }],
            scalarize="max_implausibility",
        )


def test_moment_set_rejects_bad_md_string(tmp_path: Path) -> None:
    """Only the exact string 'auto' (case-insensitive) is accepted;
    other strings are rejected."""
    target = _basic_target(tmp_path / "t.csv")
    with pytest.raises(ValueError, match="model_discrepancy_std"):
        MomentSetMetric(moments=[{
            "name": "shares",
            "source_sql": "SELECT purpose, mode, 1.0 AS share FROM Trip",
            "target": str(target),
            "target_key_cols": ["purpose", "mode"],
            "target_value_col": "share",
            "model_discrepancy_std": "adaptive",
        }])


# ----- P2 part 2: calibrate_md_from_store + CLI -----


def _seed_finished_samples(
    workspace: Path, name: str, target_csv: Path, *, n: int = 20,
) -> SampleStore:
    """Populate store with FINISHED samples whose residuals depend on x0."""
    n_moments = sum(1 for _ in target_csv.open()) - 1
    workspace.mkdir(parents=True, exist_ok=True)
    layout = workspace_layout(workspace)
    store = SampleStore.open(layout["db"], name)
    rng = np.random.default_rng(0)
    for _ in range(n):
        x = rng.uniform(size=3)
        # Residual driven by x0; extra noise emulates residual variance
        # beyond the GP's ability to explain it → attributes to md.
        residuals = np.full(n_moments, 0.5 * (x[0] - 0.5)) + 0.05 * rng.standard_normal(n_moments)
        s = store.add(Sample(phase="wave-1", inputs=x))
        s.status = SampleStatus.FINISHED
        s.metric = residuals
        store.update(s)
    return store


def test_calibrate_md_returns_positive_estimate(tmp_path: Path) -> None:
    target = _basic_target(tmp_path / "t.csv")
    store = _seed_finished_samples(tmp_path / "ws", "md-test", target, n=20)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        metric = MomentSetMetric(moments=[{
            "name": "shares",
            "source_sql": "SELECT purpose, mode, 1.0 AS share FROM Trip",
            "target": str(target),
            "target_key_cols": ["purpose", "mode"],
            "target_value_col": "share",
            "obs_noise_std": 0.005,
            "model_discrepancy_std": "auto",
        }])
    estimates = calibrate_md_from_store(store, _space_3d(), metric)
    assert len(estimates) == 1
    assert estimates[0].moment_name == "shares"
    # Empirical md should be positive (residuals have real variance).
    assert estimates[0].empirical_md_std > 0
    # residual_std should exceed obs_std since we injected extra noise.
    assert np.sqrt(estimates[0].residual_var) > np.sqrt(estimates[0].obs_var)


def test_calibrate_md_only_auto_skips_numeric_moments(tmp_path: Path) -> None:
    """Moments with numeric md are left alone under --only-auto."""
    _write_csv(tmp_path / "t1.csv", "purpose,mode,share",
               ["HBW,auto,0.5", "HBW,transit,0.5"])
    _write_csv(tmp_path / "t2.csv", "purpose,mode,share",
               ["HBW,auto,0.5", "HBW,transit,0.5"])
    workspace = tmp_path / "ws"
    workspace.mkdir()
    layout = workspace_layout(workspace)
    store = SampleStore.open(layout["db"], "mix")
    rng = np.random.default_rng(0)
    for _ in range(15):
        x = rng.uniform(size=3)
        s = store.add(Sample(phase="w", inputs=x))
        s.status = SampleStatus.FINISHED
        s.metric = np.array([0.1 * x[0], 0.1 * x[0], 0.1 * x[0], 0.1 * x[0]])
        store.update(s)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        metric = MomentSetMetric(moments=[
            {
                "name": "auto_moment",
                "source_sql": "SELECT purpose, mode, 1.0 AS share FROM Trip",
                "target": str(tmp_path / "t1.csv"),
                "target_key_cols": ["purpose", "mode"],
                "target_value_col": "share",
                "model_discrepancy_std": "auto",
            },
            {
                "name": "fixed_moment",
                "source_sql": "SELECT purpose, mode, 1.0 AS share FROM Trip",
                "target": str(tmp_path / "t2.csv"),
                "target_key_cols": ["purpose", "mode"],
                "target_value_col": "share",
                "model_discrepancy_std": 0.02,
            },
        ])
    est_only_auto = calibrate_md_from_store(store, _space_3d(), metric, only_auto=True)
    est_all = calibrate_md_from_store(store, _space_3d(), metric, only_auto=False)
    assert [e.moment_name for e in est_only_auto] == ["auto_moment"]
    assert {e.moment_name for e in est_all} == {"auto_moment", "fixed_moment"}


def test_calibrate_md_rejects_scalar_metric_widths(tmp_path: Path) -> None:
    target = _basic_target(tmp_path / "t.csv")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    layout = workspace_layout(workspace)
    store = SampleStore.open(layout["db"], "widthmix")
    for i in range(5):
        s = store.add(Sample(phase="w", inputs=np.array([0.1 * i, 0.5, 0.3])))
        s.status = SampleStatus.FINISHED
        s.metric = np.zeros(3 if i % 2 == 0 else 2)
        store.update(s)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        metric = MomentSetMetric(moments=[{
            "name": "shares",
            "source_sql": "SELECT purpose, mode, 1.0 AS share FROM Trip",
            "target": str(target),
            "target_key_cols": ["purpose", "mode"],
            "target_value_col": "share",
            "model_discrepancy_std": "auto",
        }])
    with pytest.raises(MdCalibrationError, match="inconsistent"):
        calibrate_md_from_store(store, _space_3d(), metric)


def test_calibrate_md_rejects_too_few_samples(tmp_path: Path) -> None:
    target = _basic_target(tmp_path / "t.csv")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    layout = workspace_layout(workspace)
    store = SampleStore.open(layout["db"], "small")
    for i in range(2):
        s = store.add(Sample(phase="w", inputs=np.array([0.1 * i, 0.5, 0.3])))
        s.status = SampleStatus.FINISHED
        s.metric = np.zeros(3)
        store.update(s)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        metric = MomentSetMetric(moments=[{
            "name": "shares",
            "source_sql": "SELECT purpose, mode, 1.0 AS share FROM Trip",
            "target": str(target),
            "target_key_cols": ["purpose", "mode"],
            "target_value_col": "share",
            "model_discrepancy_std": "auto",
        }])
    with pytest.raises(MdCalibrationError, match="at least 3"):
        calibrate_md_from_store(store, _space_3d(), metric)


def test_calibrated_yaml_snippet_shape() -> None:
    from polarisopt.studies.md_calibration import MdEstimate

    snippet = calibrated_yaml_snippet([
        MdEstimate(
            moment_name="mode_shares",
            empirical_md_std=0.0234,
            user_md_std=None, residual_var=0.001, obs_var=0.0001,
            emulator_var_mean=0.0001, n_samples=20,
        ),
    ])
    assert "metric:" in snippet
    assert "options:" in snippet
    assert "moments:" in snippet
    assert "mode_shares" in snippet
    assert "0.0234" in snippet


def test_format_md_report_flags_understated_md() -> None:
    from polarisopt.studies.md_calibration import MdEstimate

    text = format_md_report([
        MdEstimate(
            moment_name="under",
            empirical_md_std=0.09, user_md_std=0.02,  # 4.5× user
            residual_var=0.01, obs_var=0.001,
            emulator_var_mean=0.001, n_samples=30,
        ),
    ])
    assert "under" in text
    # 0.09 / 0.02 = 4.5× → "empirical > 3× user"
    assert "empirical > 3×" in text


# ----- calibrate-md CLI -----


def _write_yaml(workspace: Path, target_csv: Path) -> str:
    return dedent(f"""
        name: mdcli-{workspace.name}
        workspace: {workspace}
        simulator:
          type: mock
          options: {{ function: quadratic }}
        runner:
          type: local
          options: {{}}
        parameters:
          inline:
            - {{ name: x0, file: a.json, min: 0.0, max: 1.0 }}
            - {{ name: x1, file: a.json, min: 0.0, max: 1.0 }}
            - {{ name: x2, file: a.json, min: 0.0, max: 1.0 }}
        metric:
          type: moment_set
          options:
            source_key: demand_db
            scalarize: none
            moments:
              - name: shares
                source_sql: "SELECT purpose, mode, 1.0 AS share FROM Trip"
                target: {target_csv}
                target_key_cols: [purpose, mode]
                target_value_col: share
                obs_noise_std: 0.005
                model_discrepancy_std: auto
        phases:
          - name: warmup
            type: static
            design:
              type: manual
              options:
                points: [[0.5, 0.5, 0.5]]
    """)


def test_cli_calibrate_md_prints_report_and_snippet(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    target_csv = _basic_target(tmp_path / "t.csv")
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_write_yaml(workspace, target_csv))
    cfg = load_study_config(cfg_path)
    _seed_finished_samples(workspace, cfg.name, target_csv, n=20)
    res = CliRunner().invoke(cli, ["calibrate-md", str(cfg_path)])
    assert res.exit_code == 0, res.output
    assert "Model-discrepancy calibration" in res.output
    assert "shares" in res.output
    assert "metric:" in res.output  # snippet


def test_cli_calibrate_md_writes_out_file(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    target_csv = _basic_target(tmp_path / "t.csv")
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_write_yaml(workspace, target_csv))
    cfg = load_study_config(cfg_path)
    _seed_finished_samples(workspace, cfg.name, target_csv, n=20)
    out_file = tmp_path / "calibrated.yaml"
    res = CliRunner().invoke(
        cli, ["calibrate-md", str(cfg_path), "--out", str(out_file)],
    )
    assert res.exit_code == 0, res.output
    assert out_file.exists()
    body = out_file.read_text()
    assert "shares" in body
    assert "model_discrepancy_std" in body


def test_cli_calibrate_md_json(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    target_csv = _basic_target(tmp_path / "t.csv")
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_write_yaml(workspace, target_csv))
    cfg = load_study_config(cfg_path)
    _seed_finished_samples(workspace, cfg.name, target_csv, n=20)
    res = CliRunner().invoke(cli, ["calibrate-md", str(cfg_path), "--json"])
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["moments"][0]["name"] == "shares"
    assert payload["moments"][0]["empirical_md_std"] > 0


def test_cli_calibrate_md_rejects_scalar_metric(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(dedent(f"""
        name: scalar-{workspace.name}
        workspace: {workspace}
        simulator: {{ type: mock, options: {{ function: quadratic }} }}
        runner: {{ type: local, options: {{}} }}
        parameters:
          inline:
            - {{ name: x, file: a.json, min: 0.0, max: 1.0 }}
        metric:
          type: identity
          options: {{ keys: value }}
        phases:
          - name: warmup
            type: static
            design:
              type: manual
              options:
                points: [[0.5]]
    """))
    res = CliRunner().invoke(cli, ["calibrate-md", str(cfg_path)])
    assert res.exit_code != 0
    assert "moment_set metric" in res.output


def test_cli_calibrate_md_says_nothing_to_do_when_no_auto(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    target_csv = _basic_target(tmp_path / "t.csv")
    cfg_path = tmp_path / "c.yaml"
    # numeric md → nothing marked auto → --only-auto has no work.
    cfg_path.write_text(_write_yaml(workspace, target_csv).replace(
        "model_discrepancy_std: auto", "model_discrepancy_std: 0.02",
    ))
    cfg = load_study_config(cfg_path)
    _seed_finished_samples(workspace, cfg.name, target_csv, n=20)
    res = CliRunner().invoke(cli, ["calibrate-md", str(cfg_path)])
    assert res.exit_code != 0
    assert "nothing to calibrate" in res.output
