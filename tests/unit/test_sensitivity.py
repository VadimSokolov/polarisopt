"""Unit tests for the v0.24 sensitivity subcommand."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest
from click.testing import CliRunner

torch = pytest.importorskip("torch")

from polarisopt.cli import cli
from polarisopt.config import load_study_config
from polarisopt.parameters import Parameter, ParameterSpace
from polarisopt.samples.sample import Sample, SampleStatus
from polarisopt.samples.store import SampleStore
from polarisopt.studies.sensitivity import (
    SensitivityError,
    SensitivityReport,
    format_report,
    report_as_dict,
    run_sensitivity_analysis,
)
from polarisopt.utils.paths import workspace_layout


def _yaml(workspace: Path) -> str:
    """Small mock study over 3 params; only x0 is actually used by y."""
    return dedent(
        f"""\
        name: sens-{workspace.name}
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
          type: identity
          options: {{ keys: value }}
        phases:
          - name: warmup
            type: static
            design:
              type: manual
              options:
                points: [[0.5, 0.5, 0.5]]
        """
    )


def _seed_store(
    workspace: Path, name: str, n: int = 60, *, rng_seed: int = 0,
) -> SampleStore:
    """Populate the store with FINISHED samples where y depends only on x0."""
    workspace.mkdir(parents=True, exist_ok=True)
    layout = workspace_layout(workspace)
    store = SampleStore.open(layout["db"], name)
    rng = np.random.default_rng(rng_seed)
    X = rng.uniform(size=(n, 3))
    # y = (x0 - 0.3)^2 + 0.01 * noise, x1 / x2 irrelevant
    Y = (X[:, 0] - 0.3) ** 2 + 0.01 * rng.standard_normal(n)
    for x, y in zip(X, Y, strict=True):
        s = store.add(Sample(phase="warmup", inputs=x))
        s.status = SampleStatus.FINISHED
        s.metric = np.array([float(y)])
        store.update(s)
    return store


def _space_3d() -> ParameterSpace:
    return ParameterSpace.from_iterable(
        [
            Parameter("x0", "a.json", 0.0, 1.0),
            Parameter("x1", "a.json", 0.0, 1.0),
            Parameter("x2", "a.json", 0.0, 1.0),
        ]
    )


def test_run_sensitivity_ranks_active_dimension_first(tmp_path: Path) -> None:
    """The GP should learn only x0 matters; ST should rank it first and
    the inactive dimensions' length-scales should peg high."""
    store = _seed_store(tmp_path / "ws", "sens-test", n=60)
    report = run_sensitivity_analysis(
        store, _space_3d(), n_sobol=512,
    )
    ranked = report.ranked_by_st()
    top_name, top_s1, _s1c, top_st, _stc, top_ls = ranked[0]
    assert top_name == "x0"
    assert top_st > 0.5, ranked
    # Inactive dims: ST small AND length-scale much larger than x0.
    ls_by_name = dict(zip(report.parameter_names, report.length_scales, strict=True))
    assert ls_by_name["x1"] > ls_by_name["x0"], ls_by_name
    assert ls_by_name["x2"] > ls_by_name["x0"], ls_by_name


def test_run_sensitivity_rejects_multi_obj(tmp_path: Path) -> None:
    """Sensitivity is single-objective only; multi-obj samples should fail
    cleanly with a SensitivityError rather than crashing inside the GP."""
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True)
    layout = workspace_layout(workspace)
    store = SampleStore.open(layout["db"], "multi")
    for i in range(5):
        s = store.add(Sample(phase="w", inputs=np.array([0.1 * i, 0.2, 0.3])))
        s.status = SampleStatus.FINISHED
        s.metric = np.array([0.1, 0.2])  # multi-output metric
        store.update(s)
    with pytest.raises(SensitivityError, match="single-objective"):
        run_sensitivity_analysis(store, _space_3d(), n_sobol=64)


def test_run_sensitivity_rejects_empty_store(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True)
    layout = workspace_layout(workspace)
    store = SampleStore.open(layout["db"], "empty")
    with pytest.raises(SensitivityError, match="no FINISHED"):
        run_sensitivity_analysis(store, _space_3d(), n_sobol=64)


def test_run_sensitivity_rejects_shape_mismatch(tmp_path: Path) -> None:
    """A stored sample with a different ndim than the current ParameterSpace
    surfaces a clear error, not a shape crash inside the surrogate fit."""
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True)
    layout = workspace_layout(workspace)
    store = SampleStore.open(layout["db"], "mismatch")
    for i in range(5):
        s = store.add(Sample(phase="w", inputs=np.array([0.1 * i, 0.5])))  # only 2 dims
        s.status = SampleStatus.FINISHED
        s.metric = np.array([0.42])
        store.update(s)
    with pytest.raises(SensitivityError, match="ndim=2"):
        run_sensitivity_analysis(store, _space_3d(), n_sobol=64)


def test_format_report_lists_all_parameters() -> None:
    report = SensitivityReport(
        parameter_names=("a", "b"),
        s1=np.array([0.1, 0.9]),
        s1_conf=np.array([0.01, 0.01]),
        st=np.array([0.15, 0.95]),
        st_conf=np.array([0.02, 0.02]),
        length_scales=np.array([10.0, 0.3]),
        n_train=42,
        n_sobol=512,
    )
    text = format_report(report)
    assert "a" in text
    assert "b" in text
    # Ranked by ST desc — b (0.95) first, then a (0.15).
    b_pos = text.find("\n 1  b")
    a_pos = text.find("\n 2  a")
    assert b_pos > 0 and a_pos > b_pos, text


def test_report_as_dict_is_json_roundtrippable() -> None:
    report = SensitivityReport(
        parameter_names=("x",),
        s1=np.array([0.5]),
        s1_conf=np.array([0.05]),
        st=np.array([0.7]),
        st_conf=np.array([0.07]),
        length_scales=np.array([2.5]),
        n_train=10,
        n_sobol=128,
    )
    payload = report_as_dict(report)
    round = json.loads(json.dumps(payload))
    assert round["n_train"] == 10
    assert round["n_sobol"] == 128
    assert round["parameters"][0]["name"] == "x"


def test_cli_sensitivity_prints_ranked_table(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_yaml(workspace))
    cfg = load_study_config(cfg_path)
    _seed_store(workspace, cfg.name, n=50)
    res = CliRunner().invoke(
        cli, ["sensitivity", str(cfg_path), "--n-sobol", "256"],
    )
    assert res.exit_code == 0, res.output
    assert "GP-Sobol sensitivity" in res.output
    # x0 should be first in the ranked table.
    assert res.output.find(" 1  x0") > 0, res.output


def test_cli_sensitivity_json_output(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_yaml(workspace))
    cfg = load_study_config(cfg_path)
    _seed_store(workspace, cfg.name, n=50)
    res = CliRunner().invoke(
        cli, ["sensitivity", str(cfg_path), "--n-sobol", "256", "--json"],
    )
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["parameters"][0]["name"] == "x0"


def test_cli_sensitivity_empty_store_exits_cleanly(tmp_path: Path) -> None:
    """A store without FINISHED samples surfaces a ClickException, not a
    stack trace."""
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True)
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_yaml(workspace))
    # Store exists but is empty.
    cfg = load_study_config(cfg_path)
    layout = workspace_layout(workspace)
    SampleStore.open(layout["db"], cfg.name)
    res = CliRunner().invoke(cli, ["sensitivity", str(cfg_path)])
    assert res.exit_code != 0
    assert "no FINISHED" in res.output
