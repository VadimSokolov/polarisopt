"""Tests for studies.diff — side-by-side study comparison."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import numpy as np
from click.testing import CliRunner

from polarisopt.cli import cli
from polarisopt.config import load_study_config
from polarisopt.samples.sample import Sample, SampleStatus
from polarisopt.samples.store import SampleStore
from polarisopt.studies.diff import diff_studies
from polarisopt.utils.paths import workspace_layout


def _yaml(workspace: Path, study_name: str) -> str:
    return dedent(
        f"""\
        name: {study_name}
        workspace: {workspace}
        simulator: {{ type: mock, options: {{ function: quadratic }} }}
        runner: {{ type: local, options: {{}} }}
        parameters:
          inline:
            - {{ name: x, file: a.json, min: 0.0, max: 1.0 }}
        metric: {{ type: identity, options: {{ keys: value }} }}
        phases:
          - name: p
            type: static
            design: {{ type: manual, options: {{ points: [[0.5]] }} }}
        """
    )


def _seed(workspace: Path, name: str, metrics: list[list[float] | None], failed_count: int = 0) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    store = SampleStore.open(workspace_layout(workspace)["db"], name)
    for i, m in enumerate(metrics):
        s = store.add(Sample(phase="p", inputs=np.array([0.1 * i])))
        if m is None:
            s.status = SampleStatus.FAILED if failed_count > 0 else SampleStatus.PENDING
            if failed_count > 0:
                failed_count -= 1
        else:
            s.status = SampleStatus.FINISHED
            s.metric = np.asarray(m, dtype=float)
        store.update(s)


def test_diff_single_objective(tmp_path: Path) -> None:
    cfg_a_path = tmp_path / "a.yaml"
    cfg_b_path = tmp_path / "b.yaml"
    ws_a = tmp_path / "ws_a"
    ws_b = tmp_path / "ws_b"
    cfg_a_path.write_text(_yaml(ws_a, "study-a"))
    cfg_b_path.write_text(_yaml(ws_b, "study-b"))

    _seed(ws_a, "study-a", [[1.0], [0.5], [0.25]])
    _seed(ws_b, "study-b", [[2.0], [0.1], [0.05], [0.01]])

    d = diff_studies(cfg_a_path, cfg_b_path)
    assert d.name_a == "study-a"
    assert d.name_b == "study-b"
    assert d.samples == (3, 4)
    assert d.finished == (3, 4)
    assert d.best_metric[0] == [0.25]
    assert d.best_metric[1] == [0.01]
    assert d.n_objectives == 1


def test_diff_multi_objective(tmp_path: Path) -> None:
    cfg_a_path = tmp_path / "a.yaml"
    cfg_b_path = tmp_path / "b.yaml"
    ws_a = tmp_path / "ws_a"
    ws_b = tmp_path / "ws_b"
    cfg_a_path.write_text(_yaml(ws_a, "mo-a"))
    cfg_b_path.write_text(_yaml(ws_b, "mo-b"))

    _seed(ws_a, "mo-a", [[1.0, 5.0], [3.0, 2.0]])
    _seed(ws_b, "mo-b", [[0.5, 4.0], [2.0, 1.0], [3.0, 0.5]])

    d = diff_studies(cfg_a_path, cfg_b_path)
    assert d.n_objectives == 2
    assert d.pareto_size == (2, 3)
    # best per objective = elementwise min
    assert d.best_metric[0] == [1.0, 2.0]
    assert d.best_metric[1] == [0.5, 0.5]


def test_diff_counts_failures(tmp_path: Path) -> None:
    cfg_path = tmp_path / "a.yaml"
    ws = tmp_path / "ws"
    cfg_path.write_text(_yaml(ws, "f"))
    _seed(ws, "f", [[1.0], None, None], failed_count=2)

    d = diff_studies(cfg_path, cfg_path)  # compare to itself
    assert d.samples == (3, 3)
    assert d.finished == (1, 1)
    assert d.failed == (2, 2)


def test_cli_diff(tmp_path: Path) -> None:
    cfg_a_path = tmp_path / "a.yaml"
    cfg_b_path = tmp_path / "b.yaml"
    ws_a = tmp_path / "ws_a"
    ws_b = tmp_path / "ws_b"
    cfg_a_path.write_text(_yaml(ws_a, "study-a"))
    cfg_b_path.write_text(_yaml(ws_b, "study-b"))
    _seed(ws_a, "study-a", [[1.0]])
    _seed(ws_b, "study-b", [[0.5]])

    res = CliRunner().invoke(cli, ["diff", str(cfg_a_path), str(cfg_b_path)])
    assert res.exit_code == 0, res.output
    assert "study-a" in res.output and "study-b" in res.output
    assert "best metric" in res.output


def test_render_uses_study_names(tmp_path: Path) -> None:
    cfg_path = tmp_path / "c.yaml"
    ws = tmp_path / "ws"
    cfg_path.write_text(_yaml(ws, "named"))
    _seed(ws, "named", [[1.0]])

    cfg = load_study_config(cfg_path)
    d = diff_studies(cfg, cfg)
    out = d.render()
    assert "named" in out
