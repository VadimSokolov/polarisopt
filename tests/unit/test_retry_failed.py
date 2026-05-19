from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest
from click.testing import CliRunner

from polarisopt.cli import cli
from polarisopt.config import load_study_config
from polarisopt.samples.sample import Sample, SampleStatus
from polarisopt.samples.store import SampleStore
from polarisopt.studies.ops import retry_failed
from polarisopt.utils.paths import workspace_layout


def _yaml(workspace: Path) -> str:
    return dedent(
        f"""\
        name: retry-{workspace.name}
        workspace: {workspace}
        simulator:
          type: mock
          options: {{ function: quadratic }}
        runner:
          type: local
          options: {{}}
        parameters:
          inline:
            - {{ name: x, file: a.json, min: 0.0, max: 1.0 }}
        metric:
          type: identity
          options: {{ keys: value }}
        phases:
          - name: p
            type: static
            design:
              type: manual
              options:
                points: [[0.5]]
        """
    )


def _seed_with_failed(workspace: Path, name: str) -> tuple[SampleStore, list[int]]:
    workspace.mkdir(parents=True, exist_ok=True)
    layout = workspace_layout(workspace)
    store = SampleStore.open(layout["db"], name)
    ids = []
    for i, val in enumerate([0.1, 0.2, 0.3]):
        s = store.add(Sample(phase="p", inputs=np.array([val])))
        if i < 2:
            s.status = SampleStatus.FAILED
            s.message = f"boom #{i}"
            s.runner_task_id = f"stale-{i}"
            store.update(s)
        else:
            s.status = SampleStatus.FINISHED
            s.metric = np.array([0.09])
            store.update(s)
        ids.append(s.id)  # type: ignore[arg-type]
    return store, ids


def test_retry_failed_all(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_yaml(workspace))
    cfg = load_study_config(cfg_path)
    store, ids = _seed_with_failed(workspace, cfg.name)

    retried = retry_failed(cfg, store=store)
    assert len(retried) == 2
    for s in retried:
        assert s.status is SampleStatus.PENDING
        assert s.message is not None and "retry" in s.message
        assert s.runner_task_id is None  # stale id cleared

    # The previously-FINISHED sample is unchanged.
    assert store.get(ids[2]).status is SampleStatus.FINISHED


def test_retry_failed_specific_ids(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_yaml(workspace))
    cfg = load_study_config(cfg_path)
    store, ids = _seed_with_failed(workspace, cfg.name)

    # Only retry the first failed sample
    retried = retry_failed(cfg, sample_ids=[ids[0]], store=store)
    assert {s.id for s in retried} == {ids[0]}
    # The second failed sample is still FAILED
    assert store.get(ids[1]).status is SampleStatus.FAILED


def test_retry_failed_rejects_non_failed_ids(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_yaml(workspace))
    cfg = load_study_config(cfg_path)
    store, ids = _seed_with_failed(workspace, cfg.name)

    with pytest.raises(ValueError, match="are not FAILED"):
        retry_failed(cfg, sample_ids=[ids[2]], store=store)  # FINISHED


def test_retry_failed_with_no_failed_returns_empty(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_yaml(workspace))
    cfg = load_study_config(cfg_path)
    workspace.mkdir(parents=True)
    layout = workspace_layout(workspace)
    SampleStore.open(layout["db"], cfg.name)  # empty store

    retried = retry_failed(cfg)
    assert retried == []


def test_cli_retry_failed_no_run(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_yaml(workspace))
    cfg = load_study_config(cfg_path)
    _seed_with_failed(workspace, cfg.name)

    res = CliRunner().invoke(cli, ["retry-failed", str(cfg_path)])
    assert res.exit_code == 0, res.output
    assert "retried 2 sample(s)" in res.output


def test_cli_retry_failed_specific_id(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_yaml(workspace))
    cfg = load_study_config(cfg_path)
    _, ids = _seed_with_failed(workspace, cfg.name)

    res = CliRunner().invoke(cli, ["retry-failed", str(cfg_path), "--id", str(ids[0])])
    assert res.exit_code == 0, res.output
    assert f"[{ids[0]}]" in res.output


def test_cli_retry_failed_no_failures(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_yaml(workspace))
    cfg = load_study_config(cfg_path)
    layout = workspace_layout(workspace)
    SampleStore.open(layout["db"], cfg.name)

    res = CliRunner().invoke(cli, ["retry-failed", str(cfg_path)])
    assert res.exit_code == 0, res.output
    assert "no FAILED" in res.output
