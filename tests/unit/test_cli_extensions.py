"""Click invocations of the new cancel / abort / logs subcommands."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import numpy as np
from click.testing import CliRunner

from polarisopt.cli import cli
from polarisopt.samples.sample import Sample, SampleStatus
from polarisopt.samples.store import SampleStore
from polarisopt.utils.paths import workspace_layout


def _yaml(workspace: Path) -> str:
    return dedent(
        f"""\
        name: cli-ext-{workspace.name}
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


def _seed(tmp_path: Path) -> tuple[Path, int]:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    cfg = tmp_path / "c.yaml"
    cfg.write_text(_yaml(workspace))
    layout = workspace_layout(workspace)
    store = SampleStore.open(layout["db"], f"cli-ext-{workspace.name}")
    s = store.add(Sample(phase="p", inputs=np.array([0.5])))
    return cfg, s.id  # type: ignore[return-value]


def test_cancel_subcommand(tmp_path: Path) -> None:
    cfg_path, sid = _seed(tmp_path)
    res = CliRunner().invoke(cli, ["cancel", str(cfg_path), str(sid)])
    assert res.exit_code == 0, res.output
    assert "cancelled" in res.output


def test_abort_subcommand(tmp_path: Path) -> None:
    cfg_path, _ = _seed(tmp_path)
    res = CliRunner().invoke(cli, ["abort", str(cfg_path)])
    assert res.exit_code == 0, res.output
    assert "aborted" in res.output


def test_logs_subcommand_missing(tmp_path: Path) -> None:
    cfg_path, sid = _seed(tmp_path)
    res = CliRunner().invoke(cli, ["logs", str(cfg_path), str(sid)])
    # No folder yet → ClickException
    assert res.exit_code != 0
    assert "no log files" in res.output


def test_logs_subcommand_prints_files(tmp_path: Path) -> None:
    cfg_path, sid = _seed(tmp_path)
    # Give the sample a folder with logs
    workspace = tmp_path / "ws"
    layout = workspace_layout(workspace)
    store = SampleStore.open(layout["db"], f"cli-ext-{workspace.name}")
    sample = store.get(sid)
    sample.folder = workspace / "experiments" / "sim-0001"
    sample.folder.mkdir(parents=True)
    (sample.folder / "polaris.stdout.log").write_text("hello stdout")
    (sample.folder / "polaris.stderr.log").write_text("hello stderr")
    sample.status = SampleStatus.FINISHED
    sample.metric = np.array([0.25])
    store.update(sample)

    res = CliRunner().invoke(cli, ["logs", str(cfg_path), str(sid)])
    assert res.exit_code == 0, res.output
    assert "hello stdout" in res.output
    assert "hello stderr" in res.output
