"""End-to-end: load a YAML study, run it, verify results in the SampleStore."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import numpy as np
from click.testing import CliRunner

from polarisopt.cli import cli
from polarisopt.config import load_study_config
from polarisopt.samples.sample import SampleStatus
from polarisopt.samples.store import SampleStore
from polarisopt.studies.runner import StudyRunner
from polarisopt.utils.paths import workspace_layout


def _branin_yaml(workspace: Path) -> str:
    return dedent(
        f"""\
        name: branin-screen-{workspace.name}
        workspace: {workspace}
        seed: 7
        simulator:
          type: mock
          options:
            function: branin
        runner:
          type: local
          options: {{}}
        parameters:
          inline:
            - {{ name: x1, file: a.json, min: -5.0, max: 10.0 }}
            - {{ name: x2, file: a.json, min: 0.0, max: 15.0 }}
        metric:
          type: identity
          options:
            keys: value
        phases:
          - name: lhs-screen
            type: static
            design:
              type: lhs
              options:
                n: 6
        """
    )


def test_run_yaml_end_to_end(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    config_path = tmp_path / "study.yaml"
    config_path.write_text(_branin_yaml(workspace))

    config = load_study_config(config_path)
    samples = StudyRunner(config).run()

    assert len(samples) == 6
    assert all(s.status is SampleStatus.FINISHED for s in samples)
    assert all(s.metric is not None and s.metric.shape == (1,) for s in samples)

    layout = workspace_layout(workspace)
    assert layout["db"].exists()
    store = SampleStore.open(layout["db"], config.name)
    df = store.to_dataframe()
    assert len(df) == 6
    assert (df["phase"] == "lhs-screen").all()
    assert (df["status"] == "finished").all()
    # All metrics finite
    arrs = [np.asarray(m) for m in df["metric"].tolist()]
    assert all(np.isfinite(a).all() for a in arrs)


def test_run_yaml_morris(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    config_text = _branin_yaml(workspace).replace(
        "type: lhs\n              options:\n                n: 6",
        "type: morris\n              options:\n                n_trajectories: 2\n                num_levels: 4",
    )
    config_path = tmp_path / "study.yaml"
    config_path.write_text(config_text)
    config = load_study_config(config_path)
    samples = StudyRunner(config).run()
    # Morris emits N*(d+1) = 2*(2+1) = 6 rows
    assert len(samples) == 6
    assert all(s.status is SampleStatus.FINISHED for s in samples)


def test_cli_run_and_status(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    config_path = tmp_path / "study.yaml"
    config_path.write_text(_branin_yaml(workspace))

    runner = CliRunner()
    result = runner.invoke(cli, ["run", str(config_path)])
    assert result.exit_code == 0, result.output
    assert "completed: 6/6" in result.output

    result = runner.invoke(cli, ["status", str(config_path)])
    assert result.exit_code == 0, result.output
    assert "finished" in result.output


def test_cli_resume_after_partial_run(tmp_path: Path) -> None:
    """Inject pending samples into the store, then resume — they should evaluate."""
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True)
    config_path = tmp_path / "study.yaml"
    config_path.write_text(_branin_yaml(workspace))

    config = load_study_config(config_path)
    layout = workspace_layout(workspace)
    layout["root"].mkdir(parents=True, exist_ok=True)

    store = SampleStore.open(layout["db"], config.name)
    from polarisopt.samples.sample import Sample

    store.add(Sample(phase="lhs-screen", inputs=np.array([0.0, 0.0])))
    store.add(Sample(phase="lhs-screen", inputs=np.array([np.pi, 2.275])))

    cli_runner = CliRunner()
    result = cli_runner.invoke(cli, ["resume", str(config_path)])
    assert result.exit_code == 0, result.output

    store = SampleStore.open(layout["db"], config.name)
    rows = store.list()
    assert len(rows) == 2
    assert all(r.status is SampleStatus.FINISHED for r in rows)
