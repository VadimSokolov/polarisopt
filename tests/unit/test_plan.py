"""Tests for studies.plan + the plan CLI subcommand + validate --deep."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from click.testing import CliRunner

from polarisopt.cli import cli
from polarisopt.studies.plan import plan_study


def _good_yaml(workspace: Path) -> str:
    return dedent(
        f"""\
        name: plan-{workspace.name}
        workspace: {workspace}
        seed: 1
        simulator: {{ type: mock, options: {{ function: branin }} }}
        runner: {{ type: local, options: {{}} }}
        parameters:
          inline:
            - {{ name: x1, file: dummy.json, min: -5.0, max: 10.0 }}
            - {{ name: x2, file: dummy.json, min:  0.0, max: 15.0 }}
        metric: {{ type: identity, options: {{ keys: value }} }}
        phases:
          - name: bo
            type: static
            design: {{ type: lhs, options: {{ n: 4 }} }}
        """
    )


def test_plan_passes_for_good_yaml(tmp_path: Path) -> None:
    p = tmp_path / "good.yaml"
    p.write_text(_good_yaml(tmp_path / "ws"))
    report = plan_study(p)
    assert report.ok, report.render()
    assert report.workspace is not None
    assert report.workspace.exists()
    assert report.job_spec is not None
    assert "polarisopt.simulator._mock_runner" in report.job_spec.command


def test_plan_reports_simulator_prepare_failure(tmp_path: Path) -> None:
    # Point at a polaris simulator with a model_source that doesn't exist
    yaml_text = _good_yaml(tmp_path / "ws").replace(
        "simulator: { type: mock, options: { function: branin } }",
        dedent(
            """\
            simulator:
              type: polaris
              options:
                binary: /usr/bin/echo
                model_source: /does/not/exist/model
                scenario_file: scenario_abm.json
                output_db_filename: R.h5
            """
        ).strip(),
    )
    p = tmp_path / "bad.yaml"
    p.write_text(yaml_text)
    report = plan_study(p)
    assert not report.ok
    assert any("model_source" in e for e in report.errors)


def test_plan_slurm_runner_writes_script(tmp_path: Path) -> None:
    yaml_text = _good_yaml(tmp_path / "ws").replace(
        "runner: { type: local, options: {} }",
        dedent(
            """\
            runner:
              type: slurm
              options:
                default_resources:
                  partition: bdwall
                  account: POLARIS
                  time: "00:30:00"
                  cpus_per_task: 4
                  setup_commands:
                    - "module purge"
                    - "module load singularity"
            """
        ).strip(),
    )
    p = tmp_path / "slurm.yaml"
    p.write_text(yaml_text)
    report = plan_study(p)
    assert report.ok, report.render()
    assert report.script_path is not None
    text = report.script_path.read_text()
    # Directives must come before any executable line
    sbatch_partition_idx = text.find("#SBATCH --partition")
    setup_idx = text.find("module purge")
    pipefail_idx = text.find("set -euo pipefail")
    assert sbatch_partition_idx >= 0
    assert setup_idx > pipefail_idx > sbatch_partition_idx, text
    # setup_commands must render in the script
    assert "module load singularity" in text


def test_plan_strips_orchestrator_knobs_before_runner_construction(
    tmp_path: Path,
) -> None:
    """``poll_interval`` / ``orphan_threshold`` / ``heartbeat_interval`` live
    in ``runner.options`` (YAML-side) but are consumed by ``StudyRunner``,
    not by the runner constructor. ``plan_study`` must strip them too —
    otherwise it errors on every YAML that ``polarisopt run`` accepts.

    Regression for the bundled ``polaris-slurm.yaml`` example which sets
    these and was failing ``polarisopt plan`` as shipped.
    """
    yaml_text = _good_yaml(tmp_path / "ws").replace(
        "runner: { type: local, options: {} }",
        dedent(
            """\
            runner:
              type: slurm
              options:
                default_resources:
                  partition: bdwall
                  account: POLARIS
                  time: "00:30:00"
                  cpus_per_task: 4
                poll_interval: 30
                orphan_threshold: 5
                heartbeat_interval: 60
            """
        ).strip(),
    )
    p = tmp_path / "slurm-with-knobs.yaml"
    p.write_text(yaml_text)
    report = plan_study(p)
    assert report.ok, report.render()
    assert report.script_path is not None


def test_plan_works_on_bundled_polaris_slurm_example(tmp_path: Path) -> None:
    """Regression: the example shipped under src/polarisopt/examples/ must
    pass ``polarisopt plan`` so users who follow the docs aren't stuck."""
    from polarisopt.examples import read_example

    yaml_text = read_example("polaris-slurm")
    # The example uses Jinja env() lookups + a workspace path under /lcrc/
    # that doesn't exist on CI. Patch a writable workspace and stub the
    # env vars enough to render.
    yaml_text = yaml_text.replace(
        "/lcrc/project/POLARIS/{{ env.USER }}/runs/calib-{{ now('%Y%m%dT%H%M%SZ') }}",
        str(tmp_path / "ws"),
    )
    fake_bin = tmp_path / "polaris.sif"
    fake_bin.touch()
    fake_model = tmp_path / "model"
    fake_model.mkdir()
    (fake_model / "scenario_abm.json").write_text(
        '{"Output controls": {"output_dir_name": "out"}, '
        '"General simulation controls": {"database_name": "T"}}'
    )
    (fake_model / "DestinationChoice.json").write_text(
        '{"_": {"trip_threshold": 0.0, "gravity_alpha": 0.0}}'
    )
    (fake_model / "ActivityChoice.json").write_text('{"_": {"walk_max": 0}}')
    fake_target = tmp_path / "target.h5"
    fake_target.touch()
    yaml_text = (
        yaml_text
        .replace("{{ env.POLARIS_BIN }}", str(fake_bin))
        .replace("{{ env.POLARIS_MODEL }}", str(fake_model))
        .replace("{{ env.POLARIS_TARGET_H5 }}", str(fake_target))
    )
    p = tmp_path / "polaris-slurm.yaml"
    p.write_text(yaml_text)
    report = plan_study(p)
    # The example may produce warnings (e.g. unknown runner_options for
    # branch-specific knobs) but must not error.
    assert report.ok, report.render()


def test_cli_plan(tmp_path: Path) -> None:
    p = tmp_path / "good.yaml"
    p.write_text(_good_yaml(tmp_path / "ws"))
    res = CliRunner().invoke(cli, ["plan", str(p)])
    assert res.exit_code == 0, res.output
    assert "plan ok" in res.output


def test_cli_validate_deep(tmp_path: Path) -> None:
    p = tmp_path / "good.yaml"
    p.write_text(_good_yaml(tmp_path / "ws"))
    res = CliRunner().invoke(cli, ["validate", str(p), "--deep"])
    assert res.exit_code == 0, res.output
    assert "validation passed" in res.output
    assert "running deep validation" in res.output
    assert "plan ok" in res.output


def test_cli_validate_deep_catches_simulator_failure(tmp_path: Path) -> None:
    yaml_text = _good_yaml(tmp_path / "ws").replace(
        "simulator: { type: mock, options: { function: branin } }",
        dedent(
            """\
            simulator:
              type: polaris
              options:
                binary: /usr/bin/echo
                model_source: /does/not/exist
                scenario_file: scenario_abm.json
                output_db_filename: R.h5
            """
        ).strip(),
    )
    p = tmp_path / "bad.yaml"
    p.write_text(yaml_text)
    # shallow validate succeeds (only a warning) — but --deep should catch
    # the missing model_source at prepare() time.
    res = CliRunner().invoke(cli, ["validate", str(p), "--deep"])
    assert res.exit_code != 0
