"""Tests for the validate subcommand and the validate_study() helper."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from click.testing import CliRunner

from polarisopt.cli import cli
from polarisopt.studies.validate import validate_study


def _good_yaml(workspace: Path) -> str:
    return dedent(
        f"""\
        name: validate-{workspace.name}
        workspace: {workspace}
        simulator: {{ type: mock, options: {{ function: branin }} }}
        runner: {{ type: local, options: {{}} }}
        parameters:
          inline:
            - {{ name: x1, file: dummy.json, min: -5.0, max: 10.0 }}
            - {{ name: x2, file: dummy.json, min:  0.0, max: 15.0 }}
        metric:
          type: identity
          options: {{ keys: value }}
        phases:
          - name: bo
            type: sequential
            warm_up: {{ type: lhs, options: {{ n: 4 }} }}
            generator:
              type: acquisition
              options:
                surrogate:   {{ type: gp,  options: {{}} }}
                acquisition: {{ type: qei, options: {{ mc_samples: 32 }} }}
            batch_size: 2
            stop: {{ type: max_iter, options: {{ n: 3 }} }}
        """
    )


def test_validate_passes_on_good_yaml(tmp_path: Path) -> None:
    p = tmp_path / "good.yaml"
    p.write_text(_good_yaml(tmp_path / "ws"))
    report = validate_study(p)
    assert report.ok
    assert any("parameters: 2" in info for info in report.info)


def test_validate_reports_unknown_simulator(tmp_path: Path) -> None:
    text = _good_yaml(tmp_path / "ws").replace("type: mock", "type: not_a_thing")
    p = tmp_path / "bad.yaml"
    p.write_text(text)
    report = validate_study(p)
    assert not report.ok
    assert any("simulator 'not_a_thing'" in e for e in report.errors)


def test_validate_reports_unknown_design(tmp_path: Path) -> None:
    text = _good_yaml(tmp_path / "ws").replace("type: lhs", "type: not_a_design")
    p = tmp_path / "bad.yaml"
    p.write_text(text)
    report = validate_study(p)
    assert not report.ok
    assert any("warm_up design 'not_a_design'" in e for e in report.errors)


def test_validate_warns_on_missing_polaris_binary(tmp_path: Path) -> None:
    yaml = _good_yaml(tmp_path / "ws").replace(
        "simulator: { type: mock, options: { function: branin } }",
        dedent(
            """\
            simulator:
              type: polaris
              options:
                binary: /does/not/exist/Integrated_Model.sif
                model_source: /does/not/exist/model
                scenario_file: scenario_abm.json
                output_db_filename: R.h5
                num_threads: "16"
            """
        ).strip(),
    )
    p = tmp_path / "bad.yaml"
    p.write_text(yaml)
    report = validate_study(p)
    # Schema is fine but two path warnings
    assert report.ok
    assert any("simulator.binary not found" in w for w in report.warnings)
    assert any("simulator.model_source not found" in w for w in report.warnings)


def test_validate_handles_nonexistent_file(tmp_path: Path) -> None:
    report = validate_study(tmp_path / "missing.yaml")
    assert not report.ok
    assert any("does not exist" in e for e in report.errors)


def test_validate_handles_malformed_yaml(tmp_path: Path) -> None:
    p = tmp_path / "broken.yaml"
    p.write_text("name: foo\nworkspace: /tmp\n  bad-indent: x\n")
    report = validate_study(p)
    assert not report.ok
    assert any("config load failed" in e for e in report.errors)


def test_cli_validate_good(tmp_path: Path) -> None:
    p = tmp_path / "good.yaml"
    p.write_text(_good_yaml(tmp_path / "ws"))
    res = CliRunner().invoke(cli, ["validate", str(p)])
    assert res.exit_code == 0, res.output
    assert "validation passed" in res.output


def test_cli_validate_bad_exits_nonzero(tmp_path: Path) -> None:
    text = _good_yaml(tmp_path / "ws").replace("type: mock", "type: nope")
    p = tmp_path / "bad.yaml"
    p.write_text(text)
    res = CliRunner().invoke(cli, ["validate", str(p)])
    assert res.exit_code != 0
    assert "nope" in res.output


def test_cli_validate_warnings_as_errors(tmp_path: Path) -> None:
    """Warnings-as-errors flag should turn a successful warning into a nonzero exit."""
    yaml = _good_yaml(tmp_path / "ws").replace(
        "simulator: { type: mock, options: { function: branin } }",
        dedent(
            """\
            simulator:
              type: polaris
              options:
                binary: /does/not/exist/Integrated_Model.sif
                model_source: /does/not/exist/model
                scenario_file: scenario_abm.json
                output_db_filename: R.h5
                num_threads: "16"
            """
        ).strip(),
    )
    p = tmp_path / "warn.yaml"
    p.write_text(yaml)
    res = CliRunner().invoke(cli, ["validate", str(p), "--warnings-as-errors"])
    assert res.exit_code != 0
    # Same yaml without the flag should succeed (only warnings)
    res2 = CliRunner().invoke(cli, ["validate", str(p)])
    assert res2.exit_code == 0
