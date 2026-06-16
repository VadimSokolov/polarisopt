"""Tests for the bundled example studies and the ``polarisopt examples`` CLI."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from polarisopt.cli import cli
from polarisopt.config.loader import render_yaml
from polarisopt.examples import example_path, list_examples, read_example


def test_list_examples_nonempty() -> None:
    names = list_examples()
    # We ship at least the four core examples.
    assert {"branin", "morris", "multi-objective", "polaris-slurm"}.issubset(names)


def test_example_path_unknown_raises() -> None:
    with pytest.raises(FileNotFoundError):
        example_path("does-not-exist")


def test_each_example_parses_after_jinja_render() -> None:
    """Every bundled YAML should survive Jinja2 templating without errors.

    We set up minimal env vars so the polaris-slurm.yaml renders cleanly.
    """
    import os

    os.environ.setdefault("POLARIS_BIN", "/tmp/fake_bin")
    os.environ.setdefault("POLARIS_MODEL", "/tmp/fake_model")
    os.environ.setdefault("POLARIS_TARGET_H5", "/tmp/fake_target.h5")

    import yaml

    for name in list_examples():
        text = read_example(name)
        rendered = render_yaml(text)
        # Should parse as YAML.
        parsed = yaml.safe_load(rendered)
        assert isinstance(parsed, dict), f"{name} did not parse as a dict"
        assert "name" in parsed and "phases" in parsed, f"{name} missing required keys"


def test_cli_examples_list() -> None:
    res = CliRunner().invoke(cli, ["examples", "list"])
    assert res.exit_code == 0, res.output
    for name in ("branin", "morris", "multi-objective", "polaris-slurm"):
        assert name in res.output


def test_cli_examples_show() -> None:
    res = CliRunner().invoke(cli, ["examples", "show", "branin"])
    assert res.exit_code == 0
    assert "name: branin-example" in res.output
    assert "simulator:" in res.output


def test_cli_examples_show_unknown() -> None:
    res = CliRunner().invoke(cli, ["examples", "show", "bogus"])
    assert res.exit_code != 0
    assert "unknown example" in res.output


def test_cli_examples_copy(tmp_path: Path) -> None:
    dst = tmp_path / "my-study.yaml"
    res = CliRunner().invoke(cli, ["examples", "copy", "branin", str(dst)])
    assert res.exit_code == 0, res.output
    assert dst.exists()
    assert "name: branin-example" in dst.read_text()


def test_cli_examples_copy_no_overwrite(tmp_path: Path) -> None:
    dst = tmp_path / "existing.yaml"
    dst.write_text("dont touch me")
    res = CliRunner().invoke(cli, ["examples", "copy", "branin", str(dst)])
    assert res.exit_code != 0
    assert "already exists" in res.output
    assert dst.read_text() == "dont touch me"


def test_cli_examples_copy_force(tmp_path: Path) -> None:
    dst = tmp_path / "existing.yaml"
    dst.write_text("dont touch me")
    res = CliRunner().invoke(cli, ["examples", "copy", "branin", str(dst), "--force"])
    assert res.exit_code == 0
    assert "name: branin-example" in dst.read_text()
