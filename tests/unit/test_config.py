from __future__ import annotations

import os
from textwrap import dedent

import pytest

from polarisopt.config import load_study_config, render_yaml
from polarisopt.config.schema import StaticPhaseConfig


def _study_yaml() -> str:
    return dedent(
        """\
        name: my-study
        workspace: /tmp/ws-{{ now('%Y%m%d') }}
        seed: 42
        simulator:
          type: mock
          options: {}
        runner:
          type: local
          options: {}
        parameters:
          inline:
            - { name: a, file: x.json, min: 0.0, max: 1.0 }
        metric:
          type: mock_metric
          options: {}
        phases:
          - name: warmup
            type: static
            design:
              type: lhs
              options: { n: 10 }
        """
    )


def test_jinja_env_global_and_now() -> None:
    os.environ["POLARISOPT_TEST_VAR"] = "hello"
    rendered = render_yaml("greeting: {{ env.POLARISOPT_TEST_VAR }}\nts: {{ now('%Y') }}")
    assert "greeting: hello" in rendered
    assert "ts: 20" in rendered  # year starts with 20xx for the foreseeable future


def test_load_minimal_study(tmp_path) -> None:
    p = tmp_path / "study.yaml"
    p.write_text(_study_yaml())
    cfg = load_study_config(p)
    assert cfg.name == "my-study"
    assert cfg.seed == 42
    assert len(cfg.phases) == 1
    phase = cfg.phases[0]
    assert isinstance(phase, StaticPhaseConfig)
    assert phase.design.type == "lhs"
    assert phase.design.options == {"n": 10}


def test_parameters_requires_exactly_one_source(tmp_path) -> None:
    yaml_text = _study_yaml().replace(
        "parameters:\n  inline:",
        "parameters:\n  source: ./p.yaml\n  inline:",
    )
    assert "source: ./p.yaml" in yaml_text  # guard: replacement actually happened
    p = tmp_path / "study.yaml"
    p.write_text(yaml_text)
    with pytest.raises(ValueError, match="exactly one"):
        load_study_config(p)


def test_parameters_requires_at_least_one_source(tmp_path) -> None:
    yaml_text = _study_yaml().replace(
        "parameters:\n  inline:\n    - { name: a, file: x.json, min: 0.0, max: 1.0 }",
        "parameters: {}",
    )
    assert "parameters: {}" in yaml_text
    p = tmp_path / "study.yaml"
    p.write_text(yaml_text)
    with pytest.raises(ValueError, match="exactly one"):
        load_study_config(p)


def test_extra_top_level_forbidden(tmp_path) -> None:
    from pydantic import ValidationError

    p = tmp_path / "study.yaml"
    p.write_text(_study_yaml() + "\nextra_unknown_key: yes\n")
    with pytest.raises(ValidationError):
        load_study_config(p)
