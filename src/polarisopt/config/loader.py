"""Load a study YAML with Jinja2 templating, then validate with pydantic."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, StrictUndefined

from polarisopt.config.schema import StudyConfig
from polarisopt.utils._compat import UTC


def _make_env() -> Environment:
    """Jinja2 env with helpful globals — env vars, dates, paths."""
    env = Environment(undefined=StrictUndefined, keep_trailing_newline=True)
    env.globals.update(
        env=os.environ,
        now=lambda fmt="%Y%m%dT%H%M%SZ": datetime.now(UTC).strftime(fmt),
    )
    return env


def render_yaml(text: str, context: dict[str, Any] | None = None) -> str:
    """Render YAML text through Jinja2.

    Globals available in templates:
      - ``env.<NAME>`` — environment variables
      - ``now('<strftime>')`` — current UTC timestamp
      - any keys passed in ``context``
    """
    env = _make_env()
    template = env.from_string(text)
    return template.render(**(context or {}))


def load_study_config(path: Path | str, *, context: dict[str, Any] | None = None) -> StudyConfig:
    """Load a study YAML at ``path``, render through Jinja, parse via pydantic.

    The validated :class:`StudyConfig` is returned. Plugin sections inside it
    are left as raw dicts; the orchestrator validates them when instantiating
    each plugin.
    """
    p = Path(path)
    raw = p.read_text()
    rendered = render_yaml(raw, context=context)
    data = yaml.safe_load(rendered)
    if not isinstance(data, dict):
        raise ValueError(f"{p}: top-level YAML must be a mapping, got {type(data).__name__}")
    return StudyConfig.model_validate(data)
