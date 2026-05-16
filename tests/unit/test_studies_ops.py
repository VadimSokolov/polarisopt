"""Tests for studies.ops — cancel / abort / log path helpers."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest

from polarisopt.config import load_study_config
from polarisopt.samples.sample import Sample, SampleStatus
from polarisopt.samples.store import SampleStore
from polarisopt.studies.ops import (
    abort_study,
    cancel_sample,
    open_store,
    sample_log_paths,
)
from polarisopt.utils.paths import workspace_layout


def _yaml(workspace: Path) -> str:
    return dedent(
        f"""\
        name: ops-{workspace.name}
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
              type: lhs
              options: {{ n: 1 }}
        """
    )


def _seed_store(workspace: Path, name: str) -> SampleStore:
    workspace.mkdir(parents=True, exist_ok=True)
    layout = workspace_layout(workspace)
    return SampleStore.open(layout["db"], name)


def test_open_store_requires_existing_db(tmp_path: Path) -> None:
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_yaml(tmp_path / "ws"))
    cfg = load_study_config(cfg_path)
    with pytest.raises(FileNotFoundError):
        open_store(cfg)


def test_cancel_pending_sample(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_yaml(workspace))
    cfg = load_study_config(cfg_path)
    store = _seed_store(workspace, cfg.name)
    sample = store.add(Sample(phase="p", inputs=np.array([0.5])))

    updated = cancel_sample(sample.id, config=cfg, store=store)  # type: ignore[arg-type]
    assert updated.status is SampleStatus.CANCELLED
    assert updated.message is not None and "cancelled" in updated.message


def test_cancel_terminal_is_noop(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_yaml(workspace))
    cfg = load_study_config(cfg_path)
    store = _seed_store(workspace, cfg.name)
    sample = store.add(Sample(phase="p", inputs=np.array([0.5])))
    sample.status = SampleStatus.FINISHED
    sample.metric = np.array([1.23])
    store.update(sample)

    updated = cancel_sample(sample.id, config=cfg, store=store)  # type: ignore[arg-type]
    assert updated.status is SampleStatus.FINISHED  # unchanged


def test_abort_cancels_all_non_terminal(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_yaml(workspace))
    cfg = load_study_config(cfg_path)
    store = _seed_store(workspace, cfg.name)

    s1 = store.add(Sample(phase="p", inputs=np.array([0.1])))  # pending
    s2 = store.add(Sample(phase="p", inputs=np.array([0.2])))
    s2.status = SampleStatus.RUNNING
    store.update(s2)
    s3 = store.add(Sample(phase="p", inputs=np.array([0.3])))
    s3.status = SampleStatus.FINISHED
    s3.metric = np.array([1.0])
    store.update(s3)

    cancelled = abort_study(cfg, store=store)
    cancelled_ids = {s.id for s in cancelled}
    assert cancelled_ids == {s1.id, s2.id}

    # post-conditions in DB
    s1_now = store.get(s1.id)  # type: ignore[arg-type]
    s2_now = store.get(s2.id)  # type: ignore[arg-type]
    s3_now = store.get(s3.id)  # type: ignore[arg-type]
    assert s1_now.status is SampleStatus.CANCELLED
    assert s2_now.status is SampleStatus.CANCELLED
    assert s3_now.status is SampleStatus.FINISHED


def test_sample_log_paths_empty(tmp_path: Path) -> None:
    s = Sample(phase="p", inputs=np.array([0.1]))
    s.folder = tmp_path / "empty"
    s.folder.mkdir()
    assert sample_log_paths(s) == []


def test_sample_log_paths_finds_files(tmp_path: Path) -> None:
    folder = tmp_path / "sim"
    folder.mkdir()
    (folder / "polaris.stdout.log").write_text("stdout")
    (folder / "polaris.stderr.log").write_text("stderr")
    (folder / "ignored.txt").write_text("not a log")
    s = Sample(phase="p", inputs=np.array([0.1]))
    s.folder = folder
    paths = sample_log_paths(s)
    names = sorted(p.name for p in paths)
    assert names == ["polaris.stderr.log", "polaris.stdout.log"]
