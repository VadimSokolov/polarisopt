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
from polarisopt.studies.ops import (
    EXTRA_FINGERPRINT_KEY,
    ConfigDriftError,
    retry_failed,
    simulator_config_fingerprint,
)
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


def test_retry_failed_refuses_when_config_drifts(tmp_path: Path) -> None:
    """retry_failed should refuse if the simulator config has changed since
    the failed samples ran.
    """
    workspace = tmp_path / "ws"
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_yaml(workspace))
    cfg = load_study_config(cfg_path)
    store, ids = _seed_with_failed(workspace, cfg.name)

    # Stamp the failed samples with a stale fingerprint (as if they ran
    # under an older config).
    for sid in ids[:2]:
        s = store.get(sid)
        s.extra[EXTRA_FINGERPRINT_KEY] = "deadbeef" * 2  # 16 chars, definitely not current
        store.update(s)

    current_fp = simulator_config_fingerprint(cfg)
    assert current_fp != "deadbeef" * 2

    with pytest.raises(ConfigDriftError, match="config has changed"):
        retry_failed(cfg, store=store)

    # --force overrides
    retried = retry_failed(cfg, store=store, force=True)
    assert len(retried) == 2


def test_retry_failed_no_drift_when_fingerprint_matches(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_yaml(workspace))
    cfg = load_study_config(cfg_path)
    store, ids = _seed_with_failed(workspace, cfg.name)

    current_fp = simulator_config_fingerprint(cfg)
    for sid in ids[:2]:
        s = store.get(sid)
        s.extra[EXTRA_FINGERPRINT_KEY] = current_fp
        store.update(s)

    retried = retry_failed(cfg, store=store)
    assert len(retried) == 2


def test_retry_failed_orchestrator_knobs_dont_invalidate_fingerprint(
    tmp_path: Path,
) -> None:
    """poll_interval / heartbeat_interval are orchestrator-only and must not
    invalidate the fingerprint.
    """
    workspace = tmp_path / "ws"
    cfg_path_a = tmp_path / "a.yaml"
    cfg_path_a.write_text(_yaml(workspace))
    cfg_a = load_study_config(cfg_path_a)

    # Same config, different orchestrator knobs only.
    yaml_b = _yaml(workspace).replace(
        "options: {}",
        "options: { poll_interval: 1.0, heartbeat_interval: 60.0, orphan_threshold: 5 }",
        1,
    )
    cfg_path_b = tmp_path / "b.yaml"
    cfg_path_b.write_text(yaml_b)
    cfg_b = load_study_config(cfg_path_b)

    assert simulator_config_fingerprint(cfg_a) == simulator_config_fingerprint(cfg_b)


def test_cli_retry_failed_force_flag(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_yaml(workspace))
    cfg = load_study_config(cfg_path)
    store, ids = _seed_with_failed(workspace, cfg.name)
    for sid in ids[:2]:
        s = store.get(sid)
        s.extra[EXTRA_FINGERPRINT_KEY] = "stale-fingerprint"
        store.update(s)

    # Without --force: errors out.
    res_no = CliRunner().invoke(cli, ["retry-failed", str(cfg_path)])
    assert res_no.exit_code != 0
    assert "config has changed" in res_no.output
    assert "--force" in res_no.output

    # With --force: succeeds.
    res = CliRunner().invoke(cli, ["retry-failed", str(cfg_path), "--force"])
    assert res.exit_code == 0, res.output
    assert "retried 2 sample(s)" in res.output


def test_retry_failed_drift_message_shows_field_level_diff(tmp_path: Path) -> None:
    """v0.17: ConfigDriftError includes a recorded → current diff when
    the sample has a config_snapshot in extra. Previously you only got
    the hash and had to guess what changed.
    """
    from polarisopt.studies.ops import EXTRA_CONFIG_SNAPSHOT_KEY

    workspace = tmp_path / "ws"
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_yaml(workspace))
    cfg = load_study_config(cfg_path)
    store, ids = _seed_with_failed(workspace, cfg.name)
    # Stamp the first failed sample with a snapshot that differs from
    # the current YAML in a recognizable field.
    s = store.get(ids[0])
    s.extra[EXTRA_FINGERPRINT_KEY] = "stale-fingerprint"
    s.extra[EXTRA_CONFIG_SNAPSHOT_KEY] = {
        "simulator": {
            "type": "mock",
            "options": {"function": "rosenbrock"},  # current YAML says quadratic
        },
        "runner": {"type": "local", "options": {}},
    }
    store.update(s)

    with pytest.raises(ConfigDriftError) as exc_info:
        retry_failed(cfg, store=store)
    msg = str(exc_info.value)
    # Diff shows the actual field that changed.
    assert "simulator.options.function" in msg
    assert "rosenbrock" in msg
    assert "quadratic" in msg


def test_retry_failed_drift_diff_recurses_into_nested_dicts(tmp_path: Path) -> None:
    """v0.17 (CodeRabbit fix): the diff must descend into nested options
    like default_resources, not blob-compare whole sub-dicts."""
    from polarisopt.studies.ops import _diff_config_snapshots

    recorded = {
        "simulator": {"type": "mock", "options": {"function": "quadratic"}},
        "runner": {
            "type": "slurm",
            "options": {
                "default_resources": {
                    "partition": "bdwall",
                    "mem": "64G",
                    "time": "01:00:00",
                },
            },
        },
    }
    current = {
        "simulator": {"type": "mock", "options": {"function": "quadratic"}},
        "runner": {
            "type": "slurm",
            "options": {
                "default_resources": {
                    "partition": "TPS",      # changed
                    "mem": "64G",            # unchanged
                    "time": "02:00:00",      # changed
                    "exclusive": True,       # added
                    # "account" not present — would be (added)
                },
            },
        },
    }
    diff = _diff_config_snapshots(recorded, current)
    # Each leaf gets its own line — not a single blob diff on default_resources.
    assert any("default_resources.partition" in line and "TPS" in line for line in diff)
    assert any("default_resources.time" in line and "02:00:00" in line for line in diff)
    assert any("default_resources.exclusive" in line and "added" in line for line in diff)
    # Unchanged leaves are NOT in the diff.
    assert not any("default_resources.mem" in line for line in diff)


def test_retry_failed_drift_message_falls_back_when_no_snapshot(tmp_path: Path) -> None:
    """Pre-v0.17 samples have only a fingerprint, no snapshot. Drift
    error still works — just with a hash-only message, plus a hint
    explaining why no diff is available."""
    workspace = tmp_path / "ws"
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_yaml(workspace))
    cfg = load_study_config(cfg_path)
    store, ids = _seed_with_failed(workspace, cfg.name)
    for sid in ids[:2]:
        s = store.get(sid)
        s.extra[EXTRA_FINGERPRINT_KEY] = "stale"  # no snapshot
        store.update(s)

    with pytest.raises(ConfigDriftError) as exc_info:
        retry_failed(cfg, store=store)
    msg = str(exc_info.value)
    assert "stale" in msg
    assert "No field-level diff" in msg


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
