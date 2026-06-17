"""Tests for the workspace_lock context manager + CLI integration."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from textwrap import dedent

import pytest
from click.testing import CliRunner

from polarisopt.cli import cli
from polarisopt.utils.workspace_lock import (
    LOCK_FILENAME,
    META_FILENAME,
    WorkspaceLockError,
    workspace_lock,
)


def test_lock_acquires_on_empty_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    with workspace_lock(workspace, action="run"):
        # Lock file exists during the block.
        assert (workspace / LOCK_FILENAME).exists()
        meta = json.loads((workspace / META_FILENAME).read_text())
        assert meta["action"] == "run"
        assert meta["pid"] == os.getpid()
        assert "hostname" in meta
        assert "started_at" in meta
        assert "version" in meta


def test_lock_releases_metadata_on_exit(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    with workspace_lock(workspace, action="run"):
        assert (workspace / META_FILENAME).exists()
    # Meta file is cleaned up on exit; lock file persists (cheap, harmless).
    assert not (workspace / META_FILENAME).exists()


def test_lock_blocks_concurrent_acquisition(tmp_path: Path) -> None:
    """A second flock attempt on the same workspace must fail.

    Same-process attempts work because flock is per-fd on Linux (separate
    open() calls give separate fds, and flock enforces between them).
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    # Hold the lock from an outside fd to simulate another master.
    fd = os.open(workspace / LOCK_FILENAME, os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Write the metadata the contender will see.
    (workspace / META_FILENAME).write_text(
        json.dumps({
            "pid": 99999,
            "hostname": "test-host",
            "started_at": "2026-06-17T00:00:00+00:00",
            "version": "0.12.1",
            "action": "run",
        })
    )
    try:
        with pytest.raises(WorkspaceLockError, match="another polarisopt master"), workspace_lock(workspace, action="resume"):
            pass
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_lock_contention_message_includes_holder_metadata(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    fd = os.open(workspace / LOCK_FILENAME, os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (workspace / META_FILENAME).write_text(
        json.dumps({
            "pid": 12345,
            "hostname": "xover-login1",
            "started_at": "2026-06-17T09:23:45+00:00",
            "version": "0.12.0",
            "action": "run",
        })
    )
    try:
        with pytest.raises(WorkspaceLockError) as exc_info, workspace_lock(workspace, action="resume"):
            pass
        msg = str(exc_info.value)
        assert "12345" in msg
        assert "xover-login1" in msg
        assert "2026-06-17T09:23:45" in msg
        assert "0.12.0" in msg
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_force_bypasses_contention(tmp_path: Path, caplog) -> None:
    """force=True skips the acquisition and proceeds with a warning."""
    import logging
    workspace = tmp_path / "ws"
    workspace.mkdir()
    fd = os.open(workspace / LOCK_FILENAME, os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (workspace / META_FILENAME).write_text(
        json.dumps({"pid": 12345, "hostname": "h", "started_at": "x", "version": "v", "action": "run"})
    )
    try:
        with caplog.at_level(logging.WARNING), workspace_lock(workspace, action="resume", force=True):
            # Block runs despite contention. No raise.
            pass
        assert any("bypassing workspace lock" in r.message for r in caplog.records)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_lock_serial_reacquire_works(tmp_path: Path) -> None:
    """Releasing and re-acquiring works fine (back-to-back run calls)."""
    workspace = tmp_path / "ws"
    with workspace_lock(workspace, action="run"):
        pass
    with workspace_lock(workspace, action="resume"):
        pass


# ---------- CLI integration ----------


def _yaml(workspace: Path) -> str:
    return dedent(
        f"""\
        name: lock-{workspace.name}
        workspace: {workspace}
        seed: 1
        simulator: {{ type: mock, options: {{ function: quadratic }} }}
        runner: {{ type: local, options: {{}} }}
        parameters:
          inline:
            - {{ name: x, file: a.json, min: 0.0, max: 1.0 }}
        metric: {{ type: identity, options: {{ keys: value }} }}
        phases:
          - name: p
            type: static
            design: {{ type: manual, options: {{ points: [[0.5]] }} }}
        """
    )


def test_cli_run_refuses_when_lock_held(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    fd = os.open(workspace / LOCK_FILENAME, os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (workspace / META_FILENAME).write_text(
        json.dumps({"pid": 1, "hostname": "h", "started_at": "x", "version": "v", "action": "run"})
    )
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_yaml(workspace))
    try:
        res = CliRunner().invoke(cli, ["run", str(cfg_path)])
        assert res.exit_code != 0
        assert "another polarisopt master" in res.output
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_cli_run_force_bypasses_lock(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    fd = os.open(workspace / LOCK_FILENAME, os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_yaml(workspace))
    try:
        res = CliRunner().invoke(cli, ["run", str(cfg_path), "--force"])
        assert res.exit_code == 0, res.output
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
