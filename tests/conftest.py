"""Shared pytest fixtures."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A clean temporary workspace directory per test."""
    return tmp_path


@pytest.fixture
def fake_shell() -> FakeShell:
    """Programmable fake for SlurmRunner.shell_runner."""
    return FakeShell()


class FakeShell:
    """Records calls and returns programmable subprocess-like results.

    Use ``.responses`` to enqueue per-call responses; if empty, the default
    is ``returncode=0, stdout="", stderr=""``. ``.calls`` lists the argv tuples.
    """

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.responses: list[subprocess.CompletedProcess[str]] = []
        self._handler: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None

    def set_handler(self, handler: Callable[[list[str]], subprocess.CompletedProcess[str]]) -> None:
        """Set a function that decides per-call responses based on argv."""
        self._handler = handler

    def __call__(self, cmd: list[str]) -> subprocess.CompletedProcess[str]:
        self.calls.append(cmd)
        if self._handler is not None:
            return self._handler(cmd)
        if self.responses:
            return self.responses.pop(0)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
